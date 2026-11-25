"""Ordinal compact IQA and an exploratory conditional degradation constraint.

Distribution targets interpolate the observed MOS; they are not reader votes.
Synthetic ranking constraints are research assumptions, not clinical labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur

from research_audit.compact_iqa import CompactIQA, Images, save_predictions, predict
from ct_musiq.loss import CTMUSIQLoss
from ct_musiq.train import set_seed
from scipy.stats import pearsonr, spearmanr, kendalltau


class OrdinalIQA(CompactIQA):
    def __init__(self, pretrained_file=None, backbone='efficientnet_b0'):
        super().__init__(pretrained_file,backbone=backbone)
        self.head[-1] = nn.Linear(256, 21)
        self.register_buffer('bins', torch.linspace(0,4,21))

    def distribution(self, images):
        return self.head(self.backbone((images-self.mean)/self.std)).float().softmax(-1)

    def forward(self, images):
        return (self.distribution(images)*self.bins).sum(-1,keepdim=True)


def ordinal_loss(probabilities, targets):
    locations = targets.clamp(0,4)*5
    lower = locations.floor().long()
    upper = (lower+1).clamp(max=20)
    fraction = locations-lower
    soft = torch.zeros_like(probabilities)
    soft.scatter_add_(1,lower[:,None],(1-fraction)[:,None])
    soft.scatter_add_(1,upper[:,None],fraction[:,None])
    return ((probabilities.cumsum(-1)[:,:-1]-soft.cumsum(-1)[:,:-1])**2).mean()


def metrics(pred,target):
    values=[float(fn(pred,target).statistic) for fn in (pearsonr,spearmanr,kendalltau)]
    return dict(zip(['PLCC','SROCC','KROCC'],values),Aggregate=sum(abs(x) for x in values),
                mae=float(np.abs(pred-target).mean()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image-dir',required=True)
    p.add_argument('--labels',required=True)
    p.add_argument('--split')
    p.add_argument('--pretrained-file')
    p.add_argument('--checkpoint')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--degradation-weight',type=float,default=0)
    p.add_argument('--constraint-sigma',type=float,default=1.5)
    p.add_argument('--head',choices=['scalar','ordinal'],default='ordinal')
    p.add_argument('--backbone',choices=['efficientnet_b0','resnet18'],default='efficientnet_b0')
    p.add_argument('--quality-threshold',type=float,default=3.)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    if not args.checkpoint and (not args.split or not args.pretrained_file):
        p.error('Training requires a split and offline pretrained weights')
    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels=json.loads(Path(args.labels).read_text())
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True) if args.checkpoint else None
    head=checkpoint['config'].get('head','ordinal') if checkpoint else args.head
    backbone=checkpoint['config'].get('backbone','efficientnet_b0') if checkpoint else args.backbone
    model=(CompactIQA if head=='scalar' else OrdinalIQA)(None if checkpoint else args.pretrained_file,backbone=backbone).to(device)
    out.mkdir(parents=True)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        loader=DataLoader(Images(args.image_dir,labels,sorted(labels),448),batch_size=args.batch_size,num_workers=args.workers)
        names,pred,target=predict(model,loader,device)
        result={**metrics(pred,target),'n':len(names),'checkpoint_config':checkpoint['config'],
                'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}
        save_predictions(out/'predictions.csv',names,pred,target)
        (out/'metrics.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return
    split=json.loads(Path(args.split).read_text())
    if (len(set(split['train']))!=len(split['train']) or len(set(split['val']))!=len(split['val'])
        or set(split['train'])&set(split['val']) or set(split['train']+split['val'])!=set(labels)):
        raise ValueError('Invalid development partition')
    train=DataLoader(Images(args.image_dir,labels,split['train'],448,True),batch_size=args.batch_size,
                     shuffle=True,num_workers=args.workers,pin_memory=True)
    val=DataLoader(Images(args.image_dir,labels,split['val'],448),batch_size=args.batch_size,num_workers=args.workers)
    rank=CTMUSIQLoss(lambda_kl=0,lambda_rank=.5).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    config={**vars(args),'model':f'{backbone}_{head}','resolution':448,'rps_weight':0. if head=='scalar' else 1.,
            'constraint':f'blur sigma {args.constraint_sigma} on training MOS>={args.quality_threshold}; margin 0.1; synthetic assumption',
            'split_sha256':hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
            'pretrained_sha256':hashlib.sha256(Path(args.pretrained_file).read_bytes()).hexdigest()}
    best,patience=float('-inf'),0
    with (out/'training_log.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','train_loss','constraint_loss','PLCC','SROCC','KROCC','Aggregate','mae'])
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            total,count,aux_total=0.,0,0.
            for images,scores,_ in train:
                images,scores=images.to(device),scores.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                    if head=='scalar':
                        pred=model(images).flatten()
                        distribution_loss=torch.zeros((),device=device)
                    else:
                        probs=model.distribution(images)
                        pred=(probs*model.bins).sum(-1)
                        distribution_loss=ordinal_loss(probs,scores)
                    loss=rank({'score':pred[:,None],'scale_scores':[]},scores)['total']+distribution_loss
                    aux=torch.zeros((),device=device)
                    mask=scores>=args.quality_threshold
                    if args.degradation_weight and mask.any():
                        # Freeze auxiliary BN statistics and disable dropout so
                        # extra synthetic forwards do not update the running BN
                        # estimates. Gradients still flow through both scores.
                        original=images[mask]
                        model.eval()
                        kernel=2*int(np.ceil(3*args.constraint_sigma))+1
                        pair=model(torch.cat([original,gaussian_blur(original,[kernel,kernel],[args.constraint_sigma,args.constraint_sigma])])).flatten()
                        model.train()
                        first,second=pair.chunk(2)
                        aux=F.relu(.1-(first-second)).mean()
                        loss=loss+args.degradation_weight*aux
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update()
                total+=loss.item()*len(scores); aux_total+=aux.item()*len(scores); count+=len(scores)
            scheduler.step()
            names,pred,target=predict(model,val,device)
            row={'epoch':epoch+1,'train_loss':total/count,'constraint_loss':aux_total/count,**metrics(pred,target)}
            writer.writerow(row); stream.flush(); print(json.dumps(row),flush=True)
            score=sum(row[k] for k in ['PLCC','SROCC','KROCC'])
            if score>best:
                best,patience=score,0
                torch.save({'model_state_dict':model.state_dict(),'config':config,'epoch':epoch,'validation':row},out/'best.pth')
                save_predictions(out/'validation_predictions.csv',names,pred,target)
            else: patience+=1
            if epoch>=19 and patience>=20: break


if __name__=='__main__': main()
