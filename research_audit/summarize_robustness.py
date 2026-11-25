"""Generate manuscript tables and figures from all declared robustness arms."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--require-complete',action='store_true'); args=parser.parse_args()
    root=Path('research_audit/results'); paper=Path('paper'); (paper/'images').mkdir(exist_ok=True)
    arms=['compact','rps','conditional','mild']
    names=['Scalar','Ordinal','Ordinal + blur 1.5','Ordinal + blur 0.75']
    extra_stages={'scalar_cond':'scalar_stage','unconditional':'unconditional_stage',
                  'resnet_baseline':'resnet_stage','resnet_cond':'resnet_stage',
                  'strong_unconditional':'strong_unconditional_stage'}
    extra_names={'scalar_cond':'Scalar + conditional blur 0.75','unconditional':'Scalar + unconditional blur 0.75',
                 'resnet_baseline':'ResNet-18 scalar','resnet_cond':'ResNet-18 + conditional blur 0.75',
                 'strong_unconditional':'Ordinal + unconditional blur 1.5'}
    pending=[]
    for arm,stage in extra_stages.items():
        if all((root/stage/'evaluations'/f'ordinal_{arm}_seed{s}'/'metrics.json').exists() for s in [42,123,456]):
            arms.append(arm); names.append(extra_names[arm])
        else: pending.append(arm)
    result={'warning':'Seed SD describes training variation, not patient-level confidence.',
            'scoring':'Recomputed from continuous saved predictions; no component rounding.',
            'pending_arms':pending,'arms':{}}
    paired_ids=paired_targets=None
    for arm in arms:
        stage=extra_stages.get(arm,'screening' if arm=='compact' else 'mild_stage' if arm=='mild' else 'ordinal_stage')
        scores=[]; probes=[]
        for seed in [42,123,456]:
            directory=f'compact448_seed{seed}' if arm=='compact' else f'ordinal_{arm}_seed{seed}'
            records=sorted(csv.DictReader((root/stage/'evaluations'/directory/'predictions.csv').open(newline='')),key=lambda r:r['image_id'])
            ids=[r['image_id'] for r in records]; pred=np.array([float(r['predicted']) for r in records]); target=np.array([float(r['target']) for r in records])
            if len(ids)!=300 or len(set(ids))!=300 or not np.isfinite([pred,target]).all(): raise ValueError('Invalid LDCT predictions')
            if paired_ids is None: paired_ids,paired_targets=ids,target
            if ids!=paired_ids or not np.allclose(target,paired_targets,rtol=0,atol=1e-6): raise ValueError('Unpaired LDCT outcomes')
            values=[float(fn(pred,target).statistic) for fn in [pearsonr,spearmanr,kendalltau]]
            scores.append(dict(zip(['PLCC','SROCC','KROCC'],values),Aggregate=sum(abs(x) for x in values),mae=float(np.abs(pred-target).mean())))
            probe_stage='ordinal_stage' if arm=='compact' else stage
            probes.append(json.loads((root/probe_stage/'probes'/f'{arm}_seed{seed}'/'metrics.json').read_text())['summary'])
        entry={'per_seed_test':dict(zip(['42','123','456'],scores)),'test':{k:{'mean':float(np.mean([r[k] for r in scores])),
                         'sample_sd':float(np.std([r[k] for r in scores],ddof=1))}
                       for k in ['PLCC','SROCC','KROCC','Aggregate','mae']},'probe':{}}
        for variant in probes[0]:
            entry['probe'][variant]={}
            for group in probes[0][variant]:
                entry['probe'][variant][group]={key:{'mean':float(np.mean([r[variant][group][key] for r in probes])),
                    'sample_sd':float(np.std([r[variant][group][key] for r in probes],ddof=1))}
                    for key in ['mean_delta','fraction_score_increases']}
                entry['probe'][variant][group]['n']=probes[0][variant][group]['n']
        external=[root/'external_gpu'/f'{arm}_seed{s}'/'metrics.json' for s in [42,123,456]]
        if all(path.exists() for path in external):
            records=[json.loads(path.read_text()) for path in external]
            entry['external']={key:{'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
                for key,values in {
                    'SROCC':[r['overall']['SROCC'] for r in records],
                    'CHEST_SROCC':[r['by_anatomy']['CHEST']['SROCC'] for r in records],
                    'PELVIS_SROCC':[r['by_anatomy']['PELVIS']['SROCC'] for r in records],
                    'pair_agreement':[r['reconstruction_pair_agreement']['agreement'] for r in records]}.items()}
        result['arms'][arm]=entry
    if args.require_complete and (pending or any('external' not in e for e in result['arms'].values())):
        raise RuntimeError('The declared 27-model study is incomplete')
    (root/'robustness_summary.json').write_text(json.dumps(result,indent=2))
    lines=['\\begin{table}[ht]\\centering\\small',
           '\\caption{Released LDCT test results, mean $\\pm$ sample SD over three training seeds on one fixed test collection.}',
           '\\begin{tabular}{lrr}\\toprule Model & Aggregate $\\uparrow$ & MAE $\\downarrow$ \\\\ \\midrule']
    for arm,name in zip(arms,names):
        t=result['arms'][arm]['test']; a,m=t['Aggregate'],t['mae']
        lines.append(f"{name} & ${a['mean']:.4f}\\pm{a['sample_sd']:.4f}$ & ${m['mean']:.4f}\\pm{m['sample_sd']:.4f}$ \\\\")
    lines+=['\\bottomrule\\end{tabular}\\label{tab:ldct}\\end{table}']
    if all('external' in result['arms'][a] for a in arms):
        lines+=['\\begin{table}[ht]\\centering\\small',
                '\\caption{External CBCT-IQ phantom transfer, mean $\\pm$ seed SD. All models use fixed inference on 1,666 expert-annotated slices.}',
                '\\begin{tabular}{lrr}\\toprule Model & Spearman $\\uparrow$ & Pair agreement $\\uparrow$ \\\\ \\midrule']
        for arm,name in zip(arms,names):
            e=result['arms'][arm]['external']; s,a=e['SROCC'],e['pair_agreement']
            lines.append(f"{name} & ${s['mean']:.3f}\\pm{s['sample_sd']:.3f}$ & ${a['mean']:.3f}\\pm{a['sample_sd']:.3f}$ \\\\")
        lines+=['\\bottomrule\\end{tabular}\\label{tab:external}\\end{table}']
        lines+=['\\begin{table}[ht]\\centering\\small',
                '\\caption{External signed Spearman correlation by phantom anatomy, mean $\\pm$ seed SD. The thorax and pelvis contain 1,207 and 459 annotated images, respectively.}',
                '\\begin{tabular}{lrr}\\toprule Model & Thorax & Pelvis \\\\ \\midrule']
        for arm,name in zip(arms,names):
            e=result['arms'][arm]['external']; c,p=e['CHEST_SROCC'],e['PELVIS_SROCC']
            lines.append(f"{name} & ${c['mean']:.3f}\\pm{c['sample_sd']:.3f}$ & ${p['mean']:.3f}\\pm{p['sample_sd']:.3f}$ \\\\")
        lines+=['\\bottomrule\\end{tabular}\\label{tab:anatomy}\\end{table}']
    (paper/'results_generated.tex').write_text('\n'.join(lines)+'\n')
    supplement=[]
    variants=['gaussian_0.75','gaussian_1.5','gaussian_2.5','box_5','noise_0.01','noise_0.03']
    for group,label in [('all','All development validation images'),('target_at_least_3','Original MOS at least 3'),('target_at_most_1','Original MOS at most 1')]:
        # Preserve the actual probe schema rather than silently dropping a stratum.
        if group not in result['arms']['compact']['probe']['gaussian_0.75']:
            raise ValueError(f'Missing probe stratum: {group}')
        n=result['arms']['compact']['probe']['gaussian_0.75'][group]['n']
        supplement += ['\\begin{table}[ht]\\centering\\scriptsize',
            f'\\caption{{{label} ($n={n}$): percentage receiving a higher score after each alteration, mean $\\pm$ seed SD. These preferences have no transformed-image reader labels.}}',
            '\\begin{tabular}{lrrrrrr}\\toprule Model & G 0.75 & G 1.5 & G 2.5 & Box 5 & N .01 & N .03 \\\\ \\midrule']
        for arm,name in zip(arms,names):
            values=[result['arms'][arm]['probe'][v][group]['fraction_score_increases'] for v in variants]
            cells=[f"${100*v['mean']:.1f}\\pm{100*v['sample_sd']:.1f}$" for v in values]
            supplement.append(name+' & '+' & '.join(cells)+' \\\\')
        supplement += ['\\bottomrule\\end{tabular}\\end{table}']
    (paper/'probe_supplement.tex').write_text('\n'.join(supplement)+'\n')
    bootstrap=root/'external_gpu/paired_source_slice_bootstrap.json'
    if not pending and bootstrap.exists():
        comparisons=json.loads(bootstrap.read_text())['comparisons']
        if len(comparisons)!=8: raise ValueError('Incomplete final paired comparisons')
        external={a:e['external'] for a,e in result['arms'].items()}
        def mean(arm,key): return external[arm][key]['mean']
        text=(
            f"All 27 models were evaluated on all 1,666 annotated external images. The scalar baseline has signed Spearman correlation {mean('compact','SROCC'):.3f} and reader-pair agreement {100*mean('compact','pair_agreement'):.1f}\\%. "
            f"The scalar conditional mild-blur control achieves {mean('scalar_cond','SROCC'):.3f} and {100*mean('scalar_cond','pair_agreement'):.1f}\\%, respectively, whereas its unconditional counterpart achieves {mean('unconditional','SROCC'):.3f} and {100*mean('unconditional','pair_agreement'):.1f}\\%. "
            "This supports a contribution from quality conditioning in this scalar mild-blur comparison, without demonstrating population-level generalization.\n\n"
            f"The original ordinal strong-blur constraint achieves correlation {mean('conditional','SROCC'):.3f} and pair agreement {100*mean('conditional','pair_agreement'):.1f}\\%, compared with {mean('rps','SROCC'):.3f} and {100*mean('rps','pair_agreement'):.1f}\\% without the constraint. "
            f"However, the matched unconditional strong-blur arm achieves {mean('strong_unconditional','SROCC'):.3f} and {100*mean('strong_unconditional','pair_agreement'):.1f}\\%. "
            "The conditional-minus-unconditional intervals include zero for both endpoints (Table~\\ref{tab:paired}); this arm does not isolate a reliable additional benefit from the quality gate.\n\n"
            f"The ordinal mild-blur arm has correlation {mean('mild','SROCC'):.3f} and pair agreement {100*mean('mild','pair_agreement'):.1f}\\%, despite eliminating score increases on the illustrated synthetic blur probes. "
            f"ResNet-18 conditioning changes correlation from {mean('resnet_baseline','SROCC'):.3f} to {mean('resnet_cond','SROCC'):.3f}, but agreement remains poor ({100*mean('resnet_cond','pair_agreement'):.1f}\\%). "
            "Synthetic constraint satisfaction therefore does not predict successful expert-rated transfer across heads and backbones.\n\n"
            "Variation is substantial across training seeds and phantom anatomies (Table~\\ref{tab:anatomy} and Figure~\\ref{fig:external}). In particular, pooled gains do not establish stable pelvic performance. Paired source-slice intervals are descriptive summaries conditional on the two phantoms and the three fixed seeds; they do not capture uncertainty across patients, sites, physical phantoms or training-seed populations. Adaptive follow-up controls and multiple examined contrasts further limit confirmatory interpretation.\n"
        )
        (paper/'external_findings.tex').write_text(text)
        short=['Ordinal C1.5 -- ordinal','Ordinal C0.75 -- ordinal','Ordinal C1.5 -- scalar',
               'Ordinal C0.75 -- scalar','Scalar C0.75 -- scalar','Scalar C0.75 -- scalar U0.75',
               'ResNet C0.75 -- ResNet','Ordinal C1.5 -- ordinal U1.5']
        lines=['\\begin{table}[ht]\\centering\\scriptsize',
            '\\caption{Paired differences with 95\\% source-slice bootstrap percentile intervals within the two fixed phantoms, averaging three fixed seeds. C restricts the blur constraint to training MOS $\\geq3$; U applies it universally. These intervals do not quantify patient/site or training-seed population uncertainty.}',
            '\\begin{tabular}{lrr}\\toprule Contrast & $\\Delta$ Spearman [interval] & $\\Delta$ pair agreement [interval] \\\\ \\midrule']
        for label,comparison in zip(short,comparisons.values()):
            cells=[]
            for key in ['SROCC','pair_agreement']:
                v=comparison[key]; lo,_,hi=v['percentiles_2_5_50_97_5']
                cells.append(f"${v['point']:+.3f}\\;[{lo:+.3f},{hi:+.3f}]$")
            lines.append(label+' & '+' & '.join(cells)+' \\\\')
        lines+=['\\bottomrule\\end{tabular}\\label{tab:paired}\\end{table}']
        (paper/'external_comparisons.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors=['#345d7e','#8b8b8b','#cc8c38','#208c80']
    figure_arms=['compact','rps','conditional','mild']
    means=[result['arms'][a]['test']['Aggregate']['mean'] for a in figure_arms]
    std=[result['arms'][a]['test']['Aggregate']['sample_sd'] for a in figure_arms]
    labels=['Scalar','Ordinal','Blur 1.5','Blur 0.75']
    for i,(mean,sd,color) in enumerate(zip(means,std,colors)):
        axes[0].errorbar(i,mean,yerr=sd,fmt='o',capsize=4,color=color,markersize=7)
    axes[0].set_xticks(range(4),labels)
    axes[0].set_xlim(-.5,3.5)
    axes[0].set_ylim(2.6,2.8); axes[0].set_ylabel('LDCT test aggregate'); axes[0].set_title('(a) Benchmark score: mean and seed SD')
    variants=['gaussian_0.75','gaussian_1.5','box_5']; x=np.arange(3)
    for index,arm in enumerate(figure_arms):
        values=[result['arms'][arm]['probe'][v]['target_at_least_3']['fraction_score_increases']['mean']*100 for v in variants]
        deviations=[result['arms'][arm]['probe'][v]['target_at_least_3']['fraction_score_increases']['sample_sd']*100 for v in variants]
        axes[1].bar(x+(index-1.5)*.19,values,yerr=deviations,capsize=3,width=.18,label=labels[index],color=colors[index])
        for position,value in zip(x+(index-1.5)*.19,values):
            if value==0: axes[1].text(position,1,'0',ha='center',va='bottom',fontsize=8,color=colors[index])
    axes[1].set_xticks(x,['Gaussian 0.75','Gaussian 1.5','Box filter 5'])
    axes[1].set_ylim(0,105); axes[1].set_ylabel('Altered image receives higher score (%)')
    axes[1].set_title('(b) Synthetic probes on 49 high-rated slices')
    axes[1].legend(fontsize=10.5,ncol=2)
    fig.savefig(paper/'images/robustness_tradeoff.png',dpi=220)
    fig.savefig(paper/'images/robustness_tradeoff.svg')
    plt.close(fig)
    if all('external' in result['arms'][a] for a in arms) and not pending:
        fig,axes=plt.subplots(1,2,figsize=(11,6),sharey=True,layout='constrained')
        y=np.arange(len(arms)); labels=['E-B0 scalar','E-B0 ordinal','E-B0 ordinal + C(1.5)',
            'E-B0 ordinal + C(0.75)','E-B0 scalar + C(0.75)','E-B0 scalar + U(0.75)',
            'R18 scalar','R18 scalar + C(0.75)','E-B0 ordinal + U(1.5)']
        for axis,key,title in zip(axes,['SROCC','pair_agreement'],['(a) Human overall-quality Spearman','(b) Expert reconstruction-pair agreement']):
            for i,arm in enumerate(arms):
                records=[json.loads((root/'external_gpu'/f'{arm}_seed{s}'/'metrics.json').read_text()) for s in [42,123,456]]
                values=[r['overall']['SROCC'] if key=='SROCC' else r['reconstruction_pair_agreement']['agreement'] for r in records]
                stats=result['arms'][arm]['external'][key]
                axis.errorbar(stats['mean'],i,xerr=stats['sample_sd'],fmt='D',color='#233e55',capsize=4,markersize=5,zorder=3)
                axis.scatter(values,np.array([i-.14,i,i+.14]),color='#639cbe',s=20,alpha=.9,zorder=4)
            axis.set_title(title,fontsize=11); axis.grid(axis='x',alpha=.2)
            axis.set_xlim((-1.05,1.05) if key=='SROCC' else (-.03,1.03))
            axis.axvline(0 if key=='SROCC' else .5,color='#888888',lw=.8,ls='--')
        axes[0].set_yticks(y,labels,fontsize=10); axes[0].invert_yaxis()
        axes[0].set_xlabel('Signed Spearman correlation'); axes[1].set_xlabel('Fraction of reader-majority pairs')
        fig.savefig(paper/'images/external_transfer.png',dpi=220)
        fig.savefig(paper/'images/external_transfer.svg'); plt.close(fig)
    print(json.dumps({a:result['arms'][a]['test']['Aggregate'] for a in arms},indent=2))


if __name__=='__main__': main()
