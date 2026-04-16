# Thesis LaTeX Editing Guide

## File Map

| File | What it contains |
|---|---|
| `main.tex` | Preamble, metadata macros, document structure — compile this |
| `frontmatter.tex` | Abstract only |
| `declaration.tex` | Statement of originality (appears after References) |
| `acknowledgements.tex` | Acknowledgements (appears after Declaration) |
| `chapter1_introduction.tex` | Chapter 1 |
| `chapter2_literature_review.tex` | Chapter 2 |
| `chapter3_methodology.tex` | Chapter 3 |
| `chapter4_experiments.tex` | Chapter 4 |
| `chapter5_results.tex` | Chapter 5 |
| `chapter6_conclusion.tex` | Chapter 6 |
| `appendix.tex` | Appendices A, B, C … |
| `references.bib` | BibTeX database |
| `figures/` | All images (PNG, PDF, EPS) |

---

## How to Compile

Run these commands in the `writting/` directory, in order:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Or with latexmk (runs everything automatically):

```bash
latexmk -xelatex main.tex
```

> **Why four passes?** First XeLaTeX writes the `.aux` file; BibTeX reads it and writes `.bbl`; the next two XeLaTeX passes resolve cross-references and page numbers.

---

## Changing Your Personal Details

All personal details are defined in one place — lines 182–189 of `main.tex`:

```latex
\newcommand{\thesistitle}{CT-MUSIQ: An Automated ...}
\newcommand{\thesisauthor}{M.\ Samiul Hasnat}
\newcommand{\studentid}{2021141460103}
\newcommand{\thesisgrade}{2021}
\newcommand{\thesisschool}{College of Computer Science}
\newcommand{\thesismajor}{Computer Science and Technology}
\newcommand{\thesisadviser}{[Supervisor Name]}
\newcommand{\thesisdate}{April 2026}
```

Edit any value here and every place it appears (cover page, header, abstract header, acknowledgements) updates automatically.

---

## Document Structure (Page Order)

```
Cover page          (no page number)
Abstract            (roman i …)
Table of Contents
List of Figures
List of Tables
Chapter 1–6         (arabic 1 …)
References
Declaration
Acknowledgements
Appendices
```

To reorder sections, move the corresponding `\input{...}` lines in `main.tex` between the `FRONT MATTER`, `MAIN MATTER`, and `BACK MATTER` comment blocks.

---

## Writing Chapters

Each chapter file starts with:

```latex
\chapter{Your Chapter Title}
\label{chap:yourlabel}
```

### Sections and subsections

```latex
\section{Section Title}
\label{sec:yourlabel}

\subsection{Subsection Title}
\label{subsec:yourlabel}

\subsubsection{Subsubsection Title}
```

### Font sizes used (per SCU standard)

| Heading level | LaTeX command | Size |
|---|---|---|
| Chapter | `\chapter{}` | 16 pt, bold, centered |
| Section | `\section{}` | 15 pt, bold |
| Subsection | `\subsection{}` | 14 pt, bold |
| Subsubsection | `\subsubsection{}` | 12 pt, bold |
| Body text | — | 12 pt, 1.5× spacing |

---

## Adding Figures

Place your image file in `figures/` then:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.75\textwidth]{your_image.png}
  \caption{Your caption text here.}
  \label{fig:yourlabel}
\end{figure}
```

- Captions appear **below** figures automatically.
- Figure numbers are two-level: Figure 3.1, Figure 3.2, etc.
- Reference a figure with `\cref{fig:yourlabel}` or `Figure~\ref{fig:yourlabel}`.

### Side-by-side figures

```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{image_a.png}
    \caption{Left subfigure.}
    \label{fig:a}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{image_b.png}
    \caption{Right subfigure.}
    \label{fig:b}
  \end{subfigure}
  \caption{Overall caption.}
  \label{fig:both}
\end{figure}
```

---

## Adding Tables

Captions appear **above** tables automatically.

```latex
\begin{table}[htbp]
  \centering
  \caption{Your table caption.}
  \label{tab:yourlabel}
  \begin{tabular}{lcc}
    \toprule
    Column A & Column B & Column C \\
    \midrule
    Row 1    & 0.95     & 0.94     \\
    Row 2    & 0.88     & 0.87     \\
    \bottomrule
  \end{tabular}
\end{table}
```

Use `\toprule`, `\midrule`, `\bottomrule` from the `booktabs` package — never use `\hline`.

For tables wider than one column, or that span multiple pages, use `tabularx` or `longtable`.

---

## Adding Equations

Inline: `$E = mc^2$`

Display, numbered:

```latex
\begin{equation}
  \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2
  \label{eq:mse}
\end{equation}
```

Equation numbers are two-level: (3.1), (3.2), etc.

Multi-line aligned equations:

```latex
\begin{align}
  a &= b + c \\
  d &= e - f
  \label{eq:system}
\end{align}
```

Reference with `\eqref{eq:mse}` which produces `(3.1)`.

---

## Citations and References

### Citing in text

```latex
Single citation:          \cite{lee2023ldctiqac}
Multiple:                 \cite{lee2023ldctiqac, dosovitskiy2021vit}
With page number:         \cite[p.~5]{lee2023ldctiqac}
```

### Adding a reference to references.bib

Open `references.bib` and add an entry:

```bibtex
@article{lee2023ldctiqac,
  author  = {Lee, Sungho and others},
  title   = {LDCT-IQA Challenge 2023},
  journal = {Medical Image Analysis},
  year    = {2023},
  volume  = {84},
  pages   = {102710}
}

@inproceedings{dosovitskiy2021vit,
  author    = {Dosovitskiy, Alexey and others},
  title     = {An Image is Worth 16x16 Words},
  booktitle = {ICLR},
  year      = {2021}
}
```

Common entry types: `@article`, `@inproceedings`, `@book`, `@phdthesis`, `@techreport`, `@misc`.

The bibliography style is `unsrtnat` (numbered, sorted by citation order). To change style, edit line 287 of `main.tex`:

```latex
\bibliographystyle{unsrtnat}   % current — numbered, unsorted
\bibliographystyle{plainnat}   % numbered, sorted alphabetically
\bibliographystyle{ieeetr}     % IEEE format
```

---

## Algorithms / Pseudocode

```latex
\begin{algorithm}[htbp]
  \caption{Your Algorithm Name}
  \label{alg:yourlabel}
  \begin{algorithmic}[1]
    \Require Input data $X$, learning rate $\eta$
    \Ensure Trained model $\theta$
    \For{$epoch = 1$ \textbf{to} $E$}
      \State Compute loss $\mathcal{L}(\theta)$
      \State $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
    \EndFor
    \State \Return $\theta$
  \end{algorithmic}
\end{algorithm}
```

---

## Coloured Highlight Boxes

Use `tcolorbox` for callouts or highlighted results:

```latex
\begin{tcolorbox}[colback=blue!5, colframe=blue!50, title=Key Result]
  CT-MUSIQ achieves PLCC = 0.9498, SROCC = 0.9488.
\end{tcolorbox}
```

---

## Code Listings

```latex
\begin{lstlisting}[language=Python, caption={Training loop.}, label={lst:train}]
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
\end{lstlisting}
```

---

## Cross-References

Always label everything and use `\cref{}` (from `cleveref`) — it automatically adds the right word ("Figure", "Table", "Chapter", etc.):

```latex
\cref{fig:architecture}   ->  "Figure 3.1"
\cref{tab:results}        ->  "Table 5.2"
\cref{eq:kl}              ->  "Equation (4.3)"
\cref{chap:methodology}   ->  "Chapter 3"
\Cref{fig:architecture}   ->  "Figure 3.1"  (capitalised, for start of sentence)
```

---

## Changing Page Margins

Edit line 31 of `main.tex`:

```latex
\geometry{left=3cm, right=2.5cm, top=3cm, bottom=2.5cm}
```

SCU standard requires these exact values — do not change for submission.

---

## Changing the Page Header

Edit lines 141–142 of `main.tex`:

```latex
\fancyhead[L]{\small\textit{Sichuan University Undergraduate Graduation Project (Academic Thesis)}}
\fancyhead[R]{\small\textit{\thesistitle}}
```

`[L]` = left side, `[R]` = right side, `[C]` = centred.

---

## Adding Appendix Chapters

In `appendix.tex`, each appendix is a `\chapter{}`:

```latex
\chapter{Task Request}
\label{app:task}
Content here ...

\chapter{Source Code}
\label{app:code}
Content here ...
```

They are automatically lettered A, B, C … because `main.tex` calls `\appendix` before `\input{appendix}`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Reference shows `[?]` | BibTeX not run | Run the 4-pass compile sequence |
| Figure/table number shows `??` | Need another XeLaTeX pass | Run `xelatex` once more |
| `Undefined control sequence \cref` | `cleveref` not loaded | Check `main.tex` preamble |
| Font warning about Times New Roman | Font not installed (pdfLaTeX mode) | Compile with `xelatex` |
| `natbib` conflict warning | Wrong bib style | Keep `\bibliographystyle{unsrtnat}` |
| `Overfull \hbox` warning | Text too wide for column | Acceptable; fix only if visible in PDF |
