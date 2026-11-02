"""Create the organizer-format report while preserving its fixed template text."""
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'reference/organizer-template'
    target = root/'paper/organizer'
    target.mkdir(exist_ok=True)
    template = (source/'main.tex').read_text(encoding='utf8')
    abstract = template.split('\\begin{abstract}')[1].split('\\guide{')[0].strip()
    introduction = template.split('\\section{Introduction}')[1].split('\\guide{')[0].strip()
    acknowledgment = template.split('\\section{Acknowledgment}')[1].split('\\bibliographystyle')[0]
    content = r'''\documentclass[sigconf]{acmart}
\input{preamble}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\guideStylefalse
\begin{document}
\title{Native-Resolution Filament Segmentation with Temporal Validation}
\subtitle{A Solution to the Solar Filament Segmentation Challenge 2026}
\author{Arjun Vijay Prakash}
\email{arjunv.prakash12345@gmail.com}
\begin{abstract}
ABSTRACT_FIXED
We normalize radial illumination, train native-resolution SegFormer and ConvNeXt U-Net++ models with mask-derived topology supervision, and calibrate instance extraction on a development fold. Temporal groups keep adjacent observations and independent annotator views together. A refined single-model dihedral candidate obtains public leaderboard Panoptic Quality 0.34, compared with 0.33 for the ten-member ensemble, and executes all 180 test images in 30.9 minutes on one Tesla T4. A complete 905-view held-out evaluation gives reference PQ 0.3834 for ensemble component hysteresis and 0.3935 for a learned proposal filter. The latter reduces skeleton coverage and fails a predeclared promotion gate. We report incomplete validation and morphology tradeoffs explicitly rather than equating local PQ with a winning competition result.
\end{abstract}
\maketitle
\section{Introduction}
INTRODUCTION_FIXED
Our experiments focus on coherent instances, native-pixel detail, and reproducible inference. Only competition-approved segmentation masks are used as labels. Public leaderboard and local reference scores are distinguished throughout.

\section{Methodology}
\subsection{Data, splits, and normalization}
The competition training release contains 707 physical observations, 1154 annotator views, and 8199 annotations; the test set contains 180 images at $2048\times2048$ pixels. Independent annotator interpretations remain separate targets. Physical-image identities and connected temporal groups are indivisible during five-fold assignment. Adjacent dates separated by at most three days are connected. A deterministic greedy allocation balances image count, filament count, relative-area bins, and year. The minimum separation across folds is 3.000231 days. This does not remove every long-lived solar structure across folds. Fold 0 has prior development exposure and is reserved for postprocessing selection; final complete-fold comparisons use folds 1--4.

A robust circle fit to the dominant bright disk contour estimates its center and radius. We divide intensity by a smoothed 512-bin radial median profile, apply percentile scaling and CLAHE, and zero pixels beyond $0.995R$. The model receives both corrected and original grayscale channels. This retains information that normalization may distort. Mask unions, per-instance boundaries, and Gaussian heatmaps around mask-derived skeletons provide aligned targets. Supplied spine metadata and external segmentation labels are not used.

\begin{figure*}[t]
\centering
\begin{tikzpicture}[node distance=5mm and 5mm,box/.style={draw,rounded corners=2pt,align=center,font=\small,minimum height=10mm,text width=28mm},arr/.style={-{Stealth[length=2mm]},thick}]
\node[box] (raw) {GONG images\\approved masks};
\node[box,right=of raw] (split) {Physical identity\\temporal folds};
\node[box,right=of split] (prep) {Radial correction\\raw + CLAHE};
\node[box,right=of prep] (train) {Native crops\\disk context};
\node[box,right=of train] (heads) {Two backbones\\three heads};
\draw[arr] (raw)--(split);\draw[arr] (split)--(prep);\draw[arr] (prep)--(train);\draw[arr] (train)--(heads);
\node[box,below=8mm of heads] (maps) {Overlapping tiles\\D4 map average};
\node[box,left=of maps] (policy) {Fold-0-frozen\\instance policy};
\node[box,left=of policy] (check) {Disjoint masks\\COCO validation};
\node[box,left=of check] (csv) {Test CSV\\Kaggle scoring};
\draw[arr] (heads)--(maps);\draw[arr] (maps)--(policy);\draw[arr] (policy)--(check);\draw[arr] (check)--(csv);
\end{tikzpicture}
\caption{Training and inference pipeline. Development policies are frozen before complete-fold evaluation. The proposal filter is an evaluated alternative and is not promoted because it loses skeleton coverage.}
\label{fig:pipeline}
\end{figure*}

\subsection{Architecture and optimization}
We use SegFormer-B3~\cite{segformer} and ConvNeXt-Small~\cite{convnext} U-Net++~\cite{unetpp} encoders and decoders. A separate convolutional branch sees the two native-resolution channels before encoder downsampling. Resized decoder features are fused with this branch and a pooled context vector from a $128\times128$ normalized full disk. Three heads predict mask, skeleton heatmap, and boundary logits. Global context supplies image appearance; it does not encode crop heliographic position.

The mask objective is $0.4L_{\mathrm{Dice}}+0.3L_{\mathrm{focal}}+0.3L_{\mathrm{clDice}}$, with weight 0.1 for each auxiliary binary cross-entropy. clDice~\cite{cldice} uses differentiable skeleton precision and recall. Empty-target crops use a probability penalty instead of undefined topology. Initial training uses ImageNet initialization, 16 epochs of 800 crops at 512 pixels, seed 42, batch size one, two-step accumulation, AdamW learning rate $10^{-4}$, encoder multiplier 0.1, cosine decay, and mixed precision. Foreground-biased sampling has probability 0.65; the remaining crops sample locations uniformly. Foreground pixels are sampled uniformly, so large instances receive more samples. Synchronized rotations/reflections preserve target alignment. The minimum training-loss checkpoint is retained without outer-fold selection.

Refinement warm-starts ConvNeXt folds 0 and 1 for 12 epochs of 600 crops at 640 pixels, learning rate $3\times10^{-5}$ and seed 142. Crop size, optimization restart, and seed change together, so this is a recipe comparison rather than an isolated crop-size ablation.

\subsection{Inference and instance extraction}
Hann-weighted overlapping tiles blend predictions at native resolution. Dihedral augmentation averages four rotations and their reflected variants after inverse transforms. The ensemble averages ten members, two backbone families across five folds. Validation uses only the two models excluding the evaluated fold, without D4, so it does not directly measure the ten-member test configuration.

The original submitted policy uses one fallback marker per foreground component and minimum area 256 pixels. The calibrated policy uses foreground threshold 0.5 and requires a seed probability of 0.65 in each retained component. All parameters are fitted on fold-0 saved maps before evaluating complete folds 1--4. A shape alternative groups components using a radius-four closing only to establish identity; added pixels are discarded. A learned quality alternative uses 128 shallow ExtraTrees regressors, inspired by the 2018 Data Science Bowl winning system~\cite{dsb}. It fits 229 fold-0 out-of-fold proposals across 13 temporal groups. Grouped out-of-sample predictions select threshold 0.2. Features measure probability, geometry, and local intensity contrast; targets are best-match IoU above 0.5 and zero otherwise. Portable JSON trees reproduce the fitted estimator.

Each output row contains a filament identifier and compressed COCO RLE counts in Fortran order. Validation checks identities, dimensions, nonempty masks, canonical round trips, and zero overlapping prediction pixels.

\section{Evaluation}
\subsection{Metric contract and uncertainty}
We reproduce the pinned public self-evaluation notebook version 6. Its reference PQ pools qualifying-pair IoU sums and divides by $N_{\rm pairs}+0.5N_{\rm FP}+0.5N_{\rm FN}$, where qualifying pairs have IoU greater than 0.5. Tests verify the notebook's all-pairs behavior, including overlapping annotations; it differs from one-to-one matching in that case. The hidden server includes undisclosed safeguards, so parity with the public notebook does not establish parity with Kaggle. A provisional multiscale grid-IoU diagnostic is implemented but is not claimed to be an exact official MIoU.

We independently recompute complete-fold scores from per-view counts. For uncertainty, 10000 paired bootstrap draws resample temporal groups within each held-out fold, keeping all physical observations and annotator views in a group together. Intervals condition on the fixed trained models and frozen policies; they exclude retraining and model-selection uncertainty.

\begin{table}[t]
\caption{Complete folds 1--4, 905 views, no D4. Policies are selected on fold 0. Intervals are temporal-block percentile intervals.}
\label{tab:full}
\centering\small
\begin{tabular}{lrr}
\toprule Policy & Reference PQ & 95\% interval\\\midrule
Calibrated components & 0.3834 & [0.3734, 0.3930]\\
Shape grouping & 0.3900 & [0.3801, 0.3994]\\
Learned quality filter & 0.3935 & [0.3832, 0.4031]\\\bottomrule
\end{tabular}
\end{table}

\subsection{Results and morphology tradeoffs}
The quality-filter gain over calibrated components is 0.0100, with paired 95\% interval [0.0080, 0.0120]. Yet medium-instance hit rate falls from 0.6490 to 0.6262 and mean medium-instance skeleton recall from 0.7268 to 0.6979. Small-instance hit rate remains 0.1064 across 141 small annotations. The predeclared gate requires PQ gain at least 0.005, mean skeleton-recall loss at most 0.01, and small-instance hit-rate loss at most 0.02. Skeleton coverage fails this gate, so no quality-filter test submission is generated. Higher PQ alone is insufficient evidence of better morphology.

\begin{table}[t]
\caption{Separately scoped model comparisons. Each row uses its stated protocol; these scores must not be ranked as one leaderboard.}
\label{tab:recipes}
\centering\small
\begin{tabular}{p{.62\columnwidth}r}
\toprule Configuration and scope & Reference PQ\\\midrule
Original ConvNeXt, complete fold 1 & 0.3937\\
Refined ConvNeXt, same 215 views & 0.4079\\
Multiscale mask-only continuation, fold 1 & 0.4081\\
Multiscale topology control, fold 1 & 0.4024\\\bottomrule
\end{tabular}
\end{table}

The mask-only continuation misses all 22 small instances in fold 1, versus three matched by its topology control, and reduces skeleton coverage. It changes inference tile size relative to the earlier refinement, while separately calibrated policies also affect the comparison. The negligible PQ difference does not justify expanding this recipe. Instance-balanced foreground sampling and dark-negative mining are implemented and tested, but have no measured training result.

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{../kaggle_qualitative.png}
\caption{Matched held-out fold-0 example and native-pixel detail. Raw image, approved mask, conventional U-Net, and topology-supervised ConvNeXt probabilities use the same view identity. Fold 0 is a development fold; this example does not establish complete-fold improvement.}
\label{fig:example}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{../quality_filter_diagnostic.png}
\caption{Predetermined first saved fold-1 example, retained despite degradation. Filtering reduces false positives from six to four but also loses one match; PQ falls from 0.4246 to 0.4190. Missing material in a large filament remains missing.}
\label{fig:failure}
\end{figure}

\subsection{Public submissions and reproducibility}
The original five-fold ten-member D4 submission, Kaggle reference 57846445, contains 1737 instances and scores public PQ 0.33. The calibrated version, reference 57898046, contains 1682 instances and also scores 0.33. Their local difference does not establish a public gain. An earlier development SegFormer submission scores 0.27.

The portable refined candidate, reference 57906671, contains 1465 nonoverlapping instances over 177 of 180 test images and obtains public PQ 0.34. A fresh Kaggle session runs from copied full-precision weights and packaged source, without retraining or access to private source checkpoints. D4 inference at tile size 512 takes 1852.5 seconds for all 180 observations, 10.29 seconds per observation, on one Tesla T4. Its 0.4079 local result excludes D4 and covers only fold 1. The archive includes a notebook, policy, package versions, checkpoint hash, and exact submitted CSV hash. The source and portable candidate are available at \url{https://github.com/ArjunCodess/solar-filament-segmentation-2026}; final organizer delivery is pending.

The code passes 38 tests covering preprocessing, grouping, topology gradients, map alignment, public-evaluator parity, portable trees, sampling, and submission constraints. These tests establish implementation behavior, not competitive accuracy. Longer temporal gaps remain a concern: existing ensemble PQ is about 0.38 at 3--30 days from training, but 0.2812 on only 15 views beyond 30 days. This is observational stratification, not purged retraining.

\section{Limitations and disclosure}
CLAHE can amplify atmospheric artifacts, component extraction can merge nearby filaments, and boundary-driven splitting can fragment a single one. Small-instance recall remains poor. The observed public best of 0.34 remains far below the previously checked top-three cutoff of 0.62. No winning rank or award is established. A finite paired experiment now tests instance-balanced sampling and dark distractors before consuming further submission quota; no training outcome is yet available. Generative AI assisted implementation, experiment orchestration, analysis, and report drafting. Results here are attributed only to completed, recorded experiments.

\section{Acknowledgment}ACKNOWLEDGMENT_FIXED
\bibliographystyle{ACM-Reference-Format}
\bibliography{main}
\end{document}
'''
    content = content.replace('ABSTRACT_FIXED', abstract).replace('INTRODUCTION_FIXED', introduction)
    content = content.replace('ACKNOWLEDGMENT_FIXED', acknowledgment)
    (target/'main.tex').write_text(content, encoding='utf8')
    shutil.copyfile(source/'preamble.tex', target/'preamble.tex')
    bibliography = (source/'main.bib').read_text(encoding='utf8')
    bibliography += r'''
@inproceedings{segformer,author={Xie, Enze and others},title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},booktitle={NeurIPS},year={2021},eprint={2105.15203}}
@inproceedings{convnext,author={Liu, Zhuang and others},title={A ConvNet for the 2020s},booktitle={CVPR},year={2022},eprint={2201.03545}}
@inproceedings{unetpp,author={Zhou, Zongwei and others},title={UNet++: A Nested U-Net Architecture for Medical Image Segmentation},booktitle={DLMIA/ML-CDS},year={2018},eprint={1807.10165}}
@inproceedings{cldice,author={Shit, Suprosanna and others},title={clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation},booktitle={CVPR},year={2021},eprint={2003.07311}}
@misc{dsb,author={Buslaev, Alexander and Seferbekov, Selim and Durnov, Vladimir},title={{[ods.ai] topcoders, 1st place solution}},year={2018},howpublished={\url{https://github.com/selimsef/dsb2018_topcoders}},note={2018 Data Science Bowl}}
'''
    (target/'main.bib').write_text(bibliography, encoding='utf8')
    for command in (['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                    ['bibtex', 'main'],
                    ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                    ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']):
        result = subprocess.run(command, cwd=target, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            print(result.stdout[-7000:])
            raise SystemExit(result.returncode)
    log = (target/'main.log').read_text(errors='replace')
    if any(s in log for s in ('Overfull', 'undefined references', 'undefined citations')):
        raise SystemExit('Inspect organizer report log for unresolved citations or overflow')
    import pymupdf
    document = pymupdf.open(target/'main.pdf')
    qa = target/'qa'
    qa.mkdir(exist_ok=True)
    for i, page in enumerate(document):
        page.get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(qa/f'page-{i+1}.png')
    acknowledgment_page = next(i+1 for i, page in enumerate(document) if 'Acknowledgment' in page.get_text())
    if acknowledgment_page > 4:
        raise ValueError('Body exceeds four pages')
    shutil.copyfile(target/'main.pdf', root/'paper/organizer_report.pdf')
    print(f'Compiled {len(document)} pages; acknowledgment begins on page {acknowledgment_page}. Inspect {qa}.')


if __name__ == '__main__':
    main()
