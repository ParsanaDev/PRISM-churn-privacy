# PRISM — Assignment 2: Background and Related Work

## Files
- `2_Background.tex`: main LaTeX source; select this as the main document in Overleaf.
- `bibliography.bib`: eight references used by the paper.
- `2_Background_Abhishek_Dev_Ravi_Bharath.pdf`: compiled two-page paper.
- `acmart.cls`: official ACM class, release v2.20.
- `ACM-Reference-Format.bst`: official ACM bibliography style.
- `2_Background.bbl`: generated bibliography for convenient compilation.
- `acmart.dtx` and `acmart.ins`: original ACM class distribution sources.

## Compile
Upload this ZIP as a new Overleaf project and select `2_Background.tex` as its main document. Use pdfLaTeX with a recent TeX Live version.

For local compilation:

    latexmk -pdf 2_Background.tex

Or run pdfLaTeX, BibTeX, then pdfLaTeX twice. Standard ACM font packages (libertine, inconsolata, newtx) and TikZ must be installed. The figures are drawn directly in LaTeX; there are no external image dependencies.

## Coverage
The two-page paper contains the abstract and repository link, introduction, historical background, two-level hierarchy figure, comparative discussion and table, Venn diagram, project positioning, author list with the graduate lead last, and references. The page count includes all of these elements. It contains no experimental results or invented measurements.

## Project decisions to review
The original proposal did not specify the dataset, feature-tier membership, classifiers, or evaluation metrics. The paper presents candidate tiers, logistic regression and boosted trees, and evaluation metrics as proposed future work; these are not represented as completed experiments. Final feature membership requires an audit of the chosen dataset.

The repository URL is reproduced from the proposal. Repository access, individual contributions, and commit history were not verified or changed. The team must ensure that these satisfy the course requirements.

## References checked
Copeland (1923), Schmittlein et al. (1987), Mozer et al. (2000), Neslin et al. (2006), Verbeke et al. (2012), Chen and Guestrin (2016), Goldsteen et al. (2022), and Omari et al. (2025).
