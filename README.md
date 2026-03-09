# Replication package of The Dangers of Non--Self-Fixed Architecture Technical Debt and Its Impact on Time-to-Fix

## Description of this study:
Technical Debt (TD) refers to the long-term costs incurred when developers prioritize short-term delivery over quality-improving work. Architectural Technical Debt (ATD) arises when architectural decisions (e.g., technology choices, patterns, or decomposition) prioritize near-term progress over future maintainability and evolvability. Because ATD affects a systems core structure and propagates through architectural dependencies, it is often more expensive and disruptive to remediate than localized code-level debt. Although ATD has been widely studied, an important but underexplored aspect of repayment is \emph{who} performs it. Prior work provides limited empirical evidence on repayment responsibility in ATD and its relationship to time-to-fix.

We empirically study self-fixed ATD, where the introducer also repays the debt, and contrast it with non-self-fixed ATD in large Apache open-source projects. We reconstruct ATD lifecycles by tracing Jira artifacts to version-control history to identify introduction and repayment points and attribute developer roles. We address three research questions on the prevalence of self-fixed ATD, time-to-fix differences between self-fixed and non--self-fixed items, and how factors related to code change and collaboration metrics relate to repayment speed. Using descriptive statistics, non-parametric tests, and survival analysis, we show that self-fixed and non--self-fixed ATD exhibit distinct repayment dynamics and differences in how changes are shared on ATD-affected files. In particular, non--self-fixed ATD is more likely to remain unresolved longer when changes are spread across many developers. These results provide actionable guidance for maintainers to identify high-risk ATD items and to reduce handoff costs by increasing introducer involvement when possible and documenting the design rationale during repayment.

## Structure of the Replication Package

```text
├── code
│   ├── RQ1
│   │   ├── fig5_boxplot originator_vs_fixer_per_issue_WITH-OTHERS.py
│   │   └── fig6_boxplot-intro-fixer-others.py
│   ├── RQ2
│   │   ├── fig16-18_km_rq2_for_sample_data-NON-SELF-FIXED.py
│   │   ├── fig7-8_km_rq2_for_sample_data.py
│   │   └── fig9-12_km_rq2_for_sample_data_by_indicator.py
│   └── RQ3
│       ├── fig19_plot_rq3_boxplots.py
│       └── fig20_seniority_introducer_fixer.py
└── dataset
    ├── ATD-FINAL-DATASET.csv
    └── ATD-FINAL-DATASET-TRACED.csv
```


## Contents

### Dataset
- `ATD-FINAL-DATASET.csv`\
    A CSV dataset derived from Jira issue trackers of ten Apache open-source projects contains issue reports labelled ATD and Weak-ATD.
- `ATD-FINAL-DATASET-TRACED.csv`\
    A CSV dataset mined from the Jira issue trackers of ten Apache open-source projects and traced to the corresponding GitHub commits, including both introduction and repayment commits for each ATD item.

### Code
  This folder contains all source code used to conduct the analyses for each research question (RQ).

## How to Reproduce
To reproduce the results, run the scripts in this folder by following the instructions below.
```
python "<script_name>.py"
```

## Citation

If you use this dataset to support your research and publish a paper, we encourage you to cite the following BibTex in your publication:

```
@article{sutoyo2026selffixedatd,
  title={The Dangers of Non--Self-Fixed Architecture Technical Debt and Its Impact on Time-to-Fix},
  author={Sutoyo, Edi and Avgeriou, Paris and Capiluppi, Andrea},
  year={2026}
}
```


## Contact

- Please use the following email addresses if you have questions:
    - :email: <e.sutoyo@rug.nl>
