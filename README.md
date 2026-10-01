# Integrated Approach for the Identification of KRAS Inhibitors utilizing Machine Learning

A two-stage pipeline for identifying candidate inhibitors of the KRAS Switch II allosteric pocket. A pharmacophore-based virtual screen finds best hits from a compound library, and a machine learning classifier trained on experimental ChEMBL bioactivity data then acts as a second filter on those hits. The advantage of this pipeline is that pharmacophore screening ranks compounds based on features against a single, static crystal pose, while the classifier uses information from in vitro experiments. Combining the two gives a ranking that neither method produces alone.

**Full report:** [KRAS_report.pdf](KRAS_report.pdf) — *Integrated Approach for the Identification of KRAS Inhibitors, Utilizing Machine Learning*, the complete report of this study in IEEE format, with background, methods, results and references.

## Background

KRAS is a GTPase cycling between a GTP-bound active state and a GDP-bound inactive state. The transition to the inactive state requires GTPase Activating Proteins (GAPs) to interact with KRAS, accelerating GTP hydrolysis. Mutations prevent this interaction, locking KRAS in the active state.

KRAS mutations occur in over 20% of all cancers, including roughly 95% of pancreatic, 55% of thyroid, and 35% of lung and colorectal cases. The protein was characterized as undruggable: GTP binds KRAS with high affinity, so targeting the orthosteric site is not a viable option. The identification of an allosteric pocket in 2012 led to a reexamination of the possibility of finding inhibitors against KRAS.

This study targets that Switch II pocket. A ligand bound there will alter the mutant KRAS conformation, preventing the subsequent pathway activation.

## Methods

### 1. Virtual Screening

| | |
|---|---|
| Structure | 4EPY (PDB) |
| Reference ligand | 0QY, co-crystallized in the Switch II pocket |
| Search engine | Pharmit |
| Library | MolPort |

Five pharmacophore features were defined:

| Feature | Coordinates (x, y, z) |
|---|---|
| Hydrogen donor | 23.7, −10.4, 14.2 |
| Hydrogen donor | 24.4, −6.8, 14.8 |
| Hydrogen acceptor | 23.3, −1.3, 7.1 |
| Hydrophobic | 22.8, 0.9, 9.0 |
| Aromatic | 25.1, −4.2, 13.5 |

Additional filters: molecular weight ≤ 550, rotatable bonds ≤ 8.

The screen returned 1391 hits, ranked by ascending RMSD to the pharmacophore (0.357–0.945), which became the test set for the classifier.

### 2. Machine Learning Model

ChEMBL dataset CHEMBL3736 (`data/test_file_labeled.csv`):

- Compounds: 2290
- Activity threshold: IC50 < 1000 nM = Active
- The classes are almost exactly balanced at 49.9% / 50.1%

**Descriptors:** 217 RDKit descriptors computed from SMILES. Descriptor selection was based on six classifiers, each fitted over 5 epochs (30 runs total), with their top 20 descriptors recorded. The descriptors appearing more often than the epoch count were retained, 46 in total, and 15 were used as the maximum for model design.

**Training and evaluation:** SMOTE was applied for class balancing, followed by comparison of six classifiers, evaluated internally by 10-fold cross-validation and externally by hold-out.

| Classifier | Accuracy |
|---|---|
| Random Forest | 0.625 |
| CART | 0.645 |
| XGBoost | 0.604 |
| ExtraTrees | 0.650 |
| **AdaBoost** | **0.714** |
| Gradient Boost | 0.671 |

AdaBoost was selected, reaching ~71% accuracy and AUC ≈ 0.77, using 11 of the 15 available descriptors.

The same RDKit descriptors were computed from the Pharmit SDF so that the hits occupy the identical feature space.

## Results

Of the 1391 Pharmit hits, the model classified 500 as active and 891 as non-active.

> **These figures are pending re-run.** The column-alignment error described under Bugs means the classification of the hits is not reliable as reported here. The pharmacophore model, the training data and the classifier performance are unaffected.

Top-ranked candidates from this run, after combining virtual screening and ML classification:

| Overall rank | Pharmit rank | Compound | RMSD |
|---|---|---|---|
| 1 | 1 | MolPort-039-347-422 | 0.357 |
| 2 | 2 | MolPort-051-896-091 | 0.367 |
| 3 | 3 | MolPort-046-620-676 | 0.378 |
| 4 | 7 | MolPort-051-820-872 | 0.397 |
| 5 | 8 | MolPort-019-894-132 | 0.412 |
| 6 | 14 | MolPort-000-943-528 | 0.455 |
| 7 | 15 | MolPort-045-962-970 | 0.461 |
| 8 | 16 | MolPort-042-568-081 | 0.465 |
| 9 | 19 | MolPort-020-273-826 | 0.471 |
| 10 | 20 | MolPort-030-049-475 | 0.473 |

### Observations

**1. More descriptors did not mean better accuracy.** Performance with 5 descriptors exceeded performance with 14, indicating that the additional descriptors contributed noise rather than signal.

**2. SMOTE was unnecessary.** The dataset is balanced at 1144 / 1146, so oversampling generated synthetic compounds without addressing any real imbalance.

**3. The Pharmit ranking and the ML ranking disagree.** Several compounds Pharmit ranked highly were classified as non-active, and compounds far down the Pharmit list were classified as active. This is expected: Pharmit scores geometric fit to a small feature set against a rigid crystal structure.

## Limitations

**The training data itself.** CHEMBL3736 pools IC50 measurements made against different KRAS mutant forms and under heterogeneous assay conditions, and the consequence is:

- The 2290 records correspond to only 1358 unique chemical structures.
- 347 structures carry both labels — the same molecule appears as Active in one record and Not Active in another. For example, CHEMBL351706 is recorded at both 8.4 nM and 2300 nM.
- These contradictions involve 793 rows, roughly one third of the dataset.

The same applies in the evaluation phase: identical structures distributed across training and test sets mean that the model is sometimes tested on molecules it has already seen, which inflates reported performance.

## Bugs identified in the deployment script

Errors in `3b3` were found while preparing this repository and have not yet been corrected in the code.

**Descriptor column misalignment.** `indxSDF` is built from the full column list of the hits CSV (indices 3, 4, 5 …) but applied to `X` after `X = X[:,3:]` has already removed the first three columns, shifting every descriptor by three positions. The model therefore reads the wrong descriptors. Selecting the columns by name would fix it.

**RMSD read from the wrong column.** `rmsd = X[:,2]` is evaluated after the same slice and so returns a descriptor value rather than the pharmacophore RMSD. It should be read from the named column before slicing.

## About the scripts

| Script | Function |
|---|---|
| 1 | Reads the ChEMBL CSV and computes ~217 RDKit descriptors from each SMILES string |
| 2 | Selects which descriptors matter by using six classifiers, keeping only the descriptors that appear most often among each model's top 20 |
| 3a | Compares the six classifiers, scoring each by employing 10-fold cross-validation, then evaluates on a held-out set with ROC-AUC |
| 3b1 | Converts the Pharmit SDF into SMILES |
| 3b2 | Applies the same descriptor calculation to the 1391 hits so they occupy the identical feature space as the training data |
| 3b3 | Builds the final model: trains, evaluates internally and classifies the Pharmit hits |
| 4 | Employs the Mann-Whitney U test per descriptor to determine whether actives and non-actives genuinely differ |

### Label encoding

| Label in CSV | Numeric class | In prediction output |
|---|---|---|
| Active | 0 | `PREDICTED = 0.0` means active |
| Not Active | 1 | `PREDICTED = 1.0` means not active |

## Repository structure

```
├── README.md                 # this file
├── KRAS_report.pdf           # full report of the study (IEEE format)
├── requirements.txt          # Python packages needed to run the pipeline
├── scripts/
│   ├── 1_Program_add_rdkit_descriptors_to_ChemBL_csv_file.py
│   ├── 2_Program_Python_features_importance_descriptors.py
│   ├── 3a_Program_Python_ML_classification.py
│   ├── 3b1_Program_from_sdf_to_csv_with_smiles.py
│   ├── 3b2_Program_add_rdkit_descriptors_to_SDF_FILE_in_a_csv_file.py
│   ├── 3b3_Program_Python_ML-system_deployment.py
│   ├── 4_Program_Python_Statistical_Analysis.py
│   └── moduleUtils.py        # helper module (timing, console utilities)
└── data/
    ├── test_file_labeled.csv # CHEMBL3736, 2290 labeled records (288 KB)
    └── test_file_hits.sdf    # 1391 Pharmit hits (2.9 MB)
```

## Data sources

- Bioactivity data: [ChEMBL](https://www.ebi.ac.uk/chembl/) (CHEMBL3736)
- Structure: [PDB 4EPY](https://www.rcsb.org/structure/4EPY)
- Screening: [Pharmit](https://pharmit.csb.pitt.edu/), MolPort library

## Tools

RDKit · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SciPy · pandas · NumPy · matplotlib · Pharmit

## Author and attribution

**Eleni Maria Gkevreki** — Biomedical Engineering (Bachelor & Integrated Master's), University of West Attica
[LinkedIn](https://www.linkedin.com/in/eleni-gkevreki)

Carried out November 2025 – February 2026 for the Computational Drug Discovery course.

The general machine learning framework — descriptor calculation, feature selection, classifier comparison and the code that applies the trained model to new compounds — was provided as course material by the department. My contribution was the scientific design and execution of the study: selection of the target and structure (KRAS Switch II pocket, PDB 4EPY), construction of the pharmacophore model from the 0QY, definition of the screening parameters, analysis of the divergence between virtual screening and ML rankings, and the written report included here as `KRAS_report.pdf`.

## License

Released under the MIT License. See [LICENSE](LICENSE).
