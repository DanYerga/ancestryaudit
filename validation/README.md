# Validation study

`project_final.ipynb` is the notebook used to validate AncestryAudit on real
data. It is also public on Kaggle:
https://www.kaggle.com/code/danae345/project-final

## Data

Gene-level copy-number data (ABSOLUTE LiftOver, primary tumors) from the
open-access tier of The Cancer Genome Atlas, downloaded through the NCI
Genomic Data Commons API. Patients were grouped by self-reported race
(White, Asian) as recorded in TCGA clinical data; this is a social category,
not a measure of genetic ancestry.

| Analysis | White | Asian |
| --- | --- | --- |
| Liver (LIHC) vs stomach (STAD), main | 451 | 242 |
| Thyroid (THCA) vs bladder (BLCA), negative control | 636 | 89 |
| Colon (COAD) vs lung (LUAD), negative control | 594 | 17 |
| Esophagus (ESCA) vs stomach (STAD) | 373 | 125 |
| Squamous ESCA vs STAD | 307 | 124 |

## Pipeline

Missing copy-number values set to 2 (normal); 75/25 split of White
patients; scaling and PCA (50 components) fitted on White training rows only;
six classifiers (random forest, extra trees, XGBoost, LightGBM, logistic
regression, multilayer perceptron) trained on White patients only; permutation
test (1,000 shuffles) under raw and balanced accuracy with Holm correction;
negative controls; esophageal subtype check; simulation-based minimum
detectable effect; fine-tuning and CORAL; SHAP.

## Main results

Single-split results use seed 42; every analysis was also repeated on 20
random 75/25 splits of the White patients.

- Liver vs stomach: no significant gap for any classifier (0/6 under raw and
  balanced accuracy); a significant model appeared in only 1 of 20 splits.
  With these cohorts, gaps of about 11 percentage points are detected with
  80% power; the false-positive rate at zero gap was 3.0%.
- Negative controls: colon/lung 0 of 20 splits; thyroid/bladder 4 of 20 under
  balanced accuracy (always logistic regression), with small consistent gaps
  of +2 to +6 points across models.
- Esophagus vs stomach: three classifiers were significantly MORE accurate
  for Asian patients under balanced accuracy (17 of 20 splits had at least
  one). 97% of Asian versus 36% of White esophageal tumors were squamous
  (chi2 = 39.7, p < 0.001). Among squamous tumors only, raw-accuracy gaps of
  up to 20 percentage points (significant in 20 of 20 splits) disappeared
  under balanced accuracy (1 of 20 splits).
- Fine-tuning with labeled Asian patients raised accuracy on held-out Asian
  patients by 2.70 percentage points (95% CI 0.69 to 4.41, raw accuracy);
  CORAL lowered it for all six models.

All random seeds are fixed (`random_state = 42`) and cohort files are
downloaded and read in sorted order, so reruns give identical results.
