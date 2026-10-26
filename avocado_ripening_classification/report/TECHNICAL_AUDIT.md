# Technical Audit — Avocado Ripening Classification

**Audit scope:** Repository in `avocado_ripening_classification/` and the GreenPod Labs Assignment 2 reference supplied for this review.  
**Audit date:** 2026-10-26  
**Evidence boundary:** `data/raw/` contains only `.gitkeep`; there are no class folders, assignment images, trained models, or measured project metrics. Scores below assess implementation and submission readiness, not model accuracy.

## Executive decision

The repository contains both required approaches and a usable training/demo layout. In an isolated Python 3.14 environment, all declared dependencies installed, the full suite passes (**31 tests**) and measured source coverage is **93%**. The synthetic-data integration test trains both approaches, evaluates them, loads checkpoints, and exercises inference. The configured dataset download was audited and found to be a different avocado disease dataset, not the required ripeness dataset. All four notebooks and the real training command were attempted and are blocked until the correct data and model metrics are available.

**Do not submit yet.** Real-data training and EDA have not been run. The report and results remain placeholders, there is no project-specific public GitHub or Colab link, the 30-minute video is not present, and the mandatory Q4 answer is still for the candidate to complete from their own work log.

## Scorecard

| Area | Score / 100 | Evidence and limitation |
|---|---:|---|
| **Overall audit score** | **62** | Solid implementation and verified synthetic pipeline; substantial real-data and submission evidence is missing |
| Code quality | 86 | 93% measured statement coverage; guards and cross-platform decoding; limited static typing/lint evidence |
| ML | 80 | CV-tuned SVM/RF and MobileNetV2 pipeline are implemented and synthetically exercised; real performance is unknown |
| Computer vision | 84 | RGB/HSV/LAB, LBP, GLCM and image-quality EDA are implemented; no real image analysis was possible |
| EDA | 75 | Class, resolution, lighting, sharpness, clipping and border-color analyses exist; not run on assignment data |
| Documentation | 68 | README and Markdown report explain methods; required actual data/results/links and Q4 remain missing |
| Assignment compliance | 52 | Two approaches, evaluation code, demo and mandatory-question sections exist; measured outputs and external artifacts are absent |
| **Submission readiness** | **42** | Blocked by missing dataset/results, candidate answers, repository/Colab/video links |

These are reviewer estimates for implementation and evidence. No accuracy or model-performance score is assigned without real held-out data.

## Project structure checklist

| Expected item | Present? | Review status | Priority |
|---|---|---|---|
| `.gitignore` | Yes | Added `.pytest_cache/`; data and model artifacts remain excluded | Low |
| `README.md` | Yes | Corrected selection/evaluation guidance, install/test instructions, security caveat, submission checklist | Medium |
| `requirements.txt` | Yes | Runtime, test, notebook and coverage dependencies are declared | Medium |
| `data/raw/<class>/` | No images | Only placeholder content; actual dataset must be acquired and extracted | **Blocker** |
| `data/processed/` | Folder exists | No real split manifest or image-quality report until EDA/training runs | **Blocker** |
| Four notebooks | Yes | Notebook JSON and code syntax validate; all notebooks were executed and stopped at dataset discovery because there are no images | **Blocker** |
| Dataset/classical/deep-learning/evaluation modules | Yes | More modular filenames than reference; all core functions are present | Medium |
| `models/` | Folder exists | README only; no trained `.joblib` or `.pth` artifacts | **Blocker** |
| `main.py` | Yes | SVM/RF/MobileNet inference paths exist; predictions are saved as JSON; synthetic inference tested | High |
| Assignment report | DOCX/PDF and Markdown exist | Markdown source improved; DOCX/PDF are templates and are not regenerated or populated with measured results | **Blocker** |
| GitHub repository URL | No project link | Current folder is not its own Git root; the detected Git root is `C:\Users\sivak`. Do not commit from that parent. Create/publish a dedicated project repository. | **Blocker** |
| Colab link | No | Create a view-accessible Colab notebook and add its URL | **Blocker** |
| 30-minute video | No | Walkthrough script exists; recording/link is absent | **Blocker** |
| Automated tests | Added | Unit and integration coverage described below | Pass |

The reference names `src/dl_model.py`; the equivalent implementation is `src/deep_learning_model.py`. This is a naming difference, not missing model functionality. ResNet18 is optional in the supplied brief; the project implements MobileNetV2.

## Source-file review

| File | Purpose | Finding and disposition | Severity |
|---|---|---|---|
| `src/__init__.py` | Package marker | Empty and valid; no issue | Low |
| `src/utils.py` | Class names, paths, JSON, seed | Made PyTorch import lazy so data/classical utilities are independently importable; helper functions remain small | Low |
| `src/dataset.py` | Discovery, validation, split | Added readable-image validation, alias normalization, exact-content SHA-256 deduplication, conflicting-label detection, required-class checks, and minimum sample checks for 70/15/15 stratification | High (fixed) |
| `src/preprocessing.py` | Image decoding and normalization | Replaced `cv2.imread` path decoding with Pillow for Unicode-path safety; explicit corrupt-image error; closes opened images | Medium (fixed) |
| `src/feature_extraction.py` | Classical feature vector | Fixed HSV hue histogram range: OpenCV hue is `[0,180)`, not `[0,256)`. Feature vector remains 206 values, finite and normalized in tests | High (fixed) |
| `src/classical_model.py` | SVM/RF tuning and evaluation | Added nested stratified probability calibration instead of deprecated `SVC(probability=True)`, folds guard, validation macro F1, test metrics, latency/model size, saved confusion matrices | High (fixed) |
| `src/deep_learning_model.py` | MobileNetV2 training/loading | Replaced `BatchNorm1d` in the classifier head with `LayerNorm` for singleton batches; kept frozen feature BatchNorm layers in eval mode; added class-weighted loss, translation augmentation, Windows-safe worker default, CUDA-synchronized timing, metrics/plots and restricted `weights_only=True` checkpoint loading | High (fixed) |
| `src/evaluation.py` | Metrics and plots | Added weighted F1, per-class report, ROC-AUC/OVR ROC curves, confusion matrices, validation-based selection rank; probability columns are reordered for scikit-learn's sorted-label ROC-AUC requirement | High (fixed) |
| `train.py` | End-to-end orchestration | Resolves paths from project root, guards arguments, compares only files from the current run, selects on validation macro F1 rather than test F1, documents skipped DL | High (fixed) |
| `download_dataset.py` | Dataset download | Explicitly downloads only; ZIP extraction remains a documented manual step. It fails explicitly if `gdown` fails | Low |
| `main.py` | Single-image comparison | Added corrupt-image validation and synchronized GPU inference timing; demo artifacts must be trusted because Joblib uses pickle-based serialization | Medium (fixed/documented) |

## Notebook review

| Notebook | Purpose | Review |
|---|---|---|
| `notebooks/01_EDA.ipynb` | Dataset and bias screening | Uses non-strict discovery and includes brightness, sharpness, clipping, border-color proxy and outlier checks. Execution stopped at `discover_dataset`: no images exist. |
| `notebooks/02_Classical_ML.ipynb` | Classical training | Uses shared split/training code; execution stopped because no supported images exist. |
| `notebooks/03_Deep_Learning.ipynb` | MobileNetV2 | Uses shared training function and plots loss curves; execution stopped because no supported images exist. |
| `notebooks/04_Model_Comparison.ipynb` | Comparison and demo | Now raises a clear `FileNotFoundError` when no model metrics exist, instead of writing an empty comparison CSV. Execution stops at this guard until training artifacts exist. |

All four notebook documents pass nbformat validation and code-cell Python parsing. Execution was attempted; the first three stop because `data/raw/` has no supported images, and the comparison notebook stops because training metrics do not exist. The configured Drive archive was separately downloaded and found to have incompatible labels.

## Dependency and security review

- Removed duplicate Pillow declaration and unused `seaborn`, `tqdm`, `albumentations`, `python-docx`, and `reportlab` dependencies.
- Added `pytest`, `nbformat`, and Jupyter dependencies needed for project validation/notebook use.
- A previous audit recorded a clean `pip-audit` result after changing pytest to `pytest>=9.0.3,<10`; this execution did not rerun `pip-audit`. The current audit verified package installation and `pip check`, not current CVE status.
- This project uses compatibility ranges rather than a committed lock file; a previous no-known-vulnerabilities result is not a guarantee for future resolutions or deployment environments.
- `.joblib` model files are pickle-based and must not be loaded from untrusted sources. The PyTorch loader now uses restricted weights-only deserialization.

## Evaluation and ML/CV findings

- Split is stratified and approximately 70/15/15. The two-stage stratification requires at least six unique images per class; otherwise it fails with a clear remediation message.
- Byte-identical images are removed before splitting. Similar images, repeated views, or images of the same fruit can still leak across splits without capture/session grouping metadata.
- SVM standardization remains inside the CV pipeline; class weights remain enabled for SVM/RF. MobileNetV2 now uses class-weighted cross-entropy.
- Test metrics no longer drive model selection. Classical hyperparameters use training CV; candidate models use validation macro F1; MobileNetV2 epochs use validation loss; test results are reported after selection.
- Outputs include accuracy, macro precision/recall/F1, weighted F1, class report, confusion matrices, ROC-AUC/curves when valid, training duration, approximate inference time, and model size.
- The measured inference metrics describe code-level test-loop timing, not production conveyor throughput or a service-level benchmark.

## Test evidence

Validation ran in the isolated local Python 3.14 CPU environment (`.venv`):

```text
31 passed in 25.25s
TOTAL source statement coverage: 93%
```

Coverage includes dataset aliases/corrupt images/duplicates/class checks/splits/manifests; preprocessing including Unicode paths and ImageNet tensor conversion; feature-vector shape and HSV histogram; metrics/ROC/validation selection; notebook format and code parsing; model head/checkpoint round-trip/loader guards; dataset downloader warning behavior; CLI validation and orchestration; demo prediction JSON; and a synthetic end-to-end classical + MobileNet training/inference smoke test.

The integration test uses synthetic images with `pretrained=False`; it is a pipeline smoke test, not an assignment result. It does not substitute for actual EDA, ImageNet-pretrained training, or model-performance reporting. No real-data loss trend, validation score, test score, or model artifact is available.

## Assignment compliance checklist

| Requirement | Status | What is still needed |
|---|---|---|
| Dataset understanding, class imbalance and outlier/bias analysis | ❌ Blocked | Obtain the correct ripeness-labeled dataset, then run EDA and report counts, image quality and observed biases |
| Classical approach (color/texture + SVM/RF) | ✅ Implemented | Run on actual data and report output |
| Transfer-learning CNN | ✅ MobileNetV2 implemented | Run with ImageNet pretrained weights on actual data |
| Stratified 70/15/15 split | ✅ Implemented with safeguards | Preserve/submit the generated manifest |
| Metrics and model comparison | ✅ Implemented | Generate actual values and explain selection based on validation |
| Example run demonstrating both approaches | ⚠️ Script and paths exist | Run `main.py` using a held-out image and capture its output; JSON persistence is implemented but no trained real-data model exists |
| Report with dataset/model rationale/metrics | ⚠️ Markdown source exists | Fill actual findings, regenerate DOCX/PDF, verify consistency |
| Mandatory Q1 production performance | ⚠️ Safe methodology written | Measure on target hardware; do not reuse reference speed figures as measured results |
| Mandatory Q2 drawbacks | ✅ Addressed in report source | Review for candidate-specific observations |
| Mandatory Q3 improvements | ✅ Addressed in report source | May personalize to observed results |
| Mandatory Q4 time spent | ❌ Candidate-specific answer missing | Add an honest answer from the candidate's own work log |
| Public GitHub URL | ❌ Missing | Create/publish a dedicated repository and add URL |
| View-accessible Google Colab URL | ❌ Missing | Create notebook, confirm “Anyone with the link can view,” and add URL |
| 30-minute walkthrough video | ❌ Missing | Record and add link |

## Exact pre-submission steps

1. Create a dedicated project Git repository; do not commit from the detected parent Git root `C:\Users\sivak`.
2. Obtain the correct four-stage ripeness dataset. The current `download_dataset.py` Drive link downloads an incompatible archive; do not extract or train on it. Confirm the four expected class folders before proceeding.
3. Run `python -m pytest` and `python -m pytest --cov=src --cov=main --cov=train --cov=download_dataset --cov-report=term-missing`, then `python train.py --data data/raw --epochs 15` with pretrained weights enabled and documented runtime/hardware.
4. Run `notebooks/01_EDA.ipynb`; review corrupt-image and outlier CSVs, plots, class imbalance, brightness/lighting, and border-color proxy. Document limitations and actual observations.
5. Review `data/processed/split_manifest.csv`; obtain fruit/session IDs and use group-aware splitting if the dataset contains repeated fruit/views.
6. Review `outputs/metrics/` and `outputs/confusion_matrices/`; select by validation macro F1, then report the held-out test metrics without tuning against them.
7. Run `python main.py --image <held-out-test-image>`; save an evidence screenshot and record predictions. The JSON summary is written to `outputs/predictions/<image>_prediction.json`.
8. Complete `report/REPORT_CONTENT.md` with real counts/metrics/observations, write Q4 from your own time log, and regenerate/verify `Assignment_Report.docx` and `Assignment_Report.pdf`.
9. Publish the project-specific GitHub repository; create and test the view-accessible Colab link; record the 30-minute walkthrough.
10. Confirm all links, report fields, and recorded results are consistent before submission.

## Current execution and audit details

### Dataset audit

| Check | Observed result |
|---|---|
| `data/raw/` exists | Yes |
| Expected class directories in project `data/raw/` | 0 of 4 present |
| Images in project `data/raw/` | 0 |
| Configured Drive archive | 1,090 images; valid ZIP, no unsafe paths, `testzip()` passed |
| Drive archive classes | `blackspot`, `canker`, `fresh`, `grenning` — not target labels `underripe`, `breaking`, `ripe`, `overripe` |
| Archive train/test counts | Train: 991; test: 99 |
| Archive overall class counts | blackspot: 206; canker: 201; fresh: 314; grenning: 369 |
| Archive class imbalance | Largest/smallest ratio 369/201 = 1.84; irrelevant to ripeness because labels do not match |
| Corrupt images in archive | 0 of 1,090 |
| Byte-identical duplicate files in archive | 0 of 1,090 |
| Image formats | JPEG: 776; PNG: 314 |
| Image resolution | 150 distinct dimensions; 776 images are 800×800 and the remaining 314 have varied dimensions |
| Required class mapping | Loader maps the four requested ripeness aliases; none match the downloaded archive labels |

Expected structure is not present yet:

```text
data/raw/
├── underripe/
├── breaking/
├── ripe/
└── overripe/
```

### Requested structure and artifact status

| Item | Exists | Working / validated | Missing or needed |
|---|---|---|---|
| `README.md` | Yes | Install/run instructions reviewed; demo output and normalization description corrected | Project URLs and actual result fields |
| `requirements.txt` | Yes | Installed successfully in isolated environment; `pip check` clean | None for declared packages |
| `src/` | Yes | Syntax compilation and tests pass | No real-data evidence |
| `notebooks/` | Yes | Four notebooks parse and validate; execution attempted | Dataset needed to finish all cells and save notebook outputs |
| `models/` | Yes | Synthetic model save/load tested | No real `.joblib` or `.pth` model |
| `outputs/` | Yes | Directory layout exists | No real metrics, plots, confusion matrices or predictions |
| `report/` | Yes | Markdown report covers required topics | Actual counts/metrics, regenerated DOCX/PDF, candidate-specific Q4 |
| `main.py` | Yes | Synthetic SVM/RF/MobileNet inference passes; JSON persistence tested | Real trained models and held-out image |
| `train.py` | Yes | CLI guards and orchestration tests pass | Real dataset required for training |
| `src/dataset.py` | Yes | Alias, validation, duplicate and split tests pass | Actual image audit |
| `src/evaluation.py` | Yes | Metric, classification-report, ROC and plot tests pass | Actual test predictions |

The requested `dataset.py` and `evaluation.py` are implemented under `src/`, rather than at the project root. This is consistent with the repository's modular structure.

### Command log

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe -m pip install -r requirements.txt` | Passed in the isolated Python 3.14.7 environment |
| `.venv\Scripts\python.exe -m pip check` | `No broken requirements found.` |
| `.venv\Scripts\python.exe -m compileall -q src tests train.py main.py download_dataset.py` | Passed |
| `.venv\Scripts\python.exe -m pytest --cov=src --cov=main --cov=train --cov=download_dataset --cov-report=term-missing -q` | `31 passed`; total source statement coverage `93%` |
| `python download_dataset.py --output <temporary-file>` | Downloaded 145 MB; inspected 1,090 images; archive contains different disease/appearance labels |
| `.venv\Scripts\python.exe train.py --data data/raw --epochs 1 --no-pretrained` | Blocked at discovery: no supported images under `data/raw/` |
| Execute `notebooks/01_EDA.ipynb` | Blocked at dataset discovery: no supported images |
| Execute `notebooks/02_Classical_ML.ipynb` | Blocked at dataset discovery: no supported images |
| Execute `notebooks/03_Deep_Learning.ipynb` | Blocked at dataset discovery: no supported images |
| Execute `notebooks/04_Model_Comparison.ipynb` | Blocked by the guard: no trained model metrics; now fails clearly before creating an empty CSV |

The training failure against the project data folder is explicit and expected for an empty dataset:

```text
ValueError: No supported images found under ...\data\raw.
Expected class folders: underripe, breaking, ripe, overripe
```

The initial comparison-notebook attempt wrote a header-only `model_comparison.csv` despite no metric files; that empty artifact was removed and the notebook now guards against this case. The configured Drive download succeeded to a temporary file, but it is the wrong task dataset and was neither extracted into the project nor used for training. Its directory layout (`train/`, `test/`) and disease labels are incompatible with this ripeness pipeline. Final notebook execution created no model/results artifacts. Notebook runs emitted Jupyter/ZeroMQ warnings about the Windows Proactor event loop fallback and an unencrypted local TCP kernel connection; these are runtime transport warnings.

### Missing real-run outputs and submission blockers

- `data/processed/split_manifest.csv`, `image_quality.csv`, and `unreadable_images.csv`
- `outputs/plots/class_distribution.png`, resolution/brightness plots, and MobileNet training curves
- `outputs/metrics/{svm,rf,mobilenetv2}_metrics.json` and `model_comparison.csv`
- `outputs/confusion_matrices/` model plots and any available ROC plots
- `models/svm_model.joblib`, `models/rf_model.joblib`, and `models/mobilenetv2.pth`
- A real held-out prediction JSON under `outputs/predictions/`
- Actual dataset counts, corruption/duplicate results, class imbalance analysis, and observed-bias discussion
- Correct ripeness-labeled dataset: the configured Drive archive contains a different four-class avocado dataset (`blackspot/canker/fresh/grenning`)
- Public project GitHub URL, view-accessible Colab URL, 30-minute walkthrough URL, and the candidate's own Q4 answer

### Fixes applied during this audit

- The demo now saves available model predictions as JSON under `outputs/predictions/` (or a caller-selected `--output` path).
- Added tests for inference-summary serialization, path resolution and CLI/training orchestration; extended synthetic integration checks to verify saved predictions.
- Corrected the README's classifier-head description from batch normalization to the implemented `LayerNorm`.
- Added a README warning that the configured Drive link currently resolves to a non-ripeness dataset.
- Updated the downloader to warn when the incompatible default Drive URL is used.
- Added a missing-metrics guard to the comparison notebook so it cannot create a misleading empty CSV.
- Added `pytest-cov` for the requested coverage command and ignored coverage artifacts in Git.
