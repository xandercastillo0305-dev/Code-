# Deepfake Detection for Pornographic Images with Human-in-the-Loop — Prototype

Web-based digital-forensic prototype for the thesis *"Deepfake Detection for Pornographic
Images with Human-in-the-Loop"* (BS Computer Science, National University).

An authorized **investigator** uploads a suspect image and a reference image of the same
adult subject. The AI pipeline produces a **preliminary** Real/Deepfake classification. It
is never the final verdict. A trained **forensic analyst** reviews the case and confirms,
overrides or flags it with a written rationale. The system then generates a forensic
report that keeps the **automated** analysis separate from the **human-verified** conclusion.

```
Input  →  Process  →  Human Review  →  Output
Preprocessing → [EfficientNet-B4 ∥ Transformer] → Feature Fusion → Multi-Metric Similarity
→ AI-Assisted Preliminary Classification → Human-in-the-Loop Review → Forensic Report
```

> ⚠️ **UNTRAINED MODEL** — until `weights/hybrid_best.pt` exists, the system runs with
> ImageNet backbones and an untrained fusion layer. Every page and report shows
> "UNTRAINED MODEL: results not indicative of real accuracy" (Section 3.9 limitation).

---

## 1. Setup

```bash
cd deepfake_hitl
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps facenet-pytorch==2.6.0
```

`facenet-pytorch` (MTCNN) is installed with `--no-deps` because its package metadata pins
`torch<2.3`, which would downgrade PyTorch. Its MTCNN code works with current PyTorch; the
MTCNN weights ship inside the package, so no download is needed.

On first use, `timm` downloads the ImageNet weights for EfficientNet-B4 and ViT-S/16 from
Hugging Face (about 100 MB). If there is no internet, the model falls back to random init.
It still runs, and it is flagged as untrained.

## 2. Seed users and run

```bash
python seed_users.py                 # or: flask --app app seed-users
python run.py                        # starts the app AND opens http://127.0.0.1:5000/login in your browser
```

**Windows shortcut:** double-click **`start.bat`** in the `deepfake_hitl` folder. It activates `.venv`, starts
the app and opens the browser. Close the black window (or press Ctrl+C in it) to stop the app.
`flask --app app run` still works if you prefer to open the link yourself.

The address is `http://` (not `https://`) because the app runs only on your own computer (127.0.0.1);
nothing is sent over the network.

| User ID | Role | Default password |
|---|---|---|
| `INV-01` (or `inv-01`) | Investigator (submits cases, sees own reports) | `demo1234` |
| `ANA-01`, `ANA-02` | Analyst (reviews cases) | `demo1234` |
| `ADM-01` | Admin (users, evaluation page, audit log, purge) | `demo1234` |

On the login page, choose the matching **Log in as** role (Investigator, Analyst or Administrator). A correct password with the wrong role is refused and logged.

Change the passwords with `python seed_users.py --password ...` (on an empty `data/users.json`)
and set `DFHITL_SECRET_KEY` before any real use.

### Demo images (non-explicit only)

```bash
python demo/make_demo_samples.py     # writes demo/samples/*.png (see demo/README.md)
```

## 3. Using the system

1. **Investigator** → *Submit case*: upload the suspect and reference images (JPG/PNG, ≤10 MB),
   tick the adults-only / authorization attestation, and submit. The SHA-256 of each file is
   recorded, the pipeline runs, and the case enters the queue as `pending`.
   If no face is found, the message is *"No face detected. Case cannot be analyzed."* and no case is created.
2. **Analyst** → *Dashboard*: shows pending/verified/flagged counts and the live AI–human agreement rate.
   You can filter by status. Open a pending case with *Review*.
3. **Review page**: aligned face crops are shown. Full images are pixelated until you click
   *Reveal full image*, and every reveal is logged. The page also shows the preliminary label,
   confidence, cosine / Euclidean / SSIM bars, and a gauge of S against τ.
   Choose **Confirm**, **Override** or **Flag**, write a rationale (at least 30 characters), then submit.
   The case becomes read-only and the forensic report (HTML + PDF) is generated.
4. **Investigator** → *My cases*: after review, view or download the report.
5. **Admin** → *Evaluation* (agreement, κ, confirm/override/flag counts), *Users*, *Audit log*.

## 4. Training (Section 3.5)

Datasets are **not** included or downloaded. Obtain FaceForensics++, Celeb-DF and DFDC through
their official request forms. Extract frames (e.g. `ffmpeg -i v.mp4 -vf fps=1 frames/v_%04d.png`), then:

```bash
python -m training.extract_faces --src raw/ffpp/original_frames --dst data/train/real
python -m training.extract_faces --src raw/ffpp/Deepfakes_frames --dst data/train/fake
# ...same for data/val and data/test
python -m training.train          --data-root data --epochs 20 --batch-size 8
python -m training.train_baseline --data-root data --epochs 10          # XceptionNet baseline
```

`train.py` uses ImageNet-pretrained backbones and augmentation (horizontal flip, ±10° rotation,
0.9–1.1 scaling). It trains with BCE loss on the classification head and AdamW, stops early on
validation ROC-AUC, and saves the best model to `weights/hybrid_best.pt`.

### Colab (`--colab` note)

```python
# Runtime → Change runtime type → GPU
from google.colab import drive; drive.mount('/content/drive')
!git clone <your repo url> && cd <repo>/deepfake_hitl
!pip install -q timm scikit-image reportlab flask flask-login && pip install -q --no-deps facenet-pytorch==2.6.0
!python -m training.train --data-root /content/drive/MyDrive/faces --epochs 20 --batch-size 16 --amp \
    --out /content/drive/MyDrive/weights/hybrid_best.pt
```

Colab already has PyTorch with CUDA, so do **not** reinstall torch. `--amp` enables mixed precision.
Copy `hybrid_best.pt` into `weights/` on the demo machine, and the UNTRAINED banner disappears.

## 5. Calibrating τ

```bash
python -m training.build_pairs --split-dir data/val            # writes data/val/pairs.csv (check it!)
python -m pipeline.calibrate_threshold --pairs data/val/pairs.csv --method f1       # or --method youden
```

This prints the τ that maximizes F1 (or Youden's J). Put it in `config.THRESHOLD_TAU` (default 0.70).

## 6. Evaluation (Section 3.7)

```bash
python -m evaluation.evaluate_model --data-root data           # hybrid vs XceptionNet (+ pairs if present)
python -m evaluation.evaluate_agreement                        # AI vs analyst from data/cases.json
```

Results for Chapter 4 go to `evaluation/results/`: `comparison.md/.csv`, `confusion_*.png`
and `roc_curve.png`.

## 7. Tests

```bash
python -m pytest            # all tests (about 20 s on CPU)
python -m pytest -m "not slow"
```

MTCNN and the hybrid model are replaced by small stubs in the web/end-to-end tests, so CI needs no downloads.
`test_pipeline.py::test_real_hybrid_architecture_shapes` builds the real EfficientNet-B4 ∥ ViT
model (random init) and checks the tensor shapes.

## 8. Ethics and data handling (Appendix D)

* The adults-only / authorization attestation is required for every submission.
* Uploads live in `data/uploads/` (outside `static/`) and are served only by authenticated, role-checked routes.
* Face crops only by default. Full images are pixelated, and a reveal is analyst-only and logged
  (`image_revealed`). Reports and investigators only ever see blurred face crops.
* Append-only audit log `data/audit_log.jsonl`: login, case submitted/rejected, case viewed, image revealed,
  review submitted, report viewed/downloaded, case deleted, user changes.
* Data-retention purge (admin credentials required, logged as `case_deleted`):
  ```bash
  flask --app app purge-case CASE-0001
  flask --app app purge-all
  ```
  Files are overwritten with random bytes before deletion. On SSDs and journaling filesystems this
  does not guarantee physical erasure, so use full-disk encryption on any deployment machine.

---

## 9. Paper ↔ Code mapping

| Thesis section | What | File → function |
|---|---|---|
| 3.2 Conceptual framework | Input → Process → Human Review → Output | `app.py` → `submit_case` → `pipeline/verify.py::verify_media` → `review_case` → `reports/report_generator.py::generate_pdf` |
| 3.3 Preprocessing module | MTCNN detect, largest face, align, crop, 380×380, ImageNet normalization | `pipeline/preprocessing.py::FacePreprocessor.process_image`, `align_and_crop`, `to_tensor` |
| 3.3 CNN feature extraction | EfficientNet-B4, pooled 1792-d F₁ | `pipeline/model.py::HybridCNNTransformer.__init__` (`self.cnn`), `branch_features` |
| 3.3 Transformer global attention | ViT-S/16 CLS embedding F₂ (input resized 380→384 in-branch) | `pipeline/model.py::HybridCNNTransformer.branch_features` (`self.vit`) |
| 3.3 Feature fusion | concat(F₁,F₂) → Linear → LayerNorm → L2 (512-d) | `pipeline/model.py::HybridCNNTransformer.fuse`, `forward` |
| 3.3 Multi-metric similarity | cosine, Euclidean, SSIM, weighted S | `pipeline/similarity.py` |
| 3.3 Preliminary classification | S ≥ τ → Real, else Deepfake; confidence | `pipeline/classifier.py::classify`, `confidence` |
| 3.3 / 3.4 Human-in-the-loop review | Confirm / Override / Flag + rationale | `review/workflow.py::apply_review`, `submit_review`; UI `templates/review.html` |
| 3.3 Forensic report | HTML + PDF, Appendix B layout | `reports/report_generator.py::build_report_data`, `generate_pdf`; `templates/report.html` |
| 3.4 Roles and access control | Investigator / Analyst / Admin | `users.py`, `app.py::roles_required`, `can_view_case` |
| 3.4 Case queue | JSON store, thread-safe, atomic save | `review/case_store.py::CaseStore` |
| 3.4 Reviewer dashboard | counts, filter, live agreement | `app.py::dashboard`, `templates/dashboard.html` |
| 3.5 Training | augmentation, BCE, AdamW, early stopping on val AUC | `training/train.py::main`, `training/dataset.py::train_transform` |
| 3.5 Datasets | FF++ / Celeb-DF / DFDC face-crop layout | `training/dataset.py::FaceCropDataset`, `training/extract_faces.py` |
| 3.5 / 3.7 Baseline | XceptionNet (timm `legacy_xception`) | `training/train_baseline.py` |
| 3.6 Algorithm **Line 1** | procedure VERIFY_MEDIA(I, R, τ) | `pipeline/verify.py::verify_media` (signature) |
| 3.6 **Lines 2, 10** | for each X ∈ {I, R} … end for | `verify_media` (both images processed as one batch) |
| 3.6 **Line 3** | MTCNN detects the largest face | `preprocessing.py::FacePreprocessor.detect_largest_face` |
| 3.6 **Line 4** | no face → "No face detected. Case cannot be analyzed." | `preprocessing.py::NoFaceDetectedError`; handled in `app.py::submit_case` |
| 3.6 **Line 5** | align, crop, resize 380×380 | `preprocessing.py::align_and_crop` |
| 3.6 **Line 6** | normalize (ImageNet mean/std) | `preprocessing.py::to_tensor` |
| 3.6 **Line 7** | F₁ ← EfficientNet-B4 | `model.py::branch_features` (`self.cnn`) |
| 3.6 **Line 8** | F₂ ← Vision Transformer (resized to 384×384) | `model.py::branch_features` (`self.vit`) |
| 3.6 **Line 9** | X_fused ← L2(LayerNorm(Linear(concat))) | `model.py::fuse` + L2 normalize in `verify_media` |
| 3.6 **Line 11** | Score_Cos = (cos+1)/2 | `similarity.py::cosine_similarity`, `cosine_to_unit` |
| 3.6 **Line 12** | Score_Euc = 1 − d/2 | `similarity.py::euclidean_distance`, `euclidean_similarity` |
| 3.6 **Line 13** | Score_SSIM on aligned face crops | `similarity.py::ssim_score` |
| 3.6 **Line 14** | S = Σ wᵢ·metricᵢ | `similarity.py::aggregate_score` (weights in `config.METRIC_WEIGHTS`) |
| 3.6 **Line 15** | if S ≥ τ Real else Deepfake | `classifier.py::classify` |
| 3.6 **Line 16** | confidence score | `classifier.py::confidence` |
| 3.6 **Line 17** | add case to the review queue as "pending" | `app.py::submit_case` → `review/case_store.py::CaseStore.add_case` |
| 3.6 **Line 18** | Human_In_The_Loop_Review → D, Rationale | `app.py::review_case` → `review/workflow.py::submit_review`, `apply_review` |
| 3.6 **Line 19** | Generate_Forensic_Report | `reports/report_generator.py::generate_pdf`, `build_report_data` |
| 3.6 **Line 20** | return C, S, D, FR | stored in the case record (`ai_classification`, `aggregated_score`, `final_classification`, `report_path`) |
| 3.6 τ selection | F1 / Youden's J calibration | `pipeline/calibrate_threshold.py` |
| 3.7 Detection metrics | accuracy, precision, recall, F1, ROC-AUC, confusion matrix | `evaluation/metrics.py::classification_metrics`; `evaluation/evaluate_model.py` |
| 3.7 Agreement metrics | agreement rate, Cohen's κ (sklearn + manual) | `evaluation/metrics.py::agreement_rate`, `cohens_kappa`, `cohens_kappa_manual`; `evaluation/evaluate_agreement.py`; admin page `/evaluation` |
| 3.8 Tech stack | Python, PyTorch, timm, facenet-pytorch, scikit-image, scikit-learn, Flask, ReportLab, pytest | `requirements.txt` |
| 3.9 Implementation / testing | one module per stage, unit + end-to-end tests | folder layout, `tests/` |
| 3.9 Limitation (untrained model) | visible banner in UI and every report | `pipeline/model.py::load_model` (`model_trained`), `templates/base.html`, `report_generator.py::UNTRAINED_BANNER` |
| Appendix A | formulas (cosine, Euclidean, SSIM, S, metrics, κ) | `pipeline/similarity.py`, `pipeline/classifier.py`, `evaluation/metrics.py` (docstrings list each formula) |
| Appendix B | report template, sections I–V | `reports/report_generator.py`, `templates/report.html` |
| Appendix C | case record data dictionary | `review/case_store.py::CASE_FIELDS` (+ `EXTRA_FIELDS`) |
| Appendix D #1 | adults-only attestation | `app.py::submit_case`, `templates/submit.html` |
| Appendix D #2 | chain of custody (SHA-256) | `storage.py::sha256_bytes`, fields `suspect_sha256`, `reference_sha256` |
| Appendix D #3–4 | redaction: face crops, blur, logged reveal | `app.py::case_image`, `reveal_image`; `storage.py::redact`; `static/review.js` |
| Appendix D audit | append-only audit log | `review/audit_log.py::log_event` |
| Appendix D retention | secure purge CLI | `app.py::purge_case`, `purge-case` / `purge-all` commands; `storage.py::secure_delete_file` |
| Appendix D demo | non-explicit demo data only | `demo/README.md`, `demo/make_demo_samples.py` |

> **Line numbers** follow the revised Section 3.6 algorithm (lines 1–21). The comments in
> `pipeline/verify.py` and `app.py` use the same "Line N" labels. Lines 2–17 run automatically at
> submission; lines 18–20 run when the analyst submits the review.

## 10. Notes for the manuscript

* **SSIM input.** SSIM is a spatial measure and needs 2-D image data, so it is computed on the
  aligned grayscale 380×380 **face crops** of the suspect and reference images, not on the
  fused 1-D embedding. The Section 3.6 / Appendix A text should say this explicitly.
* **Metric ranges.** Cosine is stored as (cos+1)/2 and Euclidean as 1 − d/2 (d on unit
  vectors, 0–2), so all three metrics and S lie in [0, 1]. SSIM is clipped to [0, 1].
* **ViT input.** The ViT branch internally resizes the 380×380 face to 384×384, because
  patch-16 ViTs need a size divisible by 16.
* **Classification head** is used only in training. Inference uses the similarity/threshold rule.
* See the chat summary / PR description for the full list of implementation choices to confirm.

## 11. Project layout

```
deepfake_hitl/
  app.py  config.py  users.py  storage.py  seed_users.py
  pipeline/    preprocessing.py model.py similarity.py classifier.py verify.py calibrate_threshold.py
  review/      case_store.py workflow.py audit_log.py
  reports/     report_generator.py
  evaluation/  metrics.py evaluate_model.py evaluate_agreement.py results/
  training/    dataset.py train.py train_baseline.py extract_faces.py build_pairs.py
  templates/   base login submit my_cases dashboard review report evaluation users audit error
  static/      style.css review.js
  data/        runtime: cases.json users.json audit_log.jsonl uploads/ reports/   (git-ignored)
  weights/     hybrid_best.pt xception_best.pt                                      (git-ignored)
  demo/        README.md make_demo_samples.py
  tests/
```
