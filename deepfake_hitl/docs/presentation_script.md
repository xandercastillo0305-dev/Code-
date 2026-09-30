# Presentation & Demo Script — 60 minutes

**"Deepfake Detection for Pornographic Images with Human-in-the-Loop"**
BS Computer Science · National University

> **How to use this script**
> - `[P1]`–`[P4]` = presenters. Reassign to match your group size.
> - Text in *italics* is meant to be said out loud; bullets are talking points in your own words.
> - `[PLACEHOLDER]` = fill in from your manuscript. **Do not invent numbers**; use only your paper's figures.
> - ⏱ times are cumulative. Speaking pace is about 120–130 words per minute. Rehearse with a timer.
> - Slide numbers are left generic ("Slide: Title"). Once the PPT is shared, this can be aligned slide by slide.

---

## Time plan

| Time | Part | Presenter |
|---|---|---|
| 0:00 – 3:00 | Opening and introduction | P1 |
| 3:00 – 10:00 | Chapter 1 — The problem | P1 |
| 10:00 – 17:00 | Chapter 2 — Related literature and research gap | P2 |
| 17:00 – 32:00 | Chapter 3 — Methodology and system design | P2 → P3 |
| 32:00 – 47:00 | **Live system demo** | P3 (drives) + P4 (narrates) |
| 47:00 – 54:00 | Testing, results and limitations | P4 |
| 54:00 – 58:00 | Conclusions and recommendations | P1 |
| 58:00 – 60:00 | Closing | P1 |

Keep **2–3 minutes of buffer** inside the demo block; that's where time usually runs over.

---

## 0:00 – 3:00 · Opening `[P1]`

**Slide: Title**

*Good [morning/afternoon], honorable panel members, [adviser's name], and everyone present. We are [group name / members' names], and today we present our thesis, "Deepfake Detection for Pornographic Images with Human-in-the-Loop."*

*Our study builds a digital forensic tool that helps investigators and forensic analysts determine whether an image of a person has been manipulated with deepfake technology, especially in cases of non-consensual pornographic deepfakes. Our main idea is simple: artificial intelligence gives a preliminary assessment, but a trained human analyst always makes the final decision.*

**Slide: Presentation outline**

*We will first discuss the problem and our objectives, then the related literature, our methodology and system design, a live demonstration of the prototype, our testing and results, and finally our conclusions and recommendations.*

---

## 3:00 – 10:00 · Chapter 1 — The Problem `[P1]`

**Slide: Background of the study** (≈2 min)
- Deepfakes: AI-generated or AI-altered media that swaps or modifies a person's face.
- The most harmful use is **non-consensual pornographic deepfakes**: a real person's face placed onto explicit content without consent.
- `[PLACEHOLDER: the statistic from your RRL, e.g. share of online deepfakes that are pornographic, with citation. Use only the figure and source in your manuscript.]`
- `[PLACEHOLDER: local context, e.g. Philippine law / cases / agencies cited in your paper.]`

*Victims suffer reputational, psychological and legal harm, and investigators need a reliable way to examine this kind of evidence.*

**Slide: Statement of the problem** (≈2 min)
- `[PLACEHOLDER: read your general problem and specific problems exactly as written in Chapter 1.]`
- Key points to stress:
  - Fully automated detectors can be wrong, and in a forensic setting a wrong verdict has serious consequences.
  - Existing tools rarely keep the AI's output **separate** from the human expert's conclusion.
  - Handling explicit evidence requires strict privacy and chain-of-custody controls.

**Slide: Objectives** (≈1.5 min)
- `[PLACEHOLDER: your general and specific objectives.]`
- Map each objective to a system feature (you'll show them in the demo):
  - Hybrid CNN–Transformer detection → AI analysis module
  - Human-in-the-loop verification → analyst review
  - Forensic reporting → Appendix B report
  - Evaluation → accuracy, F1, ROC-AUC, agreement rate, Cohen's κ

**Slide: Scope and delimitations** (≈1.5 min)

*Our system analyzes the **facial region** of an image. It detects face-swap and face-manipulation deepfakes, which is how most pornographic deepfakes are made: a victim's face is placed onto another person's body. Manipulations limited to the body or background are outside the AI's scope and are handled by the analyst's review of the full image. The system covers adult subjects only, and every submission requires an adult-subject and authorization attestation.*

- `[PLACEHOLDER: any other delimitations in Section 1.5, e.g. images only, not video.]`

**Slide: Significance of the study** (≈30 s)
- Law enforcement / cybercrime units, forensic analysts, victims, future researchers.

---

## 10:00 – 17:00 · Chapter 2 — Related Literature `[P2]`

**Slide: How deepfakes are made** (≈1.5 min)
- Face swapping (autoencoders, GANs), face reenactment, fully synthetic faces.
- Common artefacts: blending boundaries, lighting mismatch, skin-texture inconsistency, warping.

**Slide: Deepfake detection approaches** (≈2.5 min)
- CNN-based detectors: **XceptionNet** (standard FaceForensics++ baseline), **EfficientNet** (strong accuracy per parameter; B4 uses 380×380 input).
- Transformer-based: **Vision Transformer (ViT)** captures global relationships across the whole face through self-attention.
- Hybrid CNN + Transformer: local texture detail + global consistency.
- Similarity/verification approaches: compare a suspect image to a trusted reference of the same person.
- `[PLACEHOLDER: cite the specific studies from your RRL for each.]`

**Slide: Human-in-the-loop in forensics** (≈1.5 min)
- AI as decision support, not decision maker; human accountability in evidence handling.
- `[PLACEHOLDER: your HITL / digital forensics references.]`

**Slide: Research gap / synthesis** (≈1.5 min)

*Most existing work focuses only on detection accuracy. Few systems combine a hybrid detector with a structured human review workflow, a report that clearly separates automated from human-verified findings, and built-in privacy protections for sensitive material. Our study addresses that gap.*

---

## 17:00 – 32:00 · Chapter 3 — Methodology and System Design `[P2 → P3]`

### `[P2]` Conceptual framework (17:00 – 19:00)

**Slide: Conceptual framework — Input → Process → Human Review → Output**

*The framework has four stages. **Input**: an authorized investigator submits a suspect image and a reference image of the same adult subject. **Process**: our AI pipeline analyzes both faces. **Human Review**: a trained forensic analyst examines the result and confirms, overrides, or flags it, with a written rationale. **Output**: a forensic report that keeps the automated analysis separate from the human-verified conclusion.*

### `[P2]` The AI pipeline (19:00 – 24:00)

**Slide: System pipeline (Section 3.3)**

*Preprocessing → EfficientNet-B4 and Transformer in parallel → Feature Fusion → Multi-Metric Similarity → Preliminary Classification → Human Review → Report.*

Walk through each module (≈45 s each):

1. **Preprocessing.** *MTCNN detects the largest face, aligns it so the eyes are level, crops it with a small margin, resizes it to 380×380, and normalizes it. If no face is found, the system refuses to analyze the case instead of guessing.*
2. **CNN branch — EfficientNet-B4.** *Extracts **local** features such as texture, edges and blending artefacts, giving a 1,792-dimensional feature vector.*
3. **Transformer branch — Vision Transformer.** *Runs in parallel on the same face and captures **global** relationships across the whole face through self-attention. Because the ViT works on 16×16 patches, the face is resized to 384×384 inside this branch only.*
4. **Feature fusion.** *The two feature vectors are concatenated and projected into a single 512-dimensional embedding, then normalized.*
5. **Multi-metric similarity.** *We compare the suspect's embedding with the reference's using three metrics: cosine similarity, Euclidean-based similarity, and SSIM, the structural similarity of the two aligned face crops.*
6. **Preliminary classification.** *If S is at least the threshold τ, the result is "Real"; otherwise it is "Deepfake", along with a confidence score.*

### `[P3]` Algorithm and formulas (24:00 – 27:00)

**Slide: Algorithm (Section 3.6)**
- Read the algorithm steps from your manuscript. `[PLACEHOLDER: confirm your step numbering matches the code comments in pipeline/verify.py.]`

**Slide: Formulas (Appendix A)**

| Metric | Formula |
|---|---|
| Cosine | cos(A,B) = A·B / (‖A‖‖B‖), mapped to [0,1] as (cos+1)/2 |
| Euclidean | d(A,B) = √Σ(Aᵢ−Bᵢ)², similarity = 1 − d/2 |
| SSIM | structural similarity of the aligned face crops |
| Aggregated | S = w₁·cos + w₂·euc + w₃·ssim, weights sum to 1 (default ⅓ each) |
| Decision | Real if S ≥ τ (τ = 0.70), else Deepfake |
| Confidence | 0.5 + 0.5 · min(1, \|S − τ\| / max(τ, 1 − τ)) |

*SSIM needs two-dimensional image data, so it is computed on the aligned face crops rather than on the embedding vector.*

### `[P3]` Human-in-the-loop workflow (27:00 – 29:00)

**Slide: Roles and review workflow**
- **Investigator** submits cases; **Analyst** reviews; **Admin** manages users and evaluation.
- Analyst decisions:
  - **Confirm** → final = AI result → status *Verified*
  - **Override** → final = the opposite class → status *Verified*
  - **Flag** → final = *Inconclusive* → status *Flagged*
- A written rationale is always required. Reviewed cases become read-only. An analyst can't review their own submission.

**Slide: Case record and forensic report (Appendices B & C)**
- Every case stores the Appendix C fields, including SHA-256 hashes for chain of custody.
- Report sections: I Case Information · II Submitted Images · **III AI Analysis (AUTOMATED)** · **IV Human Review (HUMAN-VERIFIED)** · V Sign-off.

### `[P3]` Data, training, evaluation, tech stack, ethics (29:00 – 32:00)

**Slide: Datasets and training (Section 3.5)**
- FaceForensics++, Celeb-DF, DFDC, obtained through the official request forms.
- Face crops extracted with the same MTCNN pipeline; augmentation: flip, rotation, scaling.
- BCE loss, AdamW, early stopping on validation ROC-AUC; trained on Google Colab GPU.

**Slide: Evaluation metrics (Section 3.7)**
- Detection: accuracy, precision, recall, F1, ROC-AUC (Deepfake = positive); baseline: XceptionNet.
- Human–AI: agreement rate and Cohen's κ = (pₒ − pₑ) / (1 − pₑ).

**Slide: Tech stack (Section 3.8)**
- Python, PyTorch + timm, facenet-pytorch (MTCNN), scikit-image, scikit-learn, Flask, ReportLab, pytest.

**Slide: Ethical safeguards (Appendix D)**
- Adults-only attestation · uploads stored outside public folders · face crops only, full images blurred by default · full-image reveal is analyst-only and logged · append-only audit log · admin-only secure purge.

*Now we will show how all of this works in our prototype.*

---

## 32:00 – 47:00 · Live System Demo `[P3 drives, P4 narrates]`

> **Before the defense (the day before and 30 minutes before)**, see the checklist at the end of this script.
> Use only the NASA demo images in `demo\samples\` (public domain, non-explicit).

### 1 · Start the app (32:00 – 33:00)
`[P3]` In the VS Code terminal:
```powershell
.venv\Scripts\activate
flask --app app run
```
Open **http://127.0.0.1:5000** in Chrome.

`[P4]` *The system runs locally on a CPU laptop. The red banner says "UNTRAINED MODEL". Our paper states this limitation: until the fine-tuned weights are loaded, the AI results are not indicative of real accuracy, and the system says so on every page and every report.*

> If your model is trained by the defense date, the banner will not appear. Skip this line and say the model is loaded with fine-tuned weights.

### 2 · Investigator submits a case (33:00 – 36:30)
`[P3]` Log in as **INV-01** / password.

`[P4]` *We are now an authorized investigator. Investigators can only submit cases and view their own reports.*

`[P3]` Click **Submit case** → Suspect: `suspect_manipulated.png` → Reference: `reference.png`.

`[P4]` *The suspect image is the one under examination. The reference is a known authentic photo of the same person. Our demo uses a public-domain NASA portrait; the suspect has a synthetic edit to the inner face region.*

`[P3]` Point at the attestation checkbox, tick it.

`[P4]` *Every submission requires this attestation: the subject is an adult and the investigator is authorized. This is from our ethics protocol in Appendix D.*

`[P3]` Click **Submit for analysis**.

`[P4]` *The system now computes a SHA-256 hash of each file for chain of custody, detects and aligns the faces, runs both the EfficientNet and Transformer branches, computes the three similarity metrics, and produces a preliminary classification.*

`[P3]` Show "My cases".

`[P4]` *The case is **Pending**. The investigator does not see the AI's result. No case is final until a forensic analyst reviews it.*

### 3 · No-face rejection (36:30 – 37:30)
`[P3]` Submit `no_face.png` as suspect + `reference.png`.

`[P4]` *If no face is detected, the system refuses: "No face detected. Case cannot be analyzed." It never guesses. The rejection is still recorded in the audit log.*

### 4 · Analyst dashboard (37:30 – 38:30)
`[P3]` Log out → log in as **ANA-01**.

`[P4]` *Now we are the forensic analyst. The dashboard shows pending, verified and flagged counts, and the live agreement rate between the AI and human reviewers. Each case shows its preliminary AI classification and aggregated score.*

### 5 · Review interface (38:30 – 43:30) — the core of our study
`[P3]` Click **Review** on the new case.

`[P4]` *This is the core of our human-in-the-loop design.*
- *Left: the aligned **face crops** only. The full images are **pixelated** by default. This is our redaction protocol.*
- *Right, in blue, marked **AUTOMATED**: the AI's preliminary classification, confidence, the three similarity metrics, and the aggregated score S compared with the threshold τ.*
  - `[P4 reads the actual values on screen: classification, confidence, S, τ.]`

`[P3]` Click **Reveal full image** on the suspect → confirm.

`[P4]` *If the analyst needs to see the full image, they must reveal it deliberately. Only analysts can do this, and each reveal is written to the audit log with the analyst's ID and time.*

`[P3]` In the green **HUMAN-VERIFIED** section, choose the decision:
- If the AI said **Real** → choose **Override** (final becomes Deepfake).
- If the AI said **Deepfake** → choose **Confirm**.

`[P3]` First type something short (e.g. "fake") and click submit to show the error.

`[P4]` *A rationale is mandatory, at least 30 characters, checked both in the browser and on the server. The analyst must justify the decision.*

`[P3]` Type the rationale, e.g. for Override:
> Although the AI scored the image as Real, the inner face region shows warping around the eyes and nose bridge, and its colour saturation differs from the forehead and neck. These localized distortions are inconsistent with the reference image. Overriding to Deepfake.

`[P3]` Submit.

`[P4]` *(If it was an override:) This is exactly why human review matters. The AI's preliminary result was wrong, and the analyst corrected it with documented reasoning.*

### 6 · Forensic report (43:30 – 45:30)
`[P3]` The report page opens. Scroll slowly.

`[P4]` *This follows our Appendix B template:*
- *Section I: case information: who submitted, who reviewed.*
- *Section II: only **blurred** face crops, plus the SHA-256 hashes.*
- *Section III, in blue: the **AUTOMATED** AI analysis, labeled as preliminary and not a verdict.*
- *Section IV, in green: the **HUMAN-VERIFIED** conclusion: decision, final classification and rationale.*
- *Section V: the analyst's sign-off.*

`[P3]` Click **Download PDF** and open it.

`[P4]` *The same report is available as a PDF for case files. The case is now read-only; its review cannot be changed.*

### 7 · Admin: evaluation and audit log (45:30 – 47:00)
`[P3]` Log out → log in as **ADM-01** → **Evaluation**.

`[P4]` *The admin's evaluation page computes the agreement rate and Cohen's kappa between AI and analyst decisions, directly from the case records. These are the human-AI metrics from Section 3.7.*

`[P3]` Click **Audit log**.

`[P4]` *And this is the append-only audit log: every login, submission, rejection, image reveal, review and report download, with the user and timestamp. Administrators can also securely purge cases for data retention, and that deletion is logged too.*

*That concludes our demonstration.*

> **Backup plan:** if the app or laptop fails, open `docs\walkthrough\` and present the screenshots in order (01 → 13) plus `CASE-0002_forensic_report.pdf`, using the same narration.

---

## 47:00 – 54:00 · Testing, Results and Limitations `[P4]`

**Slide: System testing** (≈1.5 min)
- *The system has 95 automated tests covering the similarity formulas, the classifier, the manual Cohen's kappa (checked against scikit-learn), the review workflow, access control, and a full end-to-end run from submission to PDF report. All 95 pass.*

**Slide: Detection results (Chapter 4)** (≈2.5 min)
- `[PLACEHOLDER: comparison table from evaluation/results/comparison.md: accuracy, precision, recall, F1, ROC-AUC for the hybrid model vs XceptionNet.]`
- `[PLACEHOLDER: confusion matrix and ROC curve images from evaluation/results/.]`
- `[PLACEHOLDER: calibrated τ from calibrate_threshold.py.]`
- **If training is not finished:** say so directly. *"The prototype and evaluation pipeline are complete; detection results will be produced after training on the requested datasets."* Do not present results from the untrained model as findings.

**Slide: Human–AI agreement** (≈1 min)
- `[PLACEHOLDER: agreement rate, κ, and confirm / override / flag counts from your analyst evaluation.]`

**Slide: Limitations** (≈2 min)
- The AI analyzes the **face only**; body or background edits rely on the analyst.
- Training datasets are non-explicit; performance on explicit images may differ (**domain gap**).
- SSIM is sensitive to pose and lighting differences between suspect and reference; τ calibration reduces this.
- A face-swap using the victim's own face can look similar to the reference, so detection depends on the trained model's sensitivity to artefacts, which is another reason for human review.
- Local prototype: no production deployment or scaling.

---

## 54:00 – 58:00 · Conclusions and Recommendations `[P1]`

**Slide: Conclusions**
- `[PLACEHOLDER: one conclusion per specific objective, based on your actual results.]`
- *Our study shows that a hybrid CNN–Transformer pipeline can be combined with a structured human-in-the-loop workflow, so that the AI supports the analyst without replacing them, and the final report clearly separates automated from human-verified findings.*

**Slide: Recommendations**
- Train and test on larger and more diverse datasets, including authorized domain-specific data under formal ethics approval.
- Extend to video deepfakes.
- Add detection of body and background manipulation.
- Let flagged cases be reopened and resolved after further investigation.
- Deploy on a secured server with encryption at rest and multi-factor authentication.
- Conduct a user study with actual forensic analysts to measure agreement and usability.

---

## 58:00 – 60:00 · Closing `[P1]`

*To summarize: deepfakes, especially non-consensual pornographic deepfakes, cause serious harm, and automated detection alone is not enough for forensic use. Our system combines a hybrid CNN–Transformer model with multi-metric similarity scoring and, most importantly, a trained human analyst who makes the final, documented decision.*

*Thank you for your time and attention. We are now ready for your questions.*

---

## Pre-defense checklist

**The day before**
- [ ] `git pull`, then `.venv\Scripts\activate` and `python -m pytest` → **95 passed**.
- [ ] Run one full case so the ImageNet weights are downloaded and cached.
- [ ] Start with a clean slate: stop the app, **rename** the `data` folder to `data_practice` (keeps your practice cases), then run `python seed_users.py --password <your demo password>` and `python demo/make_demo_samples.py`. The demo case will then be CASE-0001.
- [ ] Rehearse the demo block twice with a timer (target: 15 minutes).
- [ ] Charge the laptop, and bring an HDMI adapter and a phone hotspot (the page styling loads from the internet).

**30 minutes before**
- [ ] Activate `.venv`, run `flask --app app run`, open Chrome at http://127.0.0.1:5000.
- [ ] Open `demo\samples\` in File Explorer for quick file picking.
- [ ] Open `docs\walkthrough\` as the backup.
- [ ] Zoom the browser to about 125% so the panel can read it.
- [ ] Close unrelated apps and turn off notifications.

---

## Likely panel questions (quick answers)

| Question | Short answer |
|---|---|
| Why does the system say "UNTRAINED MODEL"? | Fine-tuned weights aren't loaded yet; the system flags this on every page and report so no one relies on the AI result (Section 3.9 limitation). |
| Why not let the AI decide? | Forensic decisions have legal consequences; the AI can be wrong (as shown in the demo); the analyst provides accountability through a documented rationale. |
| Why demo with non-explicit images? | The AI only analyzes the face crop, so the pipeline is identical; our ethics protocol forbids explicit material outside authorized casework. |
| What if only the body is edited? | Outside the AI's scope (face only); the analyst reviews the full image and can override or flag. |
| Why SSIM on face crops, not embeddings? | SSIM is a spatial measure that needs 2-D image data. |
| Why resize to 384 in the ViT branch? | Patch-16 ViTs need an input size divisible by 16; 380 is not. |
| How was τ = 0.70 chosen? | Default from our design; calibrated on the validation set by maximizing F1 (or Youden's J). `[PLACEHOLDER: your calibrated value.]` |
| Why are flagged cases excluded from κ? | "Inconclusive" isn't a class the AI can output; flags are reported separately. |
| How do you protect evidence? | SHA-256 hashes, storage outside public folders, role-based access, blurred views, logged reveals, audit log, secure purge. |
