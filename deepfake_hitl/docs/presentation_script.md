# Thesis Defense Script — Full 2-Hour Flow

**"Deepfake Detection for Pornographic Images with Human-in-the-Loop"**
National University • CCIT • Adviser: Ms. Susan S. Caluya

> **How to use this script**
> - It covers the whole defense: call to order, prayer, the 60-minute presentation (your 15 slides, with the **live demo after Slide 13**), the 45-minute Q&A, and the panel deliberation.
> - **Tristan (leader)** opens, presents the conceptual framework, closes, and leads the Q&A. Each member owns one continuous block, and every block ends with a hand-off line naming the next speaker.
> - *Italics* = say it out loud. You can paraphrase; don't read the slide word for word.
> - ⏱ = cumulative time. Speaking pace is about 120–130 words per minute. Rehearse with a timer.
> - Read **"Slide fixes before the defense"** at the end first. A few slide lines don't match the system and the panel could catch them.

---

## Defense day flow (2 hours)

| Clock | Segment | Duration | Who |
|---|---|---|---|
| 0:00 – 0:02 | Call to order | 2 min | Panel chair / moderator (Tristan, if the group is asked to) |
| 0:02 – 0:05 | Opening prayer | 3 min | Genesis leads |
| 0:05 – 1:05 | Presentation and system demonstration | 60 min | All five members (see the time plan below) |
| 1:05 – 1:50 | Question and answer | 45 min | Tristan moderates · everyone answers |
| 1:50 – 2:00 | Panel deliberation | 10 min | Panel only; the group waits outside |

**Before the call to order**, the laptop must already be connected to the projector, the app running, and the deck on Slide 1. Nothing should be set up during the prayer or while the panel is waiting.

### Call to order (2 min)

This is normally done by the panel chair or the program moderator. If your group is asked to do it, **Tristan** says:

*Good [morning/afternoon], everyone. We now call to order the thesis defense of the group presenting "Deepfake Detection for Pornographic Images with Human-in-the-Loop," under the advisership of Ms. Susan S. Caluya. We are honored to have with us our panel members: [Panelist 1], [Panelist 2], and [Panelist 3]. To begin, may we request everyone to please stand for the opening prayer, to be led by Genesis Navarro.*

### Opening prayer (3 min) · *Genesis*

Use your school's usual prayer if there is one. Otherwise, a sample:

*Let us bow our heads and put ourselves in the presence of the Lord.*

*Heavenly Father, we thank You for this day and for the chance to present the work we have prepared. Thank You for our adviser, Ms. Susan Caluya, for our panel members, and for everyone who guided and supported us throughout this study.*

*Grant us clear minds and calm hearts, so that we may explain our research truthfully and answer every question with humility and understanding. Give our panel wisdom and fairness as they evaluate our work. May this study serve its purpose: to help protect people, especially victims, from the harm of manipulated images.*

*We offer all of this to You, and we ask for Your guidance today and always. Amen.*

*You may now take your seats.*

After the prayer, **Tristan** goes straight to Slide 1.

---

## Presentation time plan (the 60-minute block)

The ⏱ times on each slide count from the start of the presentation: **0:00 = Tristan begins Slide 1, right after the prayer** (about 0:05 on the defense clock).

| Time | Slide(s) | Speaker |
|---|---|---|
| 0:00 – 2:30 | 1 · Title | **Tristan (leader)** |
| 2:30 – 5:00 | 2 · Chapter 1 Introduction | Genesis |
| 5:00 – 9:00 | 3 · Background of the Study | Genesis |
| 9:00 – 13:00 | 4 · Statement of the Problem | Genesis |
| 13:00 – 17:00 | 5 · Objectives of the Study | Charles |
| 17:00 – 19:30 | 6 · Significance of the Study | Charles |
| 19:30 – 23:00 | 7 · Scope and Delimitations | Charles |
| 23:00 – 26:30 | 8 · Theoretical Framework | Charles |
| 26:30 – 29:30 | 9 · Conceptual Framework | **Tristan (leader)** |
| 29:30 – 33:30 | 10 · System Architecture | Mark |
| 33:30 – 36:00 | 11 · Methodology & Sequential Steps | Mark |
| 36:00 – 40:00 | 12 · Algorithm Overview | Mark |
| 40:00 – 42:30 | 13 · Datasets, Training & Evaluation | Mark |
| **42:30 – 55:00** | **Live system demonstration** | **Alexander** (Mark = backup operator) |
| 55:00 – 58:00 | 14 · Synthesis and Core Contributions | **Tristan (leader)** |
| 58:00 – 60:00 | 15 · Thank You + opens Q&A | **Tristan (leader)** |

### Per-member summary

| Member | Role | Slides | Speaking time |
|---|---|---|---|
| **Tristan Jhay O. Salamat** | Leader, opening, framework, closing, Q&A moderator | 1, 9, 14, 15 | ~10½ min |
| Genesis F. Navarro | The problem | 2, 3, 4 | ~10½ min |
| Charles N. Medio | Objectives, scope and theory | 5, 6, 7, 8 | ~13½ min |
| Mark Jhoshua G. Taberna | System design and algorithm | 10, 11, 12, 13 | ~13 min |
| Alexander D. Castillo | Live demonstration (drives and narrates) | Demo | ~12½ min |

The demo is where time usually runs over. If you're behind, shorten Slides 6 and 8, never the demo. **Tristan keeps time**: sit where you can see a clock, and give the next speaker a small signal if a block runs long.

---

## Slide 1 · Title — ⏱ 0:00–2:30 · *Tristan (leader)*

*Good [morning/afternoon] to our honorable panel members, to our adviser, Ms. Susan Caluya, and to everyone present.*

*I am Tristan Jhay Salamat, the leader of our group. With me are Alexander Castillo, Charles Medio, Genesis Navarro, and Mark Jhoshua Taberna, from the Computer Science Department of National University. Today we present our thesis, "Deepfake Detection for Pornographic Images with Human-in-the-Loop."*

*In one sentence: we built a web-based digital forensic system in which an AI model gives a preliminary assessment of whether a facial image has been manipulated, but a trained forensic analyst always makes the final decision, and the system produces a report that keeps the two clearly separate.*

*Genesis will present the introduction and the problem. Charles will present our objectives, scope, and theoretical framework. I will present the conceptual framework. Mark will present the system architecture, algorithm, and evaluation plan. Alexander will then demonstrate the working prototype live, and I will close with our contributions.*

*To begin, here is Genesis.*

---

## Slide 2 · Chapter 1: Introduction — ⏱ 2:30–5:00 · *Genesis*

*Artificial intelligence has advanced very quickly, and one of its most controversial products is the deepfake. Deepfakes use deep learning models, especially Generative Adversarial Networks or GANs, to create facial images and videos that look real but have been manipulated.*

*With this technology, a person's face can be placed onto content they never took part in. What once needed expert skill can now be done by almost anyone with free tools. This is dangerous when used to create sexually explicit material of real people without their consent. That is the problem our study focuses on.*

---

## Slide 3 · Background of the Study — ⏱ 5:00–9:00 · *Genesis*

*Detecting and classifying deepfake face images has become more complex because generative AI keeps improving. GANs can produce synthetic faces that closely resemble real human identities, so the visible flaws of early deepfakes are disappearing.*

*When this capability is applied to sexually explicit content, it creates a distinct and growing type of online harm: non-consensual synthetic pornography, where a real person's likeness is superimposed onto explicit material they never agreed to appear in. This is supported by the literature we cite as references 3 and 11 in our manuscript.*

*There are two challenges here. The first is **technological**: detectors must catch increasingly subtle manipulations. The second is **forensic**: when this content becomes evidence, investigators need more than a yes-or-no answer from a model. They need to know how the conclusion was reached, who reviewed it, and whether the evidence was handled properly. That second challenge is the gap our system addresses.*

> Optional: add one statistic or local case from your RRL here, with its citation.

---

## Slide 4 · Statement of the Problem — ⏱ 9:00–13:00 · *Genesis*

*Deepfake generation tools are now easy to access, so it is easier than ever to produce convincing, non-consensual sexually explicit images of real people. Because this content is so damaging and its authenticity is often disputed, relying on automated classification alone, whether too little or too much, is risky.*

*Consider the two kinds of errors. An unreviewed **false positive**, where a real image is called fake, may unjustly accuse someone or dismiss a victim's genuine evidence. An unreviewed **false negative**, where a fake image is called real, may let harmful material keep circulating.*

*We also observed three weaknesses in many existing detection systems. First, they rely only on CNN-based feature extraction. Second, they use a single similarity measurement. Third, they give an automated final verdict with no structured human forensic review. These systems can catch obvious manipulations, but they struggle with subtle alterations, often perform poorly on datasets they were not trained on, and, most importantly for forensic use, leave no auditable record of how a human expert weighed the evidence.*

*These problems led to our objectives, which Charles will now present.*

---

## Slide 5 · Objectives of the Study — ⏱ 13:00–17:00 · *Charles*

*Our general objective is to design and build a web-based digital forensic system that detects and classifies suspected deepfake pornographic images using a Hybrid CNN-Transformer model and multi-metric similarity analysis, with a human-in-the-loop review stage in which a trained forensic analyst evaluates the model's output before the final classification and forensic report are issued.*

*We have four specific objectives:*

1. *The **hybrid engine**: build a Hybrid CNN-Transformer model using EfficientNet-B4 and Transformer attention, with MTCNN for facial preprocessing.*
2. *The **multi-metric analysis**: extract facial embeddings and compute cosine similarity, Euclidean distance, and SSIM, so the evidence is interpretable rather than a single hidden score.*
3. *The **human-in-the-loop review interface**: a dashboard where analysts examine the scores, record their rationale, and generate reports.*
4. ***Deployment and evaluation**: a Flask-based prototype, evaluated with accuracy, precision, recall, F1-score, ROC-AUC, and Cohen's kappa for agreement between the AI and the analyst.*

*You will see each of these working in our live demonstration later.*

---

## Slide 6 · Significance of the Study — ⏱ 17:00–19:30 · *Charles*

*Our study benefits three groups.*

*For **forensic investigators and law enforcement**, it provides auditable, human-verified forensic reports. Every case records who submitted it, the AI's analysis, who reviewed it, and why, which is the kind of documentation formal proceedings require.*

*For **victim-support organizations**, it offers transparent evidence that can support takedown requests and help defend victims.*

*For **AI researchers and future developers**, it provides an open architectural blueprint for combining interpretable multi-metric feature fusion with a human-in-the-loop audit pipeline.*

---

## Slide 7 · Scope and Delimitations — ⏱ 19:30–23:00 · *Charles*

*Our **scope**:*
- *The system focuses strictly on **adult subjects, 18 and above**, under institutional ethical protocols. Every submission requires the investigator to attest that the subject is an adult and that they are authorized.*
- *It works on **static facial images**, comparing a suspect image with a paired authentic reference portrait of the same person.*
- *It is an **end-to-end prototype**: preprocessing, feature fusion, human verification, and PDF reporting.*

*Our **delimitations**:*
- ***Material depicting minors is completely excluded.** This is an absolute legal prohibition.*
- ***Audio and full video are outside our scope**. We analyze still images only.*
- *We use a **single-reviewer workflow**: one analyst reviews each case. Multi-rater consensus is left for future deployment.*
- *The AI analyzes the **facial region only**. It is designed to detect face swaps and face manipulation, which is how most pornographic deepfakes are made. Manipulation of the body or background is not analyzed by the AI; the analyst checks it when reviewing the full image.*

> The last bullet is **not on the slide yet**. Add it (see Slide fixes).

---

## Slide 8 · Theoretical Framework — ⏱ 23:00–26:30 · *Charles*

*Our study rests on three theories.*

*First, **Signal Detection Theory**. It gives us the statistical basis for separating a "signal", an authentic face, from "noise", generative artefacts, using a decision threshold. In our system this threshold is τ: if the aggregated similarity score is at least τ, the image is classified as Real; otherwise as Deepfake. SDT is also why we evaluate with ROC-AUC, which measures performance across all possible thresholds.*

*Second, the **GAN framework** of Goodfellow and colleagues, 2014. A generator and a discriminator compete, and that process leaves subtle traces: blending boundaries, texture inconsistencies, lighting mismatches. These are the artefacts our CNN branch is designed to capture.*

*Third, the **Transformer attention mechanism** of Vaswani and colleagues, 2017. Self-attention models long-range relationships across distant parts of the face, for example whether the eyes, nose and jawline are structurally consistent with each other. A CNN, which looks at local regions, can miss these. This is why we pair the CNN with a Transformer.*

*Our leader, Tristan, will now show how these ideas come together in our conceptual framework.*

---

## Slide 9 · Conceptual Framework — ⏱ 26:30–29:30 · *Tristan (leader)*

*Our conceptual framework has four stages: Input, Process, Human Review, and Output.*

*In the **Input** stage, the investigator submits the suspect image together with a verified authentic reference portrait of the same adult subject.*

*In the **Process** stage, MTCNN aligns the faces, then EfficientNet-B4 extracts local features and the Transformer extracts global attention features in parallel. The system then computes three similarity metrics.*

*The **Human Review** stage is what makes our system different. A forensic specialist evaluates the model's confidence and similarity evidence, then confirms, overrides, or flags the case, and must write down their reasoning.*

*In the **Output** stage, the system produces an auditable forensic report that presents the algorithm's metrics and the reviewer's determination **side by side but clearly separated**, so no one can mistake the AI's preliminary result for the human's conclusion.*

*Mark will now walk you through the system architecture.*

---

## Slide 10 · System Architecture — ⏱ 29:30–33:30 · *Mark*

*Here is the architecture, module by module.*

- *The **Preprocessing Module** uses MTCNN to detect the face, align it so the eyes are level, crop it, resize it to 380 by 380 pixels, the native input size of EfficientNet-B4, and normalize it. This is done for both the suspect and the reference image. If no face is detected, the system refuses to analyze the case rather than guess.*
- *The **EfficientNet-B4 backbone** produces F1, a 1,792-value feature vector that captures fine-grained texture anomalies and blending artefacts.*
- *The **Vision Transformer branch** produces F2 from the same face. It captures global relationships across facial landmarks. One technical detail: the Transformer splits the image into 16-by-16 patches, so inside this branch only, the face is resized to 384 by 384, since 380 is not divisible by 16.*
- *In **Feature Fusion and Multi-Metric Scoring**, F1 and F2 are combined into one 512-value embedding. We compare the suspect's embedding with the reference's using cosine similarity and Euclidean distance, and we compare the two aligned face images directly with SSIM.*
- *Finally, **Human Review and Report Generation**: the scores are shown on the analyst's dashboard, the analyst's decision is logged, and the system generates a PDF forensic report.*

---

## Slide 11 · Methodology & Sequential Steps — ⏱ 33:30–36:00 · *Mark*

*From the user's point of view, the process takes six steps.*

*Step 1, the investigator uploads the suspect image and an authentic reference image through the web interface. Step 2, MTCNN preprocesses both faces. Step 3, EfficientNet-B4 and the Vision Transformer extract features in parallel. Step 4, the system computes cosine similarity and Euclidean distance between the embeddings, and SSIM between the aligned face images. Step 5, the case goes into a queue, where a forensic analyst confirms, overrides, or flags the classification with notes. Step 6, the system exports the forensic report combining the AI metrics and the analyst's determination.*

*The next slide shows the formal algorithm.*

---

## Slide 12 · Algorithm Overview — ⏱ 36:00–40:00 · *Mark*

*This is our VERIFY_MEDIA procedure. Its inputs are the suspect image I, the reference R, and the threshold τ.*

- *Lines 2 and 3 are preprocessing: MTCNN detection and alignment, then resizing to 380 by 380 and normalization. The reference image goes through the same steps.*
- *Lines 4 and 5 run the two branches: EfficientNet-B4 gives F1, the Vision Transformer gives F2.*
- *Line 6 fuses them into one embedding.*
- *Lines 7 to 9 compute the three metrics. Cosine similarity measures the angle between the two embeddings. Euclidean distance measures how far apart they are, and we convert it to a similarity, 1 minus d over 2, so that higher always means more similar. SSIM measures the structural similarity of the two aligned face images.*
- *Line 10 combines them into one aggregated score S, a weighted sum with equal weights of one-third by default.*
- *Lines 11 to 15 apply the threshold: if S is at least τ, 0.70 by default, the preliminary classification is Real; otherwise, Deepfake. The system also computes a confidence score based on how far S is from τ.*
- *Line 16 is the human-in-the-loop review: the analyst gives the final decision D and the rationale.*
- *Line 17 generates the forensic report, and line 18 returns the decision, the score, and the report.*

*Lines 2 to 15 run automatically when a case is submitted. Lines 16 and 17 happen only when the analyst acts. The AI's classification C is never the final answer; the analyst's decision D is.*

> **Fix Slide 12 before the defense** so it matches this narration (line 9 SSIM, line 8 conversion). See Slide fixes.

---

## Slide 13 · Datasets, Training & Evaluation — ⏱ 40:00–42:30 · *Mark*

*For training and evaluation we use three benchmark datasets: **FaceForensics++, Celeb-DF, and DFDC**, obtained through their official request forms. Together they cover several deepfake generation methods, which helps the model generalize.*

*For **training**, we start from pre-trained ImageNet weights to save computation, and we apply flipping, rotation, and scaling augmentation to reduce overfitting. The model is trained on Google Colab's GPU, with early stopping based on validation ROC-AUC. We then calibrate the threshold τ on the validation set.*

*Our **evaluation has two parts**. For the machine classification, we measure accuracy, precision, recall, F1-score, and ROC-AUC, and compare our hybrid model with an XceptionNet baseline. For human–AI agreement, we measure the agreement rate and Cohen's kappa between the AI's preliminary classification and the analyst's final decision.*

*Alexander will now demonstrate the working system.*

---

## LIVE DEMONSTRATION — ⏱ 42:30–55:00 · *Alexander* (Mark = backup operator)

> Use **only** the NASA demo images in `demo\samples\` (public-domain, non-explicit).
> Before the defense, follow the **checklist** at the end. Start with fresh data so the case is CASE-0001.

### D1 · Introduce the prototype (42:30–43:30)
*Alexander:* *Our prototype runs locally on this laptop, on the CPU. It is built with Python, PyTorch, Flask, and ReportLab for the PDF reports.*

*(If the red banner shows:)* *The red banner reads "UNTRAINED MODEL: results not indicative of real accuracy." Our system displays this whenever the fine-tuned weights have not been loaded, on every page and in every report, so that no one relies on an unvalidated AI result.*

*Alexander:* *For this demonstration we use a public-domain NASA portrait, not explicit material. Our AI analyzes only the face region, so the process is identical, and our ethics protocol forbids showing explicit material outside authorized casework.*

### D2 · Investigator submits a case (43:30–46:30)
**Action:** on the login page, **Log in as: Investigator** → **INV-01** → **Submit case** → Suspect `suspect_manipulated.png`, Reference `reference.png`.

*Alexander:* *On the login page I choose "Log in as Investigator". The system checks that the chosen role matches the account, so an analyst account cannot enter as an investigator. We are now logged in as an investigator. The suspect image has a synthetic edit to the inner face region; the reference is the original portrait of the same person.*

**Action:** points at the checkbox, ticks it.

*Alexander:* *Every submission requires this attestation: that the subject is an adult and that the investigator is authorized. This enforces our scope.*

**Action:** clicks **Submit for analysis**.

*Alexander:* *Right now the system computes a SHA-256 hash of each file for chain of custody, then runs our algorithm, lines 2 to 15: face detection, both neural branches, fusion, the three metrics, and the preliminary classification.*

**Action:** shows **My cases**.

*Alexander:* *The case is **Pending**. Notice that the investigator does not see the AI's result. Nothing is final until an analyst reviews it.*

### D3 · No face = no analysis (46:30–47:30)
**Action:** submits `no_face.png` + `reference.png`.

*Alexander:* *If no face is detected, the system does not guess: "No face detected. Case cannot be analyzed." The rejection is still recorded in the audit log.*

### D4 · Analyst dashboard (47:30–48:30)
**Action:** logs out → **Log in as: Analyst** → **ANA-01**.

*Alexander:* *Now we are the forensic analyst. The dashboard shows how many cases are pending, verified, and flagged, and the live agreement rate between the AI and the analysts.*

### D5 · Review interface, objective 3 (48:30–52:30)
**Action:** clicks **Review**.

*Alexander:* *This is the core of our study.*
- *On the left are only the **aligned face crops**. The full images are **pixelated** by default. This protects sensitive material.*
- *On the right, in blue and labeled **AUTOMATED**, is the AI's preliminary analysis: the classification, the confidence, the three similarity metrics, and the aggregated score S against τ.* *(Read the actual values on screen.)*

**Action:** clicks **Reveal full image** → confirms.

*Alexander:* *If the analyst must see the full image, they reveal it deliberately. Only analysts can do this, and every reveal is logged with their ID and the time.*

**Action:** in the green **HUMAN-VERIFIED** section, chooses:
- AI said **Real** → **Override** · AI said **Deepfake** → **Confirm**.

**Action:** types "fake" and clicks submit.

*Alexander:* *A rationale is mandatory, at least 30 characters. The analyst must justify every decision.*

**Action:** types the rationale (Override example):
> Although the AI scored the image as Real, the inner face region shows warping around the eyes and nose bridge, and its colour saturation differs from the forehead and neck. These localized distortions are inconsistent with the reference image. Overriding to Deepfake.

…and submits.

*Alexander (if Override):* *This is exactly why human review matters. The AI's preliminary result was wrong, and the analyst corrected it, with the reasoning on record. This is line 16 of our algorithm.*

### D6 · Forensic report (52:30–54:00)
*Alexander:* *This is line 17, the forensic report.* (Scroll slowly.)
- *Section I, case information: who submitted and who reviewed.*
- *Section II: only **blurred** face crops, with the SHA-256 hashes.*
- *Section III, in blue: the **AUTOMATED** AI analysis, labeled as preliminary and not a verdict.*
- *Section IV, in green: the **HUMAN-VERIFIED** conclusion: decision, final classification, and rationale.*
- *Section V: the analyst's sign-off.*

**Action:** clicks **Download PDF**, shows it.

*Alexander:* *The case is now read-only; the review can no longer be changed.*

### D7 · Admin: evaluation and audit log, objective 4 (54:00–55:00)
**Action:** logs out → **Log in as: Administrator** → **ADM-01** → **Evaluation** → **Audit log**.

*Alexander:* *The administrator's evaluation page computes the agreement rate and Cohen's kappa from the case records. And this is the append-only audit log: logins, submissions, rejections, image reveals, reviews, and report downloads, each with the user and time. That concludes our demonstration. I now give the floor back to our leader, Tristan.*

> **Backup:** if anything fails, **Mark** takes the laptop and opens `docs\walkthrough\` (screenshots 01 → 13 plus the sample PDF) while Alexander keeps narrating. Don't troubleshoot live for more than 30 seconds.
> **Tip:** Alexander both clicks and talks. Say each line *before* clicking, then pause while the page loads.

---

## Slide 14 · Synthesis and Core Contributions — ⏱ 55:00–58:00 · *Tristan (leader)*

*To synthesize, our study makes three contributions.*

*First, **domain specialization**: a digital forensic pipeline designed specifically to detect and document non-consensual explicit deepfake imagery, with ethics built into the software: the adults-only attestation, redacted views, logged reveals, and an audit trail.*

*Second, **hybrid multi-metric robustness**: we combine local CNN texture features, global Transformer attention, and three similarity metrics, instead of relying on a single model or a single score.*

*Third, **defensible reporting**: every case requires a human review with a written rationale, and the report keeps the AI's automated analysis separate from the analyst's verified conclusion, a structured, auditable record designed to support forensic and legal review.*

---

## Slide 15 · Thank You — ⏱ 58:00–60:00 · *Tristan (leader)*

*In summary: deepfakes, especially non-consensual pornographic deepfakes, cause serious harm, and automated detection alone is not enough when the result may be used as evidence. Our system lets AI do what it does well, analyzing facial features at scale, while a trained human analyst makes the final, documented decision.*

*On behalf of our group, thank you very much for your time and attention. We are now open to the panel's questions.*

---

## Slide fixes before the defense

These are places where the **slides don't match the system** or **claim more than a prototype can**. A panel comparing your slides to the demo could ask about them.

### Must fix (technical mismatches)

| Slide | Current text | Problem | Suggested change |
|---|---|---|---|
| 12, line 9 | `Score_SSIM ← Calculate_SSIM_Index(F_fused, R)` | SSIM is a 2-D image measure and cannot be computed on an embedding vector. The system computes it on the aligned face crops. | `Score_SSIM ← Calculate_SSIM_Index(I_aligned, R_aligned)  // structural similarity of aligned face crops` |
| 12, line 8 | `Score_Euc ← Compute_Euclidean_Distance(F_fused, R)` | A *distance* grows as images differ, so adding it to S would push fakes toward "Real". The system converts it. | `Score_Euc ← 1 − EuclideanDistance(F_fused, R_fused) / 2  // distance converted to similarity` |
| 12, header | `Reference Tensor (R), Forensic Threshold (T)` | The procedure uses τ, not T; R is the reference *image*, processed by the same lines 2–6. | `// Input: Suspect Image (I), Reference Image (R), Threshold (τ)` and add a note: *lines 2–6 are also applied to R* |
| 12, line 6 | `Concatenate_Vectors(F_1, F_2)` | The system also projects to 512 values (Linear → LayerNorm) and L2-normalizes. | Optional: `F_fused ← L2Norm(Project(Concatenate(F_1, F_2)))` |
| 11, step 4 | "…SSIM between suspect and reference embeddings" | Same SSIM issue. | "…Cosine Similarity and Euclidean Distance between the embeddings, and SSIM between the aligned face images" |
| 7 | (missing) | The AI analyzes the face only; the panel may ask about body-only edits. | Add a delimitation: **"Facial Region Only: the AI analyzes the facial region; body or background manipulation is assessed by the analyst."** |

The confidence score is also computed next to lines 11–15. You can add it as a line or just mention it verbally, as in the script.

### Should soften (overclaims)

The model isn't trained or validated yet, and it's a prototype. Words like *certified*, *binding*, *guarantee* and *ready for judicial scrutiny* invite a hard question ("has a court accepted it?").

| Slide | Current | Safer |
|---|---|---|
| 5 | "generate certified reports" | "generate structured forensic reports" |
| 6 | "capable of withstanding scrutiny in formal legal proceedings" | "designed to support scrutiny in formal proceedings" |
| 9 | "reviewer's certified determination" | "reviewer's verified determination" |
| 11 | "analyst's binding determination" | "analyst's final determination" |
| 13 | "to guarantee robust generalization" | "to improve generalization" |
| 14 | "reports ready for judicial scrutiny" | "reports designed to support forensic and legal review" |

### Consider adding
- A **"System Demonstration"** divider slide between 13 and 14, so the switch to the laptop feels planned.
- If this is your **final** defense (not the proposal): a **Results** slide (comparison table, confusion matrix, ROC curve from `evaluation/results/`) and a **Limitations / Recommendations** slide. The panel will expect Chapter 4 results. Only show results from the trained model.

---

## Pre-defense checklist

**The day before**
- [ ] In VS Code: `git pull`, `.venv\Scripts\activate`, `python -m pytest` → **96 passed**.
- [ ] Run one full case so the ImageNet weights are downloaded and cached.
- [ ] Fresh demo data: stop the app, **rename** the `data` folder to `data_practice`, then run `python seed_users.py --password <demo password>` and `python demo/make_demo_samples.py`.
- [ ] Rehearse the full 60 minutes once and the demo twice (target: 12½ minutes).
- [ ] Charge the laptop, and bring an HDMI adapter and a phone hotspot (the page styling loads from the internet).
- [ ] Print or save the manuscript PDF on the laptop, for checking page numbers during Q&A.
- [ ] Genesis brings a notebook and pen for recording the panel's comments.
- [ ] Confirm the panelists' names and titles for the call to order.

**30 minutes before**
- [ ] `.venv\Scripts\activate` → `flask --app app run` → Chrome at http://127.0.0.1:5000.
- [ ] Open `demo\samples\` in File Explorer; keep `docs\walkthrough\` ready as the backup.
- [ ] Zoom the browser to about 125%. Close unrelated apps and turn off notifications.
- [ ] Put phones on silent. Have water ready for the speakers.
- [ ] Deck open on Slide 1, and the projector tested, **before** the call to order.

---

## Question and answer — 45 minutes (led by Tristan)

### Roles during Q&A

| Member | Role |
|---|---|
| **Tristan** | **Moderator.** Receives each question, repeats it briefly if unclear, and passes it to the right member. Answers overall-design and scope questions. Watches the time. |
| **Genesis** | **Scribe.** Writes down every question, comment and suggested revision, with the panelist's name. These notes become your revision list. Still answers their own topics. |
| **Alexander** | **System operator.** Keeps the app open and logged in, ready if a panelist says "show me…". |
| **Charles** and **Mark** | Answer their topics; keep the slide deck ready to jump to Slides 7, 10, 12 or 13. |

| Question topic | Answered by |
|---|---|
| Background, problem, motivation, statistics, related studies | Genesis |
| Objectives, significance, scope and delimitations, theories | Charles |
| Overall design, human-in-the-loop, ethics, conceptual framework, future work | Tristan |
| Architecture, algorithm, formulas, datasets, training, evaluation metrics | Mark |
| How the prototype works, code, security features, the "UNTRAINED MODEL" banner | Alexander |

### How to answer

1. **Listen to the whole question.** Don't start answering while the panelist is still talking.
2. **Start with thanks or agreement**: *"Thank you for the question, sir/ma'am."*
3. **Answer in 30–90 seconds.** State the answer first, then one reason or example. Point to the chapter or slide if it helps: *"As shown in our algorithm, line 9…"*
4. **Don't argue or overclaim.** If a panelist suggests a change, accept it: *"Thank you, we will include that in our revisions."* Genesis writes it down.
5. **If you don't know,** don't guess: *"That is a valid point. We have not tested that yet; we will verify it and include it in our revisions."*
6. **Only one member answers each question.** Others add one point only if Tristan invites them.
7. **If a panelist asks to see the system,** Tristan says *"Alexander will show it,"* and Alexander logs in as the needed role (INV-01 / ANA-01 / ADM-01).

**Pacing:** 45 minutes is usually 12–20 questions. If one topic drags on, Tristan can say: *"If we may, we will expand on that in our revised manuscript."*

**Closing the Q&A (Tristan), when the chair ends it:**
*Thank you very much to our panel for your questions and suggestions. We have noted all of them and will incorporate them in our revisions.*

### Question bank

**Genesis: background and problem**

| Question | Suggested answer |
|---|---|
| Why focus on pornographic deepfakes? | It is among the most harmful uses of deepfakes: victims suffer reputational, psychological and legal harm, and the content is often used for harassment or extortion. It also needs careful forensic handling. |
| What makes your study different from existing detectors? | Most detectors give an automated final verdict. Ours combines a hybrid CNN–Transformer, three similarity metrics, and a mandatory human review, and the report keeps the AI analysis separate from the human decision. |
| What is the harm of a false positive / false negative? | A false positive can wrongly accuse someone or dismiss a victim's genuine evidence; a false negative lets harmful content keep circulating. That is why an analyst reviews every case. |

**Charles: objectives, scope and theory**

| Question | Suggested answer |
|---|---|
| Why only adults? | Material depicting minors is under absolute legal prohibition and cannot be handled by a student prototype. The system enforces this with a required attestation on every submission. |
| Why still images and not video? | Video needs temporal analysis across frames, which is outside our scope; video is in our recommendations. |
| What if only the body is edited? | The AI analyzes the facial region only, so a body-only edit is outside its scope. The analyst reviews the full image and can override or flag the case. |
| Why a single reviewer? | It's a delimitation for the prototype; multi-rater consensus is planned for later deployment. |
| How is Signal Detection Theory applied? | Our threshold τ separates "authentic" from "manipulated" scores, and ROC-AUC measures performance across all thresholds. |

**Tristan: design, human-in-the-loop and ethics**

| Question | Suggested answer |
|---|---|
| Why not let the AI decide? | Forensic results can have legal consequences and the AI can be wrong, as shown in our demo. The analyst provides accountability through a documented rationale. |
| What if the analyst is biased or wrong? | Every decision needs a written rationale, every action is in the audit log, and the AI analysis stays in the report, so reviewers can be audited. Multi-reviewer consensus is future work. |
| What happens to flagged cases? | They are marked "Inconclusive" for further investigation, still get a report, and are counted separately in the evaluation. |
| How do you protect the victims' privacy? | Face crops only by default, full images blurred or pixelated, logged reveals, role-based access, uploads outside public folders, and a secure purge for data retention. |
| Why is the demo not using explicit images? | The AI analyzes only the face crop, so the process is identical; our ethics protocol forbids explicit material outside authorized casework. |

**Mark: architecture, algorithm, data and metrics**

| Question | Suggested answer |
|---|---|
| Why EfficientNet-B4 and not another CNN? | It is accurate for its size, and its native 380×380 input keeps fine facial detail where blending artefacts appear. |
| Why add a Transformer? | Self-attention captures relationships across the whole face (structural consistency) that a local CNN can miss. |
| Why resize to 384 in the ViT branch? | Patch-16 Transformers need an input size divisible by 16; 380 isn't. |
| Why SSIM on face crops, not embeddings? | SSIM compares 2-D image structure (luminance, contrast, structure); it cannot be computed on a 1-D embedding. |
| Why convert the Euclidean distance? | A distance increases as images differ; converting it to 1 − d/2 makes all three metrics "higher = more similar" before averaging. |
| How was τ = 0.70 chosen? | It is the default; it is calibrated on the validation set by maximizing F1 or Youden's J. |
| Why equal weights (⅓ each)? | Neutral default with no metric favored; the weights are configurable and can be tuned on validation data. |
| Your datasets aren't pornographic. Does it generalize? | Face-swap artefacts appear in the face region regardless of the rest of the image, but we acknowledge a domain gap as a limitation. |
| Why compare with XceptionNet? | It is the standard FaceForensics++ baseline. |
| What is Cohen's kappa? | Agreement between the AI and the analyst beyond chance: κ = (pₒ − pₑ)/(1 − pₑ). |
| What are your results? | *(If trained)* cite your Chapter 4 numbers. *(If not)* The pipeline is complete; results will follow training on the requested datasets. Never quote results from the untrained model. |

**Alexander: the prototype**

| Question | Suggested answer |
|---|---|
| Why does it say "UNTRAINED MODEL"? | Fine-tuned weights aren't loaded yet; the system states this on every page and report so no one relies on the AI (a stated limitation). |
| How fast is it? | A few seconds per case on a normal laptop CPU; no GPU needed for the demo. Training is done on a Colab GPU. |
| How do you ensure chain of custody? | A SHA-256 hash of each uploaded file is stored in the case record and shown in the report. |
| Can the analyst change a decision later? | No. Reviewed cases are read-only. |
| Who can delete data? | Only an admin, through a secure purge command that overwrites the files and logs the deletion. |
| How was it tested? | 96 automated tests, covering the formulas, classifier, kappa, review workflow, access control, and a full submission-to-PDF run. |
| What happens with an image without a face? | The system refuses to analyze it and records the rejection; it never guesses. |

---

## Panel deliberation — 10 minutes

- When the chair asks, **the group leaves the room** (or waits quietly, as instructed). Leave the laptop and slides as they are unless told otherwise.
- **Genesis** reads the notes to the group; everyone checks whether any comment was missed. Don't discuss loudly near the room.
- Prepare questions to ask the panel afterwards, e.g. the deadline for revisions and who will check them.

### When called back: after the verdict

Listen to the verdict and the required revisions. Genesis keeps writing. Then **Tristan** says:

*On behalf of our group, thank you very much to our panel for your time, your guidance, and your valuable suggestions. We accept the recommended revisions and will incorporate them in our manuscript and system. We also thank our adviser, Ms. Susan Caluya, for guiding us throughout this study. Thank you.*

If anything about a revision is unclear, Tristan politely asks before leaving: *"May we clarify, sir/ma'am, regarding…"*
