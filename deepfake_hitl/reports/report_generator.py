"""Forensic Report Generation (Section 3.3, Appendix B template).

One data builder (build_report_data) feeds both the HTML view
(templates/report.html) and the PDF (generate_pdf), so the two always agree.

    I.   Case Information
    II.  Submitted Images        - blurred face crops only + SHA-256 hashes
    III. AI Analysis             - AUTOMATED       (blue header)
    IV.  Human Review            - HUMAN-VERIFIED  (green header)
    V.   Sign-off
"""
import io
import os

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

import config
import storage

MAX_THUMBNAILS_SUSPECT = 12       # larger cases: thumbnails for the most suspicious only
UNTRAINED_BANNER = "UNTRAINED MODEL: results not indicative of real accuracy"
AI_DISCLAIMER = ("The automated analysis is a preliminary, AI-assisted indication only. "
                 "It is not a verdict. The analyst's human-verified conclusion in Section IV "
                 "is the finding of this report.")
AUTOMATED_COLOR = colors.HexColor("#1f4e79")     # blue  - Section III
HUMAN_COLOR = colors.HexColor("#1e6b3a")         # green - Section IV


def _pct(x):
    return "—" if x is None else f"{x:.4f}"


def build_report_data(case, users=None):
    """Plain dict used by both the HTML template and the PDF."""
    users = users or {}

    def name(uid):
        u = users.get(uid) if uid else None
        return f"{u['name']} ({uid})" if u else (uid or "—")

    suspects, references = storage.case_images(case, "suspect"), storage.case_images(case, "reference")
    multi = len(suspects) > 1 or len(references) > 1
    ranked = storage.ranked_suspects(case)                      # most suspicious first
    thumb_suspects = {x["n"] for x in ranked[:MAX_THUMBNAILS_SUSPECT]}
    images = []
    for which, items in (("suspect", suspects), ("reference", references)):
        for item in items:
            label = f"{which.capitalize()} image" + (f" #{item['n']}" if multi else "")
            images.append({"which": which, "n": item["n"], "label": label,
                           "filename": item.get("filename") or os.path.basename(item.get("path") or ""),
                           "sha256": item.get("sha256") or "—",
                           "thumb": which == "reference" or item["n"] in thumb_suspects})
    per_image = []
    for item in ranked:
        r = storage.suspect_result(case, item["n"])
        per_image.append({"n": item["n"], "ai": r["ai_classification"], "s": _pct(r["aggregated_score"]),
                          "ref": r.get("best_reference", 1), "conf": _pct(r["confidence_score"]),
                          "filename": item.get("filename") or ""})
    deciding = storage.suspect_result(case, case.get("deciding_suspect") or 1)

    return {
        "case_id": case["case_id"],
        "untrained": not case.get("model_trained", False),
        "untrained_banner": UNTRAINED_BANNER,
        "disclaimer": AI_DISCLAIMER,
        "case_information": [
            ("Case ID", case["case_id"]),
            ("Date Submitted", case["submitted_at"]),
            ("Submitted By (Investigator)", name(case["submitted_by"])),
            ("Reviewed By (Analyst)", name(case.get("reviewer_id"))),
        ],
        "images": images,
        "thumbnails": [i for i in images if i["thumb"]],
        "all_thumbnails": all(i["thumb"] for i in images),
        "n_deepfake": sum(1 for r in per_image if r["ai"] == "Deepfake"),
        "skipped": case.get("skipped_files") or [],
        "multi": multi,
        "per_image": per_image,
        "ai_analysis": [
            ("Preliminary classification", case["ai_classification"]),
        ] + ([
            ("Images analysed", f"{len(suspects)} suspect · {len(references)} reference"),
            ("Suspect images below τ", f"{sum(1 for x in ranked if x['ai'] == 'Deepfake')} of {len(suspects)}"),
            ("Scores below are for", f"Suspect #{deciding['n']} (most suspicious, lowest S) "
                                     f"vs. Reference #{deciding['best_reference']}"),
        ] if multi else []) + [

            ("Confidence score", _pct(case["confidence_score"])),
            ("Cosine similarity", _pct(case["cosine_similarity"])),
            ("Euclidean similarity", _pct(case["euclidean_similarity"])),
            ("SSIM (aligned face crops)", _pct(case["ssim"])),
            ("Aggregated score S", _pct(case["aggregated_score"])),
            ("Threshold τ", _pct(case["threshold"])),
            ("Decision rule", "Real if S ≥ τ, otherwise Deepfake"),
            ("Model version", case.get("model_version") or "—"),
            ("Model status", "Fine-tuned" if case.get("model_trained") else "UNTRAINED (ImageNet / random init)"),
        ],
        "human_review": [
            ("Review decision", case.get("review_decision") or "Pending"),
            ("Final classification", case.get("final_classification") or "Pending"),
            ("Rationale", case.get("rationale") or "—"),
        ],
        "sign_off": [
            ("Analyst", name(case.get("reviewer_id"))),
            ("Date reviewed", case.get("reviewed_at") or "—"),
        ],
    }


def _blurred_face(case_id, which, upload_dir, n=1):
    p = storage.face_path(case_id, which, upload_dir, n)
    if not os.path.exists(p):
        return None
    with Image.open(p) as im:
        return storage.redact(im, "blur")


def _kv_table(rows, width, header=None, header_color=None, label_bg="#f2f2f2"):
    body = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=10.5)
    label = ParagraphStyle("label", parent=body, fontName="Helvetica-Bold")
    data, style = [], [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        hstyle = ParagraphStyle("h", parent=label, textColor=colors.white, fontSize=10)
        data.append([Paragraph(header, hstyle), ""])
        style += [("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (1, 0), header_color)]
    start = len(data)
    for k, v in rows:
        data.append([Paragraph(_escape(str(k)), label), Paragraph(_escape(str(v)), body)])
    style.append(("BACKGROUND", (0, start), (0, -1), colors.HexColor(label_bg)))
    t = Table(data, colWidths=[width * 0.35, width * 0.65])
    t.setStyle(TableStyle(style))
    return t


def _escape(s):
    # Built-in PDF fonts (Helvetica) have no glyphs for these symbols.
    s = s.replace("τ", "tau").replace("≥", ">=")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")


def generate_pdf(case, out_path, users=None, upload_dir=None):
    """Write the Appendix B forensic report PDF to out_path and return the path."""
    data = build_report_data(case, users)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title=f"Forensic Report {case['case_id']}")
    width = A4[0] - 36 * mm
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("t", parent=ss["Title"], fontSize=14, spaceAfter=6)
    h2 = ParagraphStyle("s", parent=ss["Heading2"], fontSize=11, spaceBefore=5, spaceAfter=3)
    small = ParagraphStyle("sm", parent=ss["Normal"], fontSize=8, leading=10)
    banner = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=10,
                            textColor=colors.white, backColor=colors.HexColor("#b00020"),
                            alignment=TA_CENTER, borderPadding=5, spaceAfter=8)
    story = [Paragraph("Deepfake Detection with Human-in-the-Loop — Forensic Report", h1)]
    if data["untrained"]:
        story += [Paragraph(data["untrained_banner"], banner), Spacer(1, 4)]

    story += [Paragraph("I. Case Information", h2), _kv_table(data["case_information"], width)]

    story.append(Paragraph("II. Submitted Images (redacted: blurred face crops only)", h2))
    cells = []
    if not data["all_thumbnails"]:
        story.append(Paragraph(
            f"Showing the {len(data['thumbnails'])} most relevant of {len(data['images'])} images "
            "(every reference image and the most suspicious suspect images). "
            "Every image's file name and SHA-256 is listed in Annex A.", small))
    for img in data["thumbnails"]:
        face = _blurred_face(case["case_id"], img["which"], upload_dir, img["n"])
        cell = [Paragraph(f"<b>{img['label']}</b>", small)]
        if face is not None:
            cell.append(RLImage(io.BytesIO(storage.image_png_bytes(face, 300)), 28 * mm, 28 * mm))
        cell.append(Paragraph(f"SHA-256:<br/><font face='Courier' size='6'>{img['sha256']}</font>", small))
        cells.append(cell)
    per_row = 2 if len(cells) <= 2 else 3
    rows = [cells[i:i + per_row] for i in range(0, len(cells), per_row)]
    rows[-1] += [""] * (per_row - len(rows[-1]))
    t = Table(rows, colWidths=[width / per_row] * per_row)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(t)

    story.append(Paragraph("III. AI Analysis", h2))
    story.append(_kv_table(data["ai_analysis"], width,
                           header="AUTOMATED — AI-assisted preliminary analysis (not a verdict)",
                           header_color=AUTOMATED_COLOR, label_bg="#e8f0f8"))
    if data["multi"]:
        cell = ParagraphStyle("pi", fontName="Helvetica", fontSize=8, leading=10)
        head = ParagraphStyle("pih", parent=cell, fontName="Helvetica-Bold")
        rows = [[Paragraph(h, head) for h in ("Suspect image", "File", "AI result", "Score S",
                                               "Best ref.", "Confidence")]]
        rows += [[Paragraph(f"#{r['n']}", cell), Paragraph(_escape(r["filename"]), cell), Paragraph(r["ai"], cell),
                  Paragraph(r["s"], cell), Paragraph(f"#{r['ref']}", cell), Paragraph(r["conf"], cell)]
                 for r in data["per_image"]]
        pt = Table(rows, colWidths=[w * width for w in (.12, .34, .14, .14, .12, .14)], repeatRows=1)
        pt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f8"))]))
        story += [Spacer(1, 4), Paragraph("<b>Per-image results (automated, most suspicious first)</b>", small), pt]
    story.append(Paragraph(data["disclaimer"], small))

    story.append(Paragraph("IV. Human Review", h2))
    story.append(_kv_table(data["human_review"], width,
                           header="HUMAN-VERIFIED — Forensic analyst conclusion",
                           header_color=HUMAN_COLOR, label_bg="#e8f5ec"))

    story += [Paragraph("V. Sign-off", h2), _kv_table(data["sign_off"], width),
              Spacer(1, 10), Paragraph("Signature: ________________________________", ss["Normal"])]

    if data["skipped"] or not data["all_thumbnails"]:
        mono = ParagraphStyle("mono", fontName="Courier", fontSize=6, leading=7.5)
        cell = ParagraphStyle("ac", fontName="Helvetica", fontSize=7.5, leading=9)
        head = ParagraphStyle("ach", parent=cell, fontName="Helvetica-Bold")
        grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")])
        if not data["all_thumbnails"]:
            rows = [[Paragraph(h, head) for h in ("Image", "File", "SHA-256")]]
            rows += [[Paragraph(i["label"], cell), Paragraph(_escape(i["filename"]), cell),
                      Paragraph(i["sha256"], mono)] for i in data["images"]]
            at = Table(rows, colWidths=[width * .18, width * .3, width * .52], repeatRows=1)
            at.setStyle(grid)
            story += [Paragraph("Annex A. All submitted images (chain of custody)", h2), at]
        if data["skipped"]:
            rows = [[Paragraph(h, head) for h in ("Type", "File", "Reason", "SHA-256")]]
            rows += [[Paragraph(x["which"].capitalize(), cell), Paragraph(_escape(x["filename"]), cell),
                      Paragraph(_escape(x["reason"]), cell), Paragraph(x["sha256"], mono)] for x in data["skipped"]]
            sk = Table(rows, colWidths=[width * .12, width * .26, width * .2, width * .42], repeatRows=1)
            sk.setStyle(grid)
            story += [Paragraph("Annex B. Uploaded files not analysed", h2),
                      Paragraph("These files were not part of the case (no face found, or not a valid "
                                "JPG/PNG image) and were erased; their hashes are recorded.", small), sk]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(18 * mm, 8 * mm, f"{case['case_id']} · CONFIDENTIAL — authorized personnel only")
        canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path


def report_path_for(case_id, report_dir=None):
    return os.path.join(report_dir or config.REPORT_DIR, f"{case_id}_report.pdf")
