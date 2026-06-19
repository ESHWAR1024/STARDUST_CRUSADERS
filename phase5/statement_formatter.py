"""
Phase 5 - Statement formatter / multi-format exporter.

Two distinct outputs are produced from the same master ledger:

1. The MASTER CLEAN DATASET (`master_transactions_clean.csv`) - one
   unified schema across the whole population, fraud columns stripped.

2. Per-account "AS RECEIVED" statement exports in four formats:
   CSV, Excel (xlsx), scanned PNG, and PDF - each bank using its own
   column names, date formats, and layout to stress-test Phase 6 ingestion.
"""

import os
import random
import textwrap
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# ---------------------------------------------------------------------------
# Per-bank "as received" layouts - column names, order, and date format
# deliberately differ to mimic real heterogeneous statement exports.
# ---------------------------------------------------------------------------
BANK_FORMATS = {
    "SBI": {
        "columns": ["Txn Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"],
        "date_fmt": "%d/%m/%y",
        "map": {"date": "Txn Date", "narration": "Description", "ref": "Ref No./Cheque No.",
                "debit": "Debit", "credit": "Credit", "balance": "Balance"},
    },
    "HDFC": {
        "columns": ["Date", "Narration", "Chq./Ref.No.", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"],
        "date_fmt": "%d/%m/%Y",
        "map": {"date": "Date", "narration": "Narration", "ref": "Chq./Ref.No.",
                "debit": "Withdrawal Amt.", "credit": "Deposit Amt.", "balance": "Closing Balance"},
    },
    "ICICI": {
        "columns": ["Transaction Date", "Transaction Remarks", "Cheque Number",
                    "Withdrawal Amount (INR)", "Deposit Amount (INR)", "Balance (INR)"],
        "date_fmt": "%d-%b-%Y",
        "map": {"date": "Transaction Date", "narration": "Transaction Remarks", "ref": "Cheque Number",
                "debit": "Withdrawal Amount (INR)", "credit": "Deposit Amount (INR)", "balance": "Balance (INR)"},
    },
    "AXIS": {
        "columns": ["Tran Date", "Particulars", "Cheque No", "Debit", "Credit", "Balance"],
        "date_fmt": "%Y-%m-%d",
        "map": {"date": "Tran Date", "narration": "Particulars", "ref": "Cheque No",
                "debit": "Debit", "credit": "Credit", "balance": "Balance"},
    },
    "CANARA": {
        "columns": ["Date", "Particulars", "Instrument Id", "Withdrawals", "Deposits", "Balance"],
        "date_fmt": "%d-%m-%Y",
        "map": {"date": "Date", "narration": "Particulars", "ref": "Instrument Id",
                "debit": "Withdrawals", "credit": "Deposits", "balance": "Balance"},
    },
    "PNB": {
        "columns": ["Post Date", "Remarks", "Cheque No/Ref No", "Debit", "Credit", "Balance"],
        "date_fmt": "%d.%m.%Y",
        "map": {"date": "Post Date", "narration": "Remarks", "ref": "Cheque No/Ref No",
                "debit": "Debit", "credit": "Credit", "balance": "Balance"},
    },
}


def export_master_clean(rows, out_path):
    df = pd.DataFrame(rows)
    clean = df[["transaction_id", "account_id", "account_holder", "bank_name",
                "date", "time", "narration", "channel", "debit", "credit",
                "balance", "counterparty_account_id", "counterparty_name", "utr_ref"]]
    clean.to_csv(out_path, index=False)
    return clean


def _format_account_df(account_rows, bank_code):
    fmt = BANK_FORMATS[bank_code]
    out_rows = []
    for r in account_rows:
        d = pd.to_datetime(r["date"]).strftime(fmt["date_fmt"])
        row = {
            fmt["map"]["date"]: d,
            fmt["map"]["narration"]: r["narration"],
            fmt["map"]["ref"]: r["utr_ref"],
            fmt["map"]["debit"]: r["debit"] if r["debit"] else "",
            fmt["map"]["credit"]: r["credit"] if r["credit"] else "",
            fmt["map"]["balance"]: r["balance"],
        }
        out_rows.append(row)
    return pd.DataFrame(out_rows, columns=fmt["columns"])


def export_sample_bank_statements(rows_df, accounts, sample_account_ids, out_dir,
                                   format_assignment=None):
    """
    For a small sample of accounts, export an "as received" statement in
    the account's bank's native column layout, in a randomly-assigned
    file format (csv / xlsx / scanned png) to exercise multi-format
    ingestion. Returns a manifest list describing what was written.
    """
    os.makedirs(out_dir, exist_ok=True)
    acct_index = {a.account_id: a for a in accounts}
    manifest = []
    formats_cycle = ["csv", "xlsx", "pdf", "scanned_png", "csv", "xlsx", "pdf", "scanned_png"]

    for idx, account_id in enumerate(sample_account_ids):
        acct = acct_index[account_id]
        bank_code = acct.bank_code if acct.bank_code in BANK_FORMATS else "SBI"
        sub = rows_df[rows_df["account_id"] == account_id].to_dict("records")
        if not sub:
            continue
        statement_df = _format_account_df(sub, bank_code)

        fmt = format_assignment[idx] if format_assignment else formats_cycle[idx % len(formats_cycle)]
        fname_base = f"statement_{account_id}_{bank_code}"

        if fmt == "csv":
            path = os.path.join(out_dir, fname_base + ".csv")
            with open(path, "w") as f:
                f.write(f"{acct.bank_name} - Account Statement\n")
                f.write(f"Account Holder: {acct.holder_name}\n")
                f.write(f"Account Number: {acct.account_number}    IFSC: {acct.ifsc}\n\n")
            statement_df.to_csv(path, mode="a", index=False)

        elif fmt == "xlsx":
            path = os.path.join(out_dir, fname_base + ".xlsx")
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                statement_df.to_excel(writer, index=False, startrow=4, sheet_name="Statement")
                ws = writer.sheets["Statement"]
                ws["A1"] = f"{acct.bank_name} - Account Statement"
                ws["A2"] = f"Account Holder: {acct.holder_name}"
                ws["A3"] = f"Account Number: {acct.account_number}    IFSC: {acct.ifsc}"

        elif fmt == "pdf":
            path = os.path.join(out_dir, fname_base + ".pdf")
            render_pdf_statement(acct, statement_df, path)

        elif fmt == "scanned_png":
            path = os.path.join(out_dir, fname_base + "_scanned.png")
            render_scanned_statement_image(acct, statement_df, path)

        manifest.append({"account_id": account_id, "bank_code": bank_code,
                          "format": fmt, "file": os.path.basename(path), "n_rows": len(sub)})
    return manifest


def render_pdf_statement(acct, statement_df, out_path, max_rows=50):
    """
    Render a clean, formatted bank statement PDF using ReportLab.
    Mimics the look of an official bank-generated PDF statement with
    header info, a ruled transaction table, and page numbers.
    """
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('BankTitle', fontSize=14, fontName='Helvetica-Bold',
                                  alignment=TA_CENTER, spaceAfter=4)
    header_style = ParagraphStyle('Header', fontSize=9, fontName='Helvetica',
                                   alignment=TA_LEFT, spaceAfter=2)
    small_style = ParagraphStyle('Small', fontSize=7, fontName='Helvetica',
                                  alignment=TA_CENTER, textColor=colors.grey)

    story = []

    # Bank header
    story.append(Paragraph(acct.bank_name.upper(), title_style))
    story.append(Paragraph("Account Statement", ParagraphStyle(
        'Sub', fontSize=11, fontName='Helvetica', alignment=TA_CENTER, spaceAfter=8)))
    story.append(Spacer(1, 4*mm))

    # Account info table
    info_data = [
        ["Account Holder", acct.holder_name, "Account No.", acct.account_number],
        ["IFSC Code", acct.ifsc, "Branch", acct.bank_code + " Main Branch"],
        ["Statement Period", "01/01/2026 - 31/03/2026", "Currency", "INR"],
    ]
    info_table = Table(info_data, colWidths=[40*mm, 60*mm, 35*mm, 55*mm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6*mm))

    # Transaction table — paginate at max_rows
    df = statement_df.head(max_rows)
    col_headers = list(df.columns)
    table_data = [col_headers]
    for _, row in df.iterrows():
        table_data.append([
            str(row[c]) if str(row[c]) not in ('nan', '') else '-'
            for c in col_headers
        ])

    # Auto column widths: date+ref narrow, narration wide, amounts medium
    n_cols = len(col_headers)
    page_width = A4[0] - 30*mm
    col_widths = []
    for col in col_headers:
        col_l = col.lower()
        if any(k in col_l for k in ('date', 'ref', 'cheque', 'no')):
            col_widths.append(22*mm)
        elif any(k in col_l for k in ('narration', 'description', 'remarks', 'particulars')):
            col_widths.append(None)   # will fill remaining
        else:
            col_widths.append(26*mm)
    # Fill the narration column with whatever space is left
    fixed = sum(w for w in col_widths if w is not None)
    remaining = page_width - fixed
    col_widths = [remaining if w is None else w for w in col_widths]

    txn_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    txn_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A3C6E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EEF2F7')]),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        # Grid
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#AAAAAA')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#CCCCCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(txn_table)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        "This is a computer-generated statement. For queries contact your branch.",
        small_style
    ))

    doc.build(story)


def render_scanned_statement_image(acct, statement_df, out_path, max_rows=28):
    """
    Render a basic statement table to a PNG and apply a light blur/rotation
    /noise pass to emulate a phone-camera scan of a printed statement -
    a stand-in test fixture for the OCR ingestion path (Phase 1 calls out
    a messy/scanned input as a meaningful demo differentiator).
    """
    df = statement_df.head(max_rows)
    cols = list(df.columns)
    col_width = 230
    row_height = 34
    margin = 50
    width = margin * 2 + col_width * len(cols)
    height = margin * 2 + 130 + row_height * (len(df) + 1)

    img = Image.new("L", (width, height), color=250)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 16)
    font_bold = ImageFont.truetype(FONT_PATH_BOLD, 18)
    font_title = ImageFont.truetype(FONT_PATH_BOLD, 22)

    y = margin
    draw.text((margin, y), f"{acct.bank_name} - Account Statement", font=font_title, fill=10)
    y += 32
    draw.text((margin, y), f"Account Holder: {acct.holder_name}", font=font, fill=20)
    y += 24
    draw.text((margin, y), f"Account Number: {acct.account_number}   IFSC: {acct.ifsc}", font=font, fill=20)
    y += 40

    header_y = y
    for c_idx, col in enumerate(cols):
        x = margin + c_idx * col_width
        wrapped = "\n".join(textwrap.wrap(str(col), width=18))
        draw.text((x, header_y), wrapped, font=font_bold, fill=0)
    y = header_y + row_height

    draw.line([(margin, y - 5), (width - margin, y - 5)], fill=0, width=2)

    for _, row in df.iterrows():
        for c_idx, col in enumerate(cols):
            x = margin + c_idx * col_width
            val = row[col]
            text = "" if (val == "" or pd.isna(val)) else str(val)
            draw.text((x, y), text[:24], font=font, fill=15)
        y += row_height

    # emulate scan artifacts: slight rotation, blur, gaussian noise, then re-encode at low jpeg quality
    angle = random.uniform(-1.2, 1.2)
    img = img.rotate(angle, expand=True, fillcolor=250)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))

    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    rgb = img.convert("RGB")
    tmp_jpg = out_path.replace(".png", "_tmp.jpg")
    rgb.save(tmp_jpg, quality=random.randint(55, 75))
    Image.open(tmp_jpg).save(out_path)
    os.remove(tmp_jpg)
