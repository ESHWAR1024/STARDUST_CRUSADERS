"""
Standalone PDF inspector — bypasses the ingestion pipeline entirely.
Run this directly against your statement PDF to see EXACTLY what
pdfplumber extracts: raw table headers, raw rows, and page-1 text.

Usage:
    python3 debug_pdf.py SmartCX360_MAY_2026_amxxxxxxxx05.pdf
"""

import sys
import pdfplumber


def main(file_path):
    print(f"\n{'='*70}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*70}\n")

    with pdfplumber.open(file_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")

        for page_num, page in enumerate(pdf.pages):
            print(f"\n--- PAGE {page_num + 1} ---")

            if page_num == 0:
                text = page.extract_text() or ""
                print("\n[First 800 chars of page text — used for bank name detection]")
                print("-" * 70)
                print(text[:800])
                print("-" * 70)

            # Strategy 1: lines-based
            tables_lines = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            })
            print(f"\n[lines-strategy] Found {len(tables_lines)} table(s)")
            for t_idx, table in enumerate(tables_lines):
                print(f"  Table {t_idx}: {len(table)} rows")
                for r_idx, row in enumerate(table[:5]):
                    print(f"    row[{r_idx}]: {row}")
                if len(table) > 5:
                    print(f"    ... ({len(table) - 5} more rows)")

            # Strategy 2: default
            tables_default = page.extract_tables()
            print(f"\n[default-strategy] Found {len(tables_default)} table(s)")
            for t_idx, table in enumerate(tables_default):
                print(f"  Table {t_idx}: {len(table)} rows")
                for r_idx, row in enumerate(table[:5]):
                    print(f"    row[{r_idx}]: {row}")
                if len(table) > 5:
                    print(f"    ... ({len(table) - 5} more rows)")

            if page_num >= 1:
                # Don't spam for huge PDFs — show first 2 pages of table detail in full,
                # just confirm presence for the rest.
                pass

    print(f"\n{'='*70}")
    print("Done. Copy/paste this whole output back.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 debug_pdf.py <path_to_pdf>")
        sys.exit(1)
    main(sys.argv[1])