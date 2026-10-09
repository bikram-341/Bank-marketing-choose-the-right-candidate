"""
Dataset Downloader Utility for UCI Bank Marketing Dataset.

Fetches the authentic benchmark dataset from the official UCI Machine Learning Repository
and extracts 'bank-full.csv' (~45,211 rows) and 'bank.csv' (~4,521 rows) into the data/ directory.

UCI Citation:
S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing.
Decision Support Systems, In Press, http://dx.doi.org/10.1016/j.dss.2014.03.001
"""

import os
import sys
import zipfile
import io
import urllib.request
from pathlib import Path


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET_DIR = Path("data")


def download_and_extract_dataset(target_dir: Path = TARGET_DIR, verbose: bool = True) -> bool:
    """
    Downloads and extracts bank-full.csv and bank.csv to target_dir.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    bank_full = target_dir / "bank-full.csv"
    bank_sample = target_dir / "bank.csv"

    if bank_full.exists() and bank_full.stat().st_size > 100000:
        if verbose:
            print(f"[INFO] Dataset already present at: {bank_full} ({bank_full.stat().st_size:,} bytes)")
        return True

    if verbose:
        print(f"[INFO] Fetching Bank Marketing dataset from UCI ML Repository:")
        print(f"       URL: {UCI_ZIP_URL}")

    try:
        req = urllib.request.Request(
            UCI_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BankMarketingML/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_content = response.read()

        if verbose:
            print(f"[INFO] Downloaded {len(zip_content):,} bytes. Extracting archives...")

        with zipfile.ZipFile(io.BytesIO(zip_content)) as outer_zip:
            # The UCI zip contains nested zip files: 'bank.zip' or 'bank-additional.zip'
            namelist = outer_zip.namelist()
            if verbose:
                print(f"[INFO] Archive contents: {namelist}")

            # Check if outer zip directly contains bank-full.csv
            extracted_any = False
            for member in namelist:
                if member in ["bank-full.csv", "bank.csv", "bank-additional-full.csv"]:
                    outer_zip.extract(member, target_dir)
                    extracted_any = True
                elif member.endswith(".zip"):
                    # Extract nested zip into memory
                    nested_bytes = outer_zip.read(member)
                    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner_zip:
                        for inner_member in inner_zip.namelist():
                            if inner_member.endswith(".csv"):
                                inner_zip.extract(inner_member, target_dir)
                                extracted_any = True

        if verbose:
            print(f"[SUCCESS] Dataset successfully saved in: {target_dir.resolve()}")
            for f in target_dir.glob("*.csv"):
                print(f"          - {f.name} ({f.stat().st_size:,} bytes)")

        return True

    except Exception as e:
        if verbose:
            print(f"[ERROR] Could not download dataset automatically: {e}", file=sys.stderr)
            print("[GUIDE] You can manually download the Bank Marketing dataset from:", file=sys.stderr)
            print("        https://archive.ics.uci.edu/dataset/222/bank+marketing", file=sys.stderr)
            print("        and place 'bank-full.csv' or 'bank.csv' inside the 'data/' folder.", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = download_and_extract_dataset(verbose=True)
    sys.exit(0 if success else 1)
