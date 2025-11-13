"""
convert_medquad_xml_to_txt.py
------------------------------
Converts NIH MedQuAD XML files from subfolders (even if files have no extension) to cleaned text, JSON, or CSV.

USAGE (see terminal commands below)
"""

import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def clean_text(text: Optional[str]) -> str:
    """Remove HTML tags, extra spaces, and control chars."""
    if not text:
        return ""
    try:
        text = BeautifulSoup(text, "lxml").get_text()
    except Exception as e:
        logging.warning(f"BeautifulSoup error: {e}")
        text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    return text.strip()

def parse_medquad_xml(file_path: str) -> List[Dict[str, Any]]:
    docs = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"XML parse failed for {file_path}: {e}")
        return docs

    # Loop through QAPairs in each Document (your XML root is Document)
    for qapairs in root.findall(".//QAPairs"):
        for qa in qapairs.findall("QAPair"):
            q_text = clean_text(qa.findtext("Question") or "")
            a_text = clean_text(qa.findtext("Answer") or "")
            if q_text and a_text:
                docs.append({
                    "doc_id": root.attrib.get("id", "unknown"),
                    "category": root.attrib.get("source", "unknown"),
                    "question": q_text,
                    "answer": a_text
                })
    return docs

def process_medquad_files(all_files: List[Path], parallel: bool = False, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    files_to_parse = all_files
    if limit:
        files_to_parse = files_to_parse[:limit]
    all_records = []
    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        logging.info(f"🔄 Parallel parsing enabled ({os.cpu_count()} workers)...")
        with ProcessPoolExecutor() as executor:
            results_out = list(tqdm(executor.map(parse_medquad_xml, files_to_parse), total=len(files_to_parse), desc="Parsing files as XML"))
            for batch in results_out:
                all_records.extend(batch)
    else:
        for p in tqdm(files_to_parse, desc="Parsing files as XML"):
            records = parse_medquad_xml(str(p))
            all_records.extend(records)
    return all_records

def save_output(all_records: List[Dict[str, Any]], output_dir: str, prefix: str,
                save_json: bool = False, save_csv: bool = False, dry_run: bool = False):
    if not all_records:
        logging.warning("No records to save!")
        return
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, f"{prefix}_medquad_qa_corpus.txt")
    if not dry_run:
        with open(text_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(f"Q: {rec['question']}\nA: {rec['answer']}\n\n")
        logging.info(f"✅ Saved cleaned text corpus: {text_path}")

    if save_json:
        json_path = os.path.join(output_dir, f"{prefix}_medquad_qa_corpus.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(all_records, jf, indent=2, ensure_ascii=False)
        logging.info(f"✅ Saved structured JSON: {json_path}")

    if save_csv:
        if not HAS_PANDAS:
            logging.error("CSV output requires pandas. Install with `pip install pandas`.")
        else:
            csv_path = os.path.join(output_dir, f"{prefix}_medquad_qa_corpus.csv")
            import pandas as pd
            df = pd.DataFrame(all_records)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logging.info(f"✅ Saved CSV: {csv_path}")

    logging.info(f"📊 Total Q/A pairs extracted: {len(all_records)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MedQuAD XML → text/JSON/CSV. Handles files with or without .xml extension!")
    parser.add_argument("--input_dir", type=str, required=True, help="MedQuAD folder path with all files in subfolders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder (e.g., ./docs)")
    parser.add_argument("--json", action="store_true", help="Also save JSON output")
    parser.add_argument("--csv", action="store_true", help="Also save CSV output (requires pandas)")
    parser.add_argument("--prefix", type=str, default="medquad", help="Output file prefix")
    parser.add_argument("--parallel", action="store_true", help="Parallel parse files (recommended for big corpora)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files (for testing)")
    parser.add_argument("--dry_run", action="store_true", help="Only show stats, don't write files")
    args = parser.parse_args()

    # Find all files (not folders), extension or not
    print("DEBUG: Searching in", Path(args.input_dir).resolve())
    all_files = [p for p in Path(args.input_dir).rglob("*") if p.is_file()]
    print("DEBUG: Found", len(all_files), "potential XML files. First two:", all_files[:2])

    if not all_files:
        logging.error(f"No files found in {args.input_dir}")
        exit(1)

    all_records = process_medquad_files(all_files, parallel=args.parallel, limit=args.limit)
    save_output(all_records, args.output_dir, args.prefix, save_json=args.json, save_csv=args.csv, dry_run=args.dry_run)
