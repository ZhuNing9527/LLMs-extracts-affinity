#!/usr/bin/env python3
import os
import re
import argparse
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic

def preprocess_text(text: str) -> str:
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'(Header|Footer):\s*.*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\n\s*', ' ', text)
    match = re.search(r"References", text, flags=re.IGNORECASE)
    if match:
        text = text[:match.start()]
    return re.sub(r'\s+', ' ', text).strip()

def build_prompt(clean_text: str) -> str:
    return f"""
You are an expert in extracting data from scientific literature. Your task is to extract all the "Ligand name", "Receptor protein name", "Receptor protein organism source", "Affinity value", "Wet lab method for affinity measurement" and "corresponding complex PDBID" from the converted scientific literature text.

Special attention:
- Please note that the data is extracted directly from the literature; please do not infer or tamper with the data.

Output requirements:
- The output must be in TSV format, and each line can only be one wet test measurement of affinity.
- There are exactly 6 columns in this order.

- Columns:
1. Ligand name (extract peptide ligand name from converted scientific literature text.)
2. Receptor protein name (extract the protein name from the converted text format.)
3. Receptor protein organism source (such as "Homo sapiens"; "Mus musculus"; "Rattus norvegicus"; "Hepatitis C virus", etc.)
4. Affinity value (for example: "Kd=1.2 nM"; "Ki=2.3 pM"; "IC50=11 uM"; "Ki=1.13 uM"; must follow the example format.)
5. Wet lab method for affinity measurement (for example: "isothermal titration calorimetry"; "surface plasmon resonance"; "fluorescence polarization"; "biomembrane interference method"; "ELISA"; "enzyme activity assay"; etc.)
6. Corresponding complex PDBID
Cleaned text for extraction:
{clean_text}
""".strip()

def call_openai_api(prompt: str) -> str:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specializing in data extraction from scientific literature."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        top_p=1.0,
    )
    return response.content.strip()

def extract_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    raw_text = []
    for page in doc:
        raw_text.append(page.get_text("text"))
    doc.close()
    clean = preprocess_text(" ".join(raw_text))
    prompt = build_prompt(clean)
    return call_openai_api(prompt)

def main():
    parser = argparse.ArgumentParser(description="SCI PDF to TSV")
    parser.add_argument("--input_dir", "-i", required=True, help="PDF dir")
    parser.add_argument("--output_file", "-o", default="results.tsv", help="TSV name (ignored, individual files will be created)")
    args = parser.parse_args()

    headers = [
        "Ligand name",
        "Receptor protein name",
        "Receptor protein organism source",
        "Affinity value",
        "Wet lab method for affinity measurement",
        "Corresponding complex PDBID"
    ]

    pdf_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".pdf")
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(extract_from_pdf, path): path for path in pdf_files}
        for future in as_completed(future_to_file):
            path = future_to_file[future]
            fname = os.path.basename(path)
            stem, _ = os.path.splitext(fname)
            out_path = os.path.join(args.input_dir, f"{stem}.tsv")
            try:
                tsv = future.result()
                with open(out_path, "w", encoding="utf-8") as fout:
                    fout.write("\t".join(headers) + "\n")
                    fout.write(tsv + "\n")
                print(f"[✓] {fname} → {stem}.tsv")
            except Exception as e:
                print(f"[✗] {fname} ERROR: {e}")

    print("All done! Individual TSV files have been written next to each PDF.")

if __name__ == "__main__":
    main()