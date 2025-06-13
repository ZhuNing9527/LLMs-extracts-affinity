#!/usr/bin/env python3
import os
import re
import argparse
# import openai # No longer using openai library for API calls
import google.generativeai as genai # Import Google Generative AI
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed

# THE API KEY IS HARDCODED BELOW. BE CAREFUL IF SHARING THIS SCRIPT.
HARDCODED_API_KEY = "xxxxxxxx"

# -----------------------------
def preprocess_text(text: str) -> str:
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'(Header|Footer):\s*.*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\n\s*', ' ', text)

    match = re.search(r"References", text, flags=re.IGNORECASE)
    if match:
        text = text[:match.start()]

    return re.sub(r'\s+', ' ', text).strip()

# -----------------------------
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

# MODIFIED FUNCTION FOR GEMINI 1.5 PRO
def call_gemini_api(prompt: str) -> str:
    try:
        # API Key is now hardcoded
        genai.configure(api_key=HARDCODED_API_KEY)

        system_instruction = "You are a helpful assistant specializing in data extraction from scientific literature."

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro", # Uses Gemini 1.5 Pro
            system_instruction=system_instruction
        )

        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_p=1.0
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"ERROR_CALLING_GEMINI_API: {str(e)}"


def extract_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    raw_text = []
    for page in doc:
        raw_text.append(page.get_text("text", sort=True))
    doc.close()
    clean = preprocess_text(" ".join(raw_text))
    prompt = build_prompt(clean)
    return call_gemini_api(prompt)

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
                if tsv.startswith("ERROR_CALLING_GEMINI_API"):
                    print(f"[✗] {fname} ERROR: API call failed. Details: {tsv}. TSV not written.")
                    continue

                with open(out_path, "w", encoding="utf-8") as fout:
                    fout.write("\t".join(headers) + "\n")
                    fout.write(tsv + "\n")
                print(f"[✓] {fname} → {stem}.tsv")
            except Exception as e:
                print(f"[✗] {fname} ERROR during file processing or future result: {e}")

    print("All done! Individual TSV files have been written next to each PDF.")

if __name__ == "__main__":
    main()