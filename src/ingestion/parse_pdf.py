import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This file inspects a PDF and shows how it will be processed by the RAG pipeline.
#
# 1. Loads a selected PDF file from the raw_pdfs directory.
# 2. Extracts the full text content from the PDF.
# 3. Saves the extracted text to a .txt file for inspection.
# 4. Splits the text into chunks using the same settings as ingest_pdfs.py.
# 5. Saves all generated chunks to a separate text file.
# 6. Reports the total number of chunks created.
# 7. Helps validate PDF parsing and chunking before ingestion into Qdrant.

# --- CONFIGURATION ---
# The name of the PDF you want to inspect (Change this string as needed)
TARGET_PDF_NAME = "Bookmarks_online.pdf" 

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")

def inspect_pdf(pdf_filename):
    pdf_path = os.path.join(RAW_PDF_DIR, pdf_filename)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
        return

    print(f"🔍 Inspecting: {pdf_filename}...")

    # 1. Extract Full Text
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    except Exception as e:
        print(f"❌ Failed to parse PDF: {e}")
        return

    # 2. Save Full Text File (Request #1)
    # Result: pdf_name.pdf -> pdf_name.txt
    full_txt_filename = pdf_filename.replace(".pdf", ".txt")
    full_txt_path = os.path.join(RAW_PDF_DIR, full_txt_filename)
    
    with open(full_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"✅ Full text saved to: {full_txt_filename}")

    # 3. Create Chunks (Request #2)
    # Using the same settings as your ingest_pdfs.py for consistency
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    # 4. Save Chunked Text File
    # Result: pdf_name.pdf -> chunks_pdf_name.txt
    chunks_txt_filename = f"chunks_{pdf_filename.replace('.pdf', '.txt')}"
    chunks_txt_path = os.path.join(RAW_PDF_DIR, chunks_txt_filename)

    with open(chunks_txt_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            separator = f"\n\n{'='*20} CHUNK - {i} {'='*20}\n\n"
            f.write(separator)
            f.write(chunk)
            
    print(f"✅ Chunked text saved to: {chunks_txt_filename}")
    print(f"📊 Total Chunks created: {len(chunks)}")

if __name__ == "__main__":
    inspect_pdf(TARGET_PDF_NAME)