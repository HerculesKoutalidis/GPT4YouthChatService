import os
import uuid
import logging
import fitz  # PyMuPDF
from langdetect import detect, LangDetectException
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This file builds the RAG knowledge base by processing PDF documents.
#
# 1. Reads all PDF files from the raw_pdfs directory.
# 2. Extracts text content from each PDF.
# 3. Detects the document language and filters unsupported languages.
# 4. Splits the text into overlapping chunks.
# 5. Generates embeddings for each chunk using a SentenceTransformer model.
# 6. Creates metadata (payload) for each chunk.
# 7. Creates the Qdrant collection if it does not already exist.
# 8. Uploads chunk embeddings and metadata to Qdrant.
# 9. Stores the data for semantic retrieval in the chatbot RAG pipeline.

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
RAW_PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw_pdfs")
COLLECTION_NAME = "eu_job_market"

# Connect to local Qdrant (since we will run this script from the terminal)
QDRANT_URL = "http://localhost:6343" 
qdrant = QdrantClient(url=QDRANT_URL)
encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# --- Initialize Qdrant Collection ---
def init_qdrant():
    """Creates the collection if it doesn't exist."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating collection '{COLLECTION_NAME}' in Qdrant...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists.")

# --- Processing Pipeline ---
def process_pdfs():
    """Reads PDFs, filters languages, chunks text, and uploads to Qdrant."""
    init_qdrant()
    
    # 500 tokens roughly translates to ~2000 characters. 15% overlap is ~300 chars.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    if not os.path.exists(RAW_PDF_DIR):
        logger.error(f"Directory not found: {RAW_PDF_DIR}")
        return

    for filename in os.listdir(RAW_PDF_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(RAW_PDF_DIR, filename)
        logger.info(f"Processing: {filename}")
        
        # 1. Extract Text
        try:
            doc = fitz.open(filepath)
            full_text = "\n".join([page.get_text() for page in doc])
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            continue

        # 2. Filter Language (Discard bg, uk)
        try:
            lang = detect(full_text[:2000]) # Detect based on first 2000 chars
            if lang in ['bg', 'uk']:
                logger.warning(f"Skipping {filename} due to unsupported language: {lang}")
                continue
        except LangDetectException:
            logger.warning(f"Could not detect language for {filename}. Skipping.")
            continue

        # 3. Chunk Text
        chunks = text_splitter.split_text(full_text)
        logger.info(f"Created {len(chunks)} chunks for {filename}.")

        # 4. Prepare Embeddings and Metadata
        points = []
        document_id = str(uuid.uuid4()) # Unique ID for the whole document
        
        for i, chunk_text in enumerate(chunks):
            # The Metadata Schema (Payload)
            payload = {
                "document_id": document_id,
                "file_name": filename,
                "chunk_index": i,
                "text": chunk_text,
                # Placeholders for advanced metadata - you can update these later!
                "country": "Unknown", 
                "title": filename.replace(".pdf", ""),
                "summary": "",
                "tags": [],
                "who_it_helps": "",
                "how_it_helps": ""
            }
            
            vector = encoder.encode(chunk_text).tolist()
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), # Unique ID for the specific chunk
                    vector=vector,
                    payload=payload
                )
            )

        # 5. Upload to Qdrant
        if points:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            logger.info(f"Successfully uploaded {len(points)} chunks for {filename} to Qdrant.\n")

if __name__ == "__main__":
    process_pdfs()
    logger.info("Ingestion complete!")