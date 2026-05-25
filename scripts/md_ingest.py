import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import  PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

import hashlib
from pathlib import Path
from tqdm import tqdm

if load_dotenv():
    print("Cargado correctamente")

url= os.getenv("QDRANT_URL")
qdrant_client = QdrantClient(
    url=url
)

# paths 
MARKDOWN_DIR = Path(__file__).parent.parent / "data/procesed/markdown"
TABLES_DIR = Path(__file__).parent.parent / "data/procesed/tables"
IMAGES_DES_DIR = Path(__file__).parent.parent / "data/procesed/images_desc"

#qdrant configuration
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "models/gemini-embedding-001"

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name= "Qdrant/bm25" )

vector_store = QdrantVectorStore.from_documents(
  documents=[],
  embedding=embeddings,
  sparse_embedding=sparse_embeddings,
  url=url,
  collection_name=COLLECTION_NAME,
  retrieval_mode=RetrievalMode.HYBRID,
  force_recreate=False,
 )

def exctract_metada_from_filename(filename: str) -> dict:
    """Ectract metadata from filenama.
    Examples:
        - Amazon 10-Q Q1 2024.pdf
        - Microsoft 10-k 2023.pdf
    """
    filename = filename.replace(".pdf","").replace(".md","")
    parts = filename.split()

    return {
        "company_name": parts[0],
        "doc_type": parts[1],
        "fical_quarter": parts[2] if len(parts) == 4 else None,
        "fiscal_year": parts[-1] 
    }

def compute_file_has(file_path: Path):

    sha256_hash = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

all_points = vector_store.client.scroll(
    collection_name=COLLECTION_NAME,
    limit=10_00,
    with_payload=True
)

def get_processed_hashes():
    processed_hashes=set()
    offset=None
    
    while True:
        points, offset = vector_store.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10_000,
            with_payload=True,
            offset=offset
        )

        if not points:
            break

        processed_hashes.update(point.payload['metadata']['file_hash'] for point in points)

        if offset is None:
            break
    return processed_hashes

import re
def extract_page_number(file_path: Path):
    pattern = r'page_(\d+)'
    match = re.search(pattern=pattern, string=file_path.stem)
    return int(match.group(1)) if match else None

def ingest_file_in_db(file_path: Path, processed_hashes):
    file_hash = compute_file_has(file_path)
    if file_hash in processed_hashes:
        return print(f"El siguiente archivo ya fue subido: {file_path}")

    path_str = str(file_path)
    if 'markdown' in path_str:
        content_type = 'text'
        doc_name = file_path.name
    elif 'tables' in path_str:
        content_type = 'tables'
        doc_name = file_path.parent.name
    elif 'images_desc' in path_str:
        content_type = 'image'
        doc_name = file_path.parent.name
    else:
        content_type = 'unknown'
        doc_name = file_path.name

    content = file_path.read_text(encoding='utf-8')
    base_metadata = exctract_metada_from_filename(doc_name)
    base_metadata.update({
        'content_type': content_type,
        'file_hash': file_hash,
        'source_file': doc_name
    })

    if content_type == 'text':
        #write methid for ingesting markdown data
        pages = content.split('<!---page break--->')
        documents = []
        for idx, page in enumerate(pages, start=1):
            metadata = base_metadata.copy()
            metadata.update({'page':idx})
            documents.append(Document(page_content=page, metadata=metadata))
        
        vector_store.add_documents(documents)

    else:
        #write method to ingest images desc and tables .md tada
        page_num = extract_page_number(file_path)
        metadata = base_metadata.copy()
        metadata.update({'page':page_num})
        documents = [Document(page_content=content, metadata=metadata)]

        vector_store.add_documents(documents)

    processed_hashes.add(file_hash)

processed_hashes = get_processed_hashes()
base_path = Path(__file__).parent.parent / "notebooks/documents/"
print(f"Directorio: {base_path}")
all_md_files = list(base_path.rglob("*.md"))
print(f"Archivos encontrados: {len(all_md_files)}")

def main():
    for md_file in tqdm(all_md_files):
        ingest_file_in_db(md_file, processed_hashes)

if __name__ == "__main__":
    main()

