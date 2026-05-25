from pathlib import Path
from typing import List, Tuple

from docling_core.types.doc import PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import TextItem, TableItem, SectionHeaderItem

import logging
from pypdf import PdfReader
import gc
import psutil
import os

OUTPUT_MD_DIR = "D:/Cursos/Agentes/financial_deep_research_agent/data/procesed/markdown"
OUTPUT_IMGES_DIR = "D:/Cursos/Agentes/financial_deep_research_agent/data/procesed/images"
OUTPUT_TABLES_DIR = "D:/Cursos/Agentes/financial_deep_research_agent/data/procesed/tables"
DATA_DIR = "D:/Cursos/Agentes/financial_deep_research_agent/data/raw"
LOG_DIR = "D:/Cursos/Agentes/financial_deep_research_agent/logs"

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

def convert_pdf_to_docling(pdf_file: Path, page_range):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return doc_converter.convert(source=pdf_file,page_range=page_range)

#save document on markdown 
def append_text(md_dir: Path, pdf_file: Path, markdown_text: str):
    with open(md_dir / f"{pdf_file.stem}.md", "a", encoding="utf-8") as f:
        f.write(markdown_text)

#Save images
def save_page_images(doc_convert, img_dir: Path):
    """Find and save pages with large images (> 500x500 pix¿xels)"""
    pages_to_save = set()

    for item in doc_convert.document.iterate_items():
        element = item[0]
        if isinstance(element, PictureItem):
            image = element.get_image(doc_convert.document)

            if image.size[0] > 500 and image.size[1] > 500:
                page_no = element.prov[0].page_no if element.prov  else None

                if page_no:
                    pages_to_save.add(page_no)
        
        #save iamges
        for page_no in pages_to_save:
            page = doc_convert.document.pages[page_no]

            page.image.pil_image.save(img_dir/f"page_{page_no}.png", "PNG")

#save tables with context
def save_tables_with_context(doc_converter,table_dir:Path ,n_before = 2):
    """Extract tables with n context paragraphs/headers before"""
    elements = []
    for item in doc_converter.document.iterate_items():
        elements.append(item[0])
    
    existing = list(table_dir.glob("table_*.md"))
    table_count = len(existing)
    for i, element in enumerate(elements):
        if isinstance(element, TableItem):
            table_count += 1
            page_no = element.prov[0].page_no if element.prov else None
            table_md = element.export_to_dataframe(doc=doc_converter.document)

            context = []
            count = 0
            for j in range(i-1,-1,-1):
                if count >= n_before:
                    break
                prev = elements[j]
                if isinstance(prev, (TextItem, SectionHeaderItem)):
                    context.insert(0, prev.text)
                    count += 1
            content_md = f"**Page:** {page_no}\n\n" + "\n\n".join(context) + "\n\n" + table_md.to_markdown(index=False)
            with open(table_dir / f"table_{table_count}_page_{page_no}.md", "w", encoding="utf-8") as f:
                f.write(content_md)

def setup_logging(log_dir: Path, pdf_file: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{pdf_file.stem}.log"

    # Forzar reconfiguración
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path

def log_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    logging.info(f"Memoria en uso: {mem:.1f} MB")

def extract_pdf_content(pdf_file, page_range = 8):
    logging.info(pdf_file.stem)
    setup_logging(Path(LOG_DIR), pdf_file)

    reader = PdfReader(pdf_file)
    pages_length = len(reader.pages)
    del reader

    metadata = exctract_metada_from_filename(pdf_file.stem)
    company_name = metadata["company_name"]

    md_dir = Path(OUTPUT_MD_DIR) / company_name
    img_dir = Path(OUTPUT_IMGES_DIR) / company_name / pdf_file.stem
    table_dir = Path(OUTPUT_TABLES_DIR) / company_name / pdf_file.stem

    for inicio in range(0, pages_length,page_range):
        fin = min(inicio + 10, pages_length)



        for dir_path in [md_dir, img_dir, table_dir]:
            dir_path.mkdir(parents = True, exist_ok=True)

        doc_converter = convert_pdf_to_docling(pdf_file, page_range=(inicio + 1, fin))
        markdown_text = doc_converter.document.export_to_markdown(page_break_placeholder="<!---page break--->")
        log_memory()

        # guaradar en markdown 
        append_text(md_dir, pdf_file ,markdown_text)

        #Save images
        save_page_images(doc_converter, img_dir)

        #Save tables
        save_tables_with_context(doc_converter, table_dir)

        del doc_converter
        gc.collect()
        log_memory()

        logging.info(f"Total de páginas: {pages_length} -- escritas: {fin}")


def run_exctractor_data(start_from: int = 0):
    data_path = Path(DATA_DIR)
    pdf_files = data_path.rglob("*.pdf")

    print(DATA_DIR)

    for idx, pdf_file in enumerate(pdf_files):
        if idx < start_from:
            print(f"Saltando: [{idx}]: {pdf_file.name}")
            continue
        print(f"[{idx}] Procesando: {pdf_file.name}")
        extract_pdf_content(pdf_file, page_range=5)

def main():
    print("="*50)
    # start_from = -1 al pdf que se desea iniciar 
    run_exctractor_data(start_from=6)
    print("ejecutando")

if __name__ == "__main__":
    main()