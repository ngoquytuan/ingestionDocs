import os
import json
import logging
from datetime import datetime
from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import fitz  # PyMuPDF
from docx import Document

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class IngestionModule:
    def __init__(self, chunk_size=1000, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.ingested_json_dir = "../data/ingested_json"
        self.raw_documents_dir = "../data/raw_documents"
        self.supported_extensions = {
            ".pdf", ".docx", ".txt", ".md"
        }
        self.md_converter = MarkItDown(enable_plugins=True)
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text):
        return len(self.token_encoder.encode(text))

    # === Extract with heading detection ===
    def _extract_with_headings_pdf(self, file_path):
        doc = fitz.open(file_path)
        sections = []
        current_section = {"heading": "Document Start", "content": ""}

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        # Detect heading based on font size threshold
                        if span["size"] >= 14:  # heading threshold
                            if current_section["content"].strip():
                                sections.append(current_section)
                            current_section = {"heading": text, "content": ""}
                        else:
                            current_section["content"] += text + " "
        if current_section["content"].strip():
            sections.append(current_section)
        return sections

    def _extract_with_headings_docx(self, file_path):
        doc = Document(file_path)
        sections = []
        current_section = {"heading": "Document Start", "content": ""}

        for para in doc.paragraphs:
            style_name = para.style.name if para.style else ""
            text = para.text.strip()
            if not text:
                continue
            if style_name.startswith("Heading"):
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"heading": text, "content": ""}
            else:
                current_section["content"] += text + " "
        if current_section["content"].strip():
            sections.append(current_section)
        return sections

    def _extract_with_headings_md(self, file_path):
        sections = []
        current_section = {"heading": "Document Start", "content": ""}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    if current_section["content"].strip():
                        sections.append(current_section)
                    heading_text = line.strip("#").strip()
                    current_section = {"heading": heading_text, "content": ""}
                else:
                    current_section["content"] += line.strip() + " "
        if current_section["content"].strip():
            sections.append(current_section)
        return sections

    def _extract_text_with_headings(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._extract_with_headings_pdf(file_path)
        elif ext == ".docx":
            return self._extract_with_headings_docx(file_path)
        elif ext == ".md":
            return self._extract_with_headings_md(file_path)
        else:
            # Fallback for txt and unsupported heading extraction
            text = self.md_converter.convert_local(file_path).markdown
            return [{"heading": "Document Start", "content": text}]

    # === Main ingestion ===
    def ingest_document(self, file_path, document_id, title, source, version, last_updated, metadata):
        try:
            sections = self._extract_text_with_headings(file_path)
            if not sections:
                raise ValueError(f"Không trích xuất được nội dung từ {file_path}")

            processed_chunks = []
            for sec_index, section in enumerate(sections):
                heading = section["heading"]
                content = section["content"].strip()
                if not content:
                    continue

                # Chunk nội dung dài trong từng heading
                chunks = self.text_splitter.split_text(content)
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{document_id}-{sec_index:02d}-{i:03d}"
                    token_count = self._count_tokens(chunk_text)
                    processed_chunks.append({
                        "chunk_id": chunk_id,
                        "heading": heading,
                        "text": chunk_text.strip(),
                        "start_page": None,
                        "end_page": None,
                        "tokens": token_count,
                        "embedding": None
                    })

            document_data = {
                "document_id": document_id,
                "title": title,
                "source": source,
                "version": version,
                "language": "vi",
                "last_updated": last_updated,
                "metadata": {
                    "author": metadata.get("author", "Unknown"),
                    "category": metadata.get("category", "Uncategorized"),
                    "access_roles": metadata.get("access_roles", ["all"]),
                    "confidentiality_level": metadata.get("confidentiality_level", "internal"),
                    "keywords": metadata.get("keywords", []),
                    "summary": metadata.get("summary", "")
                },
                "chunks": processed_chunks
            }
            return document_data

        except Exception as e:
            logger.error(f"Lỗi khi ingest {file_path}: {str(e)}")
            return None

    def save_document_data(self, document_data):
        os.makedirs(self.ingested_json_dir, exist_ok=True)
        output_path = os.path.join(self.ingested_json_dir, f'{document_data["document_id"]}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu JSON: {output_path}")
        return True

    def _get_current_version(self, document_id):
        path = os.path.join(self.ingested_json_dir, f"{document_id}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("version")
            except:
                return None
        return None

    def process_new_document(self, file_path, document_id, title, source, metadata):
        current_version = self._get_current_version(document_id)
        new_version = "1.0"
        if current_version:
            try:
                major, minor = map(int, current_version.split("."))
                new_version = f"{major}.{minor + 1}"
            except:
                new_version = current_version + "_new"

        last_updated = datetime.now().isoformat()
        logger.info(f"Xử lý {file_path} (ID: {document_id}, Version: {new_version})")
        doc_data = self.ingest_document(file_path, document_id, title, source, new_version, last_updated, metadata)
        if doc_data and self.save_document_data(doc_data):
            logger.info(f"Hoàn tất xử lý {file_path}")
            return True
        return False

    def process_all_documents(self):
        logger.info("Bắt đầu xử lý batch...")
        for root, dirs, files in os.walk(self.raw_documents_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if not self._get_file_type(file_path) in self.supported_extensions:
                    logger.warning(f"Bỏ qua: {file_name} (không hỗ trợ)")
                    continue

                user_input = input(f"Xử lý file '{file_name}'? (y/n): ")
                if user_input.lower() != 'y':
                    logger.info(f"Bỏ qua {file_name}")
                    continue

                doc_id = os.path.splitext(file_name)[0].replace(" ", "_")
                metadata = {
                    "author": "Unknown",
                    "category": "Uncategorized",
                    "access_roles": ["all"],
                    "confidentiality_level": "internal",
                    "keywords": [],
                    "summary": ""
                }
                self.process_new_document(file_path, doc_id, file_name, file_path, metadata)
        logger.info("Hoàn tất batch.")

    def _get_file_type(self, file_path):
        return os.path.splitext(file_path)[1].lower()


if __name__ == "__main__":
    ingestor = IngestionModule()
    os.makedirs(ingestor.raw_documents_dir, exist_ok=True)
    os.makedirs(ingestor.ingested_json_dir, exist_ok=True)
    ingestor.process_all_documents()
