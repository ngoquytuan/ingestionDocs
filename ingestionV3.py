import os
import json
import logging
from datetime import datetime, timedelta
from markitdown import MarkItDown
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken  # For token counting

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
            ".pdf", ".docx", ".txt", ".pptx", ".xlsx", ".jpg", ".jpeg", ".png", ".html", ".py", ".epub", ".md"
        }
        self.md_converter = MarkItDown(enable_plugins=True)
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

    def _extract_text_with_markitdown(self, file_path):
        """Trích xuất text từ file bằng MarkItDown."""
        try:
            result = self.md_converter.convert_local(file_path)
            return result.markdown if result and result.markdown else None
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất text từ {file_path}: {str(e)}")
            return None

    def _get_file_type(self, file_path):
        return os.path.splitext(file_path)[1].lower()

    def _validate_document(self, file_path):
        file_type = self._get_file_type(file_path)
        return file_type in self.supported_extensions

    def _count_tokens(self, text):
        return len(self.token_encoder.encode(text))

    def ingest_document(self, file_path, document_id, title, source, version, last_updated, metadata):
        """Xử lý 1 tài liệu và trả về JSON chuẩn."""
        try:
            raw_text = self._extract_text_with_markitdown(file_path)
            if not raw_text:
                raise ValueError(f"Không trích xuất được nội dung từ {file_path}")

            chunks = self.text_splitter.split_text(raw_text)
            if not chunks:
                raise ValueError(f"Không tạo được chunk từ {file_path}")

            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{document_id}-{i:03d}"
                token_count = self._count_tokens(chunk_text)

                processed_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                    "start_page": 1,
                    "end_page": 1,
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
        """Lưu file JSON."""
        try:
            os.makedirs(self.ingested_json_dir, exist_ok=True)
            output_path = os.path.join(self.ingested_json_dir, f'{document_data["document_id"]}.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Đã lưu JSON: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu JSON: {str(e)}")
            return False

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
                if not self._validate_document(file_path):
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

                if not self.process_new_document(file_path, doc_id, file_name, file_path, metadata):
                    logger.warning(f"Xử lý thất bại: {file_name}")
        logger.info("Hoàn tất batch.")
        return True


if __name__ == "__main__":
    ingestor = IngestionModule()
    os.makedirs(ingestor.raw_documents_dir, exist_ok=True)
    os.makedirs(ingestor.ingested_json_dir, exist_ok=True)
    ingestor.process_all_documents()
