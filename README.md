# Ingestion Module (Vietnamese Document Preprocessing)

## 1. Mục đích

Module này dùng để:

- Đọc các tài liệu **tiếng Việt** từ nhiều định dạng (PDF, DOCX, TXT, MD, PPTX, XLSX, HTML, v.v.).
- Tách tài liệu thành các **chunk** văn bản.
- Tạo file **JSON chuẩn** chứa thông tin tài liệu, metadata, các chunk văn bản kèm số token và placeholder embedding (`null`).
- Sẵn sàng cho bước import vào FAISS hoặc các vector database khác.

---

## 2. Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10/11
- **Python:** 3.9 – 3.11
- **pip** (đi kèm với Python)
- **Git** (nếu muốn clone repo từ git)

---

## 3. Cài đặt môi trường Python ảo

Mở **Command Prompt** hoặc **PowerShell**:

```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
venv\Scripts\activate

# Nâng cấp pip
python -m pip install --upgrade pip
```

## 4. Cài đặt thư viện cần thiết

```
powershellSao chépChỉnh sửapip install fitz pymupdf python-docx langchain tiktoken markitdown
```

> **Ghi chú:**
>
> - `PyMuPDF` (tên pip là `pymupdf`) để đọc PDF.
> - `python-docx` để đọc file DOCX.
> - `langchain` để dùng `RecursiveCharacterTextSplitter`.
> - `tiktoken` để đếm token.
> - `markitdown` để trích xuất văn bản từ nhiều định dạng khác nhau.

## 5. Cấu trúc thư mục

```
plaintextSao chépChỉnh sửaproject/
│
├── ingestion_module_upgraded.py    # Script xử lý tài liệu
├── data/
│   ├── raw_documents/              # Thư mục chứa file đầu vào
│   └── ingested_json/              # Thư mục chứa file JSON đầu ra
└── README.md
```

---

## 6. Cách chạy

1. **Chuẩn bị file đầu vào**
   - Đặt tất cả tài liệu cần xử lý vào thư mục `data/raw_documents/`.
   - Hỗ trợ các định dạng: `.pdf`, `.docx`, `.txt`, `.md`, `.pptx`, `.xlsx`, `.html`, `.zip`, `.epub`.
2. **Chạy script**

   ```
   powershellSao chépChỉnh sửavenv\Scripts\activate
   python ingestion_module_upgraded.py
   ```
3. **Quy trình chạy**
   - Script sẽ tìm các file trong `data/raw_documents/`.
   - Hỏi xác nhận bạn có muốn xử lý từng file không.
   - Với mỗi file được chọn, script sẽ:
     - Đọc và trích xuất văn bản.
     - Tách thành các chunk.
     - Tạo file JSON chuẩn chứa metadata + chunks.
     - Lưu vào thư mục `data/ingested_json/`.

---

## 7. Đầu vào

- Một hoặc nhiều file tài liệu **tiếng Việt**.
- Metadata cơ bản (tác giả, danh mục, quyền truy cập, từ khóa, mức độ bảo mật, tóm tắt) được set mặc định trong script, có thể chỉnh khi gọi hàm.

---

## 8. Đầu ra

- Mỗi file đầu vào sẽ tạo **một file JSON** trong `data/ingested_json/` theo định dạng sau:

```
jsonSao chépChỉnh sửa{
  "document_id": "TECH-MANUAL-004",
  "title": "Firewall Configuration Guide",
  "source": "manuals/firewall_v3.pdf",
  "version": "3.0",
  "language": "vi",
  "last_updated": "2023-11-15T10:00:00Z",
  "metadata": {
    "author": "IT Department",
    "category": "Technical Manuals",
    "access_roles": ["technician", "devops"],
    "keywords": ["firewall", "network", "security"],
    "confidentiality_level": "internal",
    "summary": "Hướng dẫn cấu hình tường lửa cho hệ thống mạng nội bộ."
  },
  "chunks": [
    {
      "chunk_id": "TECH-MANUAL-004-001",
      "text": "Để cấu hình quy tắc tường lửa chính, truy cập tab An ninh mạng và chọn 'Rule Engine'.",
      "start_page": 4,
      "end_page": 4,
      "tokens": 26,
      "embedding": null
    }
    // ... các chunk khác
  ]
}
```

---

## 9. Lưu ý

- `tokens` được tính bằng `tiktoken` (chuẩn OpenAI).
- `embedding` mặc định `null` để sau này hệ thống khác xử lý.
- `language` mặc định `"vi"`.
- Có thể thay đổi `chunk_size` và `chunk_overlap` khi khởi tạo `IngestionModule`.

## 10. Bước tiếp theo

Sau khi có các file JSON này:

- Bạn có thể import trực tiếp vào FAISS hoặc ChromaDB.
- Có thể tính embedding và cập nhật vào trường `"embedding"` trong JSON.

Nếu bạn muốn, tôi có thể viết thêm **script import JSON này vào FAISS** ngay sau bước tiền xử lý để pipeline của bạn khép kín.  
Bạn có muốn tôi làm thêm phần đó không?

Ở đây việc chia **chunk** trong `ingestion_module_upgraded.py` (và bản tôi chỉnh lại) **không dùng embedding model**, mà dựa trên **logic tách văn bản theo độ dài ký tự/tokens**.

---

## 1. Căn cứ chia chunk trong code của bạn

Hiện tại script đang dùng:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len
)
```

- `chunk_size=1000` → chia văn bản sao cho mỗi đoạn có tối đa 1000 ký tự (hoặc token nếu thay đổi hàm `length_function`).
- `chunk_overlap=50` → khi chia, mỗi chunk sẽ “lặp lại” 50 ký tự cuối của chunk trước sang chunk sau để tránh mất ngữ cảnh.
- `RecursiveCharacterTextSplitter` → ưu tiên tách ở dấu xuống dòng, dấu chấm câu, khoảng trắng… Nếu không tìm thấy thì mới cắt “thẳng” theo số ký tự.

Nói đơn giản:

- Nó **không hiểu ngữ nghĩa**.
- Nó chỉ cắt theo **độ dài** và cố gắng không làm vỡ câu nếu có thể.

---

## 2. Có cần dùng embedding model để chia chunk không?

Không bắt buộc, nhưng có một số cách nâng cao:

| Phương pháp chia                                                                   | Ưu điểm                                                             | Nhược điểm                                                                 |
|------------------------------------------------------------------------------------|---------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Theo độ dài (hiện tại)**                                                             | Nhanh, đơn giản, không cần model.                                   | Có thể cắt giữa câu, giữa đoạn logic → mất nghĩa.                          |
| **Semantic chunking** (dùng embedding hoặc mô hình NLP để xác định ranh giới nội dung) | Giữ được câu/đoạn có nghĩa, giúp LLM hiểu ngữ cảnh hơn khi trả lời. | Chậm hơn, phức tạp hơn, cần tính embedding cho toàn bộ câu trước khi chia. |
| **Theo cấu trúc tài liệu** (heading, mục lục, bullet points)                           | Rất phù hợp cho manual, quy định, hướng dẫn kỹ thuật.               | Cần parser tốt cho từng loại file.                                         |

---

## 3. Đề xuất cho trường hợp của bạn (100 trang A4, tài liệu kỹ thuật tiếng Việt)

- Nếu muốn **nhanh, đơn giản** → giữ như hiện tại (chunk theo ký tự/tokens, có overlap).
- Nếu muốn **độ chính xác cao hơn khi tìm kiếm** → dùng **semantic chunking** hoặc **chunk theo heading**.

Ví dụ **semantic chunking**:

1. Dùng một tokenizer hoặc sentence splitter (`underthesea`, `pyvi`) để tách câu.
2. Tạo embedding cho từng câu.
3. Gom các câu gần nhau **nếu vector cosine similarity > ngưỡng** → thành chunk.
4. Mỗi chunk sẽ là 1 “đoạn ý nghĩa” thay vì bị cắt ngang.

---

## 4. Kết luận

- Bạn **không cần dùng embedding model để chia chunk** nếu ưu tiên tốc độ.
- Nếu muốn kết quả tìm kiếm chính xác hơn (đặc biệt khi văn bản nhiều câu dài hoặc chứa công thức, heading), bạn nên kết hợp:
  - **Chunk theo heading** nếu tài liệu có cấu trúc rõ ràng.
  - Hoặc **semantic chunking** nếu muốn LLM nhận ngữ cảnh tốt nhất.

---

Nếu bạn muốn, tôi có thể chỉnh luôn script của bạn để:

- **Phân tích heading** từ tài liệu (PDF/DOCX/MD).
- **Kết hợp semantic chunking** bằng embedding model `AITeamVN/Vietnamese_Embedding`.

Như vậy chunk sẽ vừa gọn, vừa giữ nguyên ý.  
 Bạn có muốn tôi làm phiên bản này không?

Ok, tôi sẽ chỉnh lại chương trình để **ưu tiên phân tách nội dung theo heading** trước khi chunk text.  
 Cách tiếp cận:

- **PDF**: Dùng `PyMuPDF` để lấy font size, font weight, hoặc ký hiệu heading (số chương, số mục) để xác định heading.
- **DOCX**: Dùng `python-docx` đọc style (Heading 1, Heading 2, …).
- **MD**: Nhận diện heading qua ký tự `#`, `##`, `###`.
- Nếu không tìm thấy heading → fallback về cách chunk theo `RecursiveCharacterTextSplitter` như cũ.

---

### Bản sửa đổi (tích hợp phân tích heading)

```python
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
```

---

### **Điểm mới**

- **Phân tích heading** cho từng loại file:
  - PDF: dựa vào **font size >= 14** để coi là heading.
  - DOCX: dựa vào style name "Heading X".
  - MD: dựa vào ký tự `#`.
- Mỗi heading trở thành một **section**, và chỉ chunk khi nội dung section quá dài.
- Giữ heading trong từng chunk (`"heading": ...`) để khi query FAISS có thể ưu tiên hiển thị heading liên quan.
- Output JSON vẫn theo format chuẩn (có tokens và embedding null).

---

Nếu bạn muốn, tôi có thể **viết thêm script import JSON này vào FAISS** để pipeline của bạn hoàn thiện từ ingestion → search → Ollama.  
 Bạn có muốn tôi làm luôn không?
