#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingestion module (upgraded):
- Extracts text (to Markdown) from many formats with MarkItDown
- Parses headings (H1..H6) to create sections
- Performs semantic chunking within each section using AITeamVN/Vietnamese_Embedding
- Outputs normalized JSON with tokens and embedding=null, ready for FAISS import
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer, util
from markitdown import MarkItDown

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ingestion")

# -------------------- Utilities --------------------
def count_tokens(enc, text: str) -> int:
    return len(enc.encode(text or ""))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assumes 1D arrays
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

# -------------------- Core Class --------------------
class IngestionModule:
    def __init__(
        self,
        target_chunk_tokens: int = 550,
        chunk_overlap_tokens: int = 60,
        similarity_threshold: float = 0.55,
        min_chunk_tokens: int = 120,
    ):
        """
        Parameters
        ----------
        target_chunk_tokens : int
            Target tokens per chunk. Chunks try to stay <= this size.
        chunk_overlap_tokens : int
            Overlap tokens between consecutive chunks (applied when possible).
        similarity_threshold : float
            If cosine similarity between consecutive paragraphs drops below this,
            start a new chunk (semantic boundary).
        min_chunk_tokens : int
            Ensure a minimum chunk size so we don't produce tiny fragments.
        """
        self.ingested_json_dir = "../data/ingested_json"
        self.raw_documents_dir = "../data/raw_documents"
        self.supported_extensions = {
            ".pdf", ".docx", ".txt", ".pptx", ".xlsx", ".jpg", ".jpeg",
            ".png", ".html", ".zip", ".epub", ".md"
        }

        self.markitdown = MarkItDown(enable_plugins=True)

        # tokenization (Llama/CL100k compatible)
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # embedding model for semantic chunking
        logger.info("Loading embedding model: AITeamVN/Vietnamese_Embedding ...")
        self.embedder = SentenceTransformer("AITeamVN/Vietnamese_Embedding")

        # chunking parameters
        self.target_chunk_tokens = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.similarity_threshold = similarity_threshold
        self.min_chunk_tokens = min_chunk_tokens

    # --------------- Extraction ---------------
    def _extract_markdown(self, file_path: str) -> Optional[str]:
        try:
            result = self.markitdown.convert_local(file_path)
            md = result.markdown if result and result.markdown else None
            if not md:
                logger.warning(f"No text extracted from {file_path}")
            return md
        except Exception as e:
            logger.error(f"Extraction error [{file_path}]: {e}")
            return None

    def _validate_document(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() in self.supported_extensions

    # --------------- Heading parsing ---------------
    def _parse_sections_from_markdown(self, md: str) -> List[Dict[str, Any]]:
        """
        Parse Markdown into sections by headings. If no headings, return single section.
        Returns a list of {"heading": str, "level": int, "content": str}
        """
        lines = md.splitlines()
        sections = []
        current = {"heading": "Untitled", "level": 1, "content": []}
        found_heading = False

        heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
        for line in lines:
            m = heading_pattern.match(line.strip())
            if m:
                # push previous section
                if current["content"] or found_heading:
                    sections.append({
                        "heading": current["heading"],
                        "level": current["level"],
                        "content": "\n".join(current["content"]).strip()
                    })
                found_heading = True
                hashes = m.group(1)
                title = m.group(2).strip()
                current = {"heading": title, "level": len(hashes), "content": []}
            else:
                current["content"].append(line)

        # last section
        if current["content"] or not found_heading:
            sections.append({
                "heading": current["heading"],
                "level": current["level"],
                "content": "\n".join(current["content"]).strip()
            })

        # remove empties
        sections = [s for s in sections if s["content"].strip() or s["heading"] != "Untitled"]
        if not sections:
            sections = [{"heading": "Untitled", "level": 1, "content": md.strip()}]

        return sections

    # --------------- Semantic chunking ---------------
    def _semantic_chunk_section(self, text: str) -> List[str]:
        """
        Split a section into semantically coherent chunks using paragraph embeddings.
        """
        # split into paragraphs (by blank lines)
        raw_pars = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not raw_pars:
            return []

        # embed all paragraphs (batch)
        logger.debug(f"Embedding {len(raw_pars)} paragraphs for semantic chunking ...")
        para_embeddings = self.embedder.encode(raw_pars, convert_to_numpy=True, normalize_embeddings=True)

        chunks = []
        current = []
        current_tokens = 0

        def flush_chunk():
            nonlocal current, current_tokens
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_tokens = 0

        for i, para in enumerate(raw_pars):
            ptoks = count_tokens(self.encoder, para)

            # If empty-ish, skip
            if ptoks == 0:
                continue

            # if adding this paragraph will exceed target and current has enough tokens -> flush
            if current_tokens + ptoks > self.target_chunk_tokens and current_tokens >= max(self.min_chunk_tokens, int(self.target_chunk_tokens*0.6)):
                flush_chunk()

            # if it's the first paragraph in chunk: just start it
            if not current:
                current.append(para)
                current_tokens = ptoks
                continue

            # Check semantic boundary with previous paragraph
            sim = cosine_sim(para_embeddings[i-1], para_embeddings[i])
            if sim < self.similarity_threshold and current_tokens >= self.min_chunk_tokens:
                # start a new chunk
                # optional overlap: move tail tokens to next chunk if feasible (skipped for simplicity)
                flush_chunk()
                current.append(para)
                current_tokens = ptoks
            else:
                # continue the current chunk
                current.append(para)
                current_tokens += ptoks

        # final flush
        if current:
            flush_chunk()

        # Ensure we don't produce overly tiny trailing chunk: merge with previous if needed
        if len(chunks) >= 2:
            last_tokens = count_tokens(self.encoder, chunks[-1])
            if last_tokens < self.min_chunk_tokens:
                prev = chunks[-2] + "\n\n" + chunks[-1]
                chunks = chunks[:-2] + [prev]

        return chunks

    # --------------- Public API ---------------
    def ingest_document(self, file_path: str, document_id: str, title: str, source: str,
                        version: str, last_updated: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            md = self._extract_markdown(file_path)
            if not md:
                raise ValueError("No text extracted")

            sections = self._parse_sections_from_markdown(md)
            logger.info(f"Parsed {len(sections)} sections from headings.")

            processed_chunks = []
            chunk_counter = 0

            for s_idx, sec in enumerate(sections):
                sec_chunks = self._semantic_chunk_section(sec["content"])
                if not sec_chunks:  # fallback: keep whole section
                    sec_chunks = [sec["content"]]

                for local_idx, chunk_text in enumerate(sec_chunks):
                    # apply overlap if requested (token-based overlap not implemented per simplicity)
                    tokens = count_tokens(self.encoder, chunk_text)
                    chunk_id = f"{document_id}-{chunk_counter:03d}"
                    processed_chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "start_page": 1,
                        "end_page": 1,
                        "tokens": tokens,
                        "embedding": None,
                        "heading": sec["heading"],
                        "heading_level": sec["level"],
                        "section_index": s_idx,
                        "section_chunk_index": local_idx
                    })
                    chunk_counter += 1

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
            logger.error(f"Ingest error for {file_path}: {e}")
            return None

    def save_document_data(self, document_data: Dict[str, Any]) -> bool:
        try:
            os.makedirs(self.ingested_json_dir, exist_ok=True)
            output_path = os.path.join(self.ingested_json_dir, f'{document_data["document_id"]}.json')
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved JSON -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

    def _get_current_version(self, document_id: str) -> Optional[str]:
        json_file_path = os.path.join(self.ingested_json_dir, f"{document_id}.json")
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("version")
            except Exception:
                return None
        return None

    def process_new_document(self, file_path: str, document_id: str, title: str, source: str, metadata: Dict[str, Any]) -> bool:
        current_version = self._get_current_version(document_id)
        new_version = "1.0"
        if current_version:
            try:
                major, minor = map(int, current_version.split("."))
                new_version = f"{major}.{minor + 1}"
            except Exception:
                new_version = f"{current_version}_new"

        last_updated = datetime.now().isoformat()
        logger.info(f"Processing {file_path} (ID={document_id}, v{new_version})")

        data = self.ingest_document(
            file_path=file_path,
            document_id=document_id,
            title=title,
            source=source,
            version=new_version,
            last_updated=last_updated,
            metadata=metadata
        )

        if data and self.save_document_data(data):
            logger.info(f"Done: {file_path}")
            return True
        logger.warning(f"Failed: {file_path}")
        return False

    def process_all_documents(self) -> bool:
        logger.info("Starting batch processing...")
        for root, _, files in os.walk(self.raw_documents_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if not self._validate_document(file_path):
                    logger.warning(f"Skipping unsupported file: {file_name}")
                    continue

                user_input = input(f"Process '{file_name}'? (y/n): ").strip().lower()
                if user_input != "y":
                    logger.info(f"Skipped {file_name}")
                    continue

                doc_id = os.path.splitext(file_name)[0].replace(" ", "_")
                meta = {
                    "author": "Unknown",
                    "category": "Uncategorized",
                    "access_roles": ["all"],
                    "confidentiality_level": "internal",
                    "keywords": [],
                    "summary": ""
                }
                ok = self.process_new_document(
                    file_path=file_path,
                    document_id=doc_id,
                    title=file_name,
                    source=file_path,
                    metadata=meta
                )
                if not ok:
                    logger.warning(f"Failed to process: {file_name}")
        logger.info("Batch processing finished.")
        return True


if __name__ == "__main__":
    ingestor = IngestionModule()
    os.makedirs(ingestor.raw_documents_dir, exist_ok=True)
    os.makedirs(ingestor.ingested_json_dir, exist_ok=True)
    ingestor.process_all_documents()
