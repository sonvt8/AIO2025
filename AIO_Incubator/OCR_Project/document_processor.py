# document_processor.py
import json
import zipfile
import os
import uuid
import logging
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List

from config import ProcessorConfig

logger = logging.getLogger("document_processor")


# ==== Helpers về đường dẫn (dùng config.docs_root / unzip_root) ====
def resolve_doc_path(raw_path: str, cfg: ProcessorConfig) -> str:
    """
    Chuẩn hóa `documentPath`:
    - Nếu `raw_path` đang tồn tại (absolute hoặc relative) -> trả lại path tuyệt đối.
    - Nếu KHÔNG tồn tại, coi `raw_path` là đường dẫn logic dưới `cfg.docs_root`.
    - Hỗ trợ cả dấu '/' và '\\' trên Windows.
    """
    if not raw_path:
        return ""
    p = Path(raw_path)

    if p.exists():
        return str(p.resolve())

    rel = str(raw_path).lstrip("\\/")
    rel = Path(*Path(rel).parts)
    candidate = (cfg.docs_root / rel).resolve()
    return str(candidate)


class DocumentProcessor:
    """Xử lý tài liệu đầu vào, xác định yêu cầu OCR và thêm metadata."""

    def __init__(self, cfg: ProcessorConfig):
        self.cfg = cfg
        self.cache_pdf_results: Dict[str, Tuple[bool, bool, bool]] = {}

    # ===== Text helpers =====
    @staticmethod
    def _collect_block_text(block: Dict[str, Any], join_with: str = " ") -> str:
        spans_text: List[str] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if txt:
                    spans_text.append(txt)
        return join_with.join(spans_text) if spans_text else ""

    def _detect_watermark(self, page) -> bool:
        """Phát hiện watermark dạng text hoặc ảnh"""
        # 1 - image
        try:
            img_list = page.get_images(full=True)
            page_area = page.rect.width * page.rect.height or 1.0
            for img in img_list:
                xref = img[0]
                pix = None
                try:
                    pix = fitz.Pixmap(page.parent, xref)
                    img_area = (pix.width or 0) * (pix.height or 0)
                    if img_area / page_area > 0.8:  # ảnh phủ ≥ 80% trang
                        return True
                finally:
                    if pix is not None:
                        del pix
        except Exception as e:
            logger.debug(f"Watermark image check error: {e}")

        # 2 - text (giữ nguyên phần cũ)
        text = page.get_text("text") or ""
        for pattern in self.cfg.watermark_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return True

        return False

    # ===== Signature heuristics =====
    def _is_signature_block(self, block: Dict[str, Any], page_rect: fitz.Rect) -> bool:
        """Heuristic nhận diện block chữ ký"""
        if block.get("type", 0) != 0:
            return False

        block_text = self._collect_block_text(block, join_with=" ").strip()

        # Bỏ qua số trang
        if re.match(r"^\d+$", block_text) or re.match(r"^trang\s*\d+(/\d+)?$", block_text, re.IGNORECASE):
            return False

        # 1) Keyword-based
        if block_text:
            for pat in self.cfg.signature_keywords:
                if re.search(pat, block_text, flags=re.IGNORECASE):
                    return True

        # 2) Vị trí
        x0, y0, x1, y1 = block.get("bbox", [0, 0, 0, 0])
        page_h = page_rect.height
        page_w = page_rect.width
        in_bottom = (y0 >= (1.0 - self.cfg.signature_bottom_ratio) * page_h)
        in_right = (x0 >= (1.0 - self.cfg.signature_right_ratio) * page_w)

        # 3) Kích thước
        block_area = max(0, (x1 - x0) * (y1 - y0))
        page_area = max(1, page_w * page_h)
        very_small = (block_area / page_area <= self.cfg.signature_area_ratio)

        # 4) Độ ngắn gọn
        num_lines = len(block.get("lines", []))
        num_chars = len(block_text)
        compact_text = (num_chars <= 200 and num_lines <= 10)

        # 5) Có ảnh scan chữ ký trong block không
        has_image_in_block = False
        try:
            img_list = block.get("image", [])
            if img_list:  # Nếu parser cung cấp thông tin ảnh theo block
                has_image_in_block = True
        except:
            pass

        if (in_bottom or in_right) and very_small and compact_text:
            if has_image_in_block:
                return True
            else:
                return False

        return False

    # ===== PDF features & decision =====
    def _extract_text_features(self, pdf_path: str) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng text của PDF:
        - total_chars: tổng ký tự (thô)
        - filtered_chars: ký tự sau khi loại khối signature-like
        - coverage_ratio: tổng diện tích khối non-signature / tổng diện tích các trang
        - blocks_non_sign: số khối non-signature
        - like_watermark: phát hiện watermark
        - like_signature: tồn tại ít nhất 1 khối signature-like
        """
        total_chars = filtered_chars = 0
        total_non_sign_area = 0.0
        blocks_non_sign = 0
        like_watermark = like_signature = False
        total_page_area = 0.0

        # Dùng context manager để đảm bảo đóng file
        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                page_rect = page.rect
                total_page_area += max(1.0, page_rect.width * page_rect.height)

                if self._detect_watermark(page):
                    like_watermark = True

                tdict = page.get_text("dict")
                for block in tdict.get("blocks", []):
                    if block.get("type", 0) != 0:
                        continue
                    block_text = self._collect_block_text(block, join_with="")
                    if not block_text:
                        continue

                    total_chars += len(block_text)

                    if self._is_signature_block(block, page_rect):
                        like_signature = True
                        continue

                    filtered_chars += len(block_text)
                    x0, y0, x1, y1 = block.get("bbox", [0, 0, 0, 0])
                    block_area = max(0, (x1 - x0) * (y1 - y0))
                    total_non_sign_area += block_area
                    blocks_non_sign += 1

        coverage_ratio = (total_non_sign_area / total_page_area) if total_page_area > 0 else 0.0

        return {
            "total_chars": total_chars,
            "filtered_chars": filtered_chars,
            "coverage_ratio": coverage_ratio,
            "blocks_non_sign": blocks_non_sign,
            "like_watermark": like_watermark,
            "like_signature": like_signature,
        }


    def _is_pdf_text_based(self, pdf_path: str) -> Tuple[bool, bool, bool]:
        """
        Quyết định text-based với lọc chữ ký/tem số hoá.
        Trả về (is_text_based, like_watermark, like_signature).
        """
        if not pdf_path:
            return (False, False, False)
        if pdf_path in self.cache_pdf_results:
            return self.cache_pdf_results[pdf_path]

        try:
            feats = self._extract_text_features(pdf_path)
            is_text_based = (
                (feats["filtered_chars"] >= self.cfg.text_threshold) and
                (feats["coverage_ratio"] >= self.cfg.min_coverage_ratio or
                 feats["blocks_non_sign"] >= self.cfg.min_blocks_non_sign)
            )
            result = (is_text_based, feats["like_watermark"], feats["like_signature"])
            self.cache_pdf_results[pdf_path] = result
            return result
        except Exception as e:
            logger.error(f"Error checking PDF {pdf_path}: {e}", exc_info=True)
            return (False, False, False)
        
    # ===== Validation =====
    def _validate_input(self, input_data) -> bool:
        """Xác thực & chuẩn hoá đường dẫn của input.
        - Lỗi cấp entry (mất cấu trúc) → trả False.
        - Lỗi cấp document: luôn loại doc nếu path không tồn tại (kể cả khi skip_invalid_docs=False).
        """
        if not isinstance(input_data, list):
            logger.error("Input data must be a list of dictionaries.")
            return False

        required_entry_fields = ["type", "customerID", "documents"]
        required_doc_fields = ["documentID", "documentType", "documentFormat", "documentPath"]

        for entry_idx, entry in enumerate(input_data):
            if not isinstance(entry, dict):
                logger.error(f"Entry {entry_idx} is not a dictionary.")
                return False

            # Kiểm tra các field bắt buộc của entry
            for field in required_entry_fields:
                if field not in entry:
                    logger.error(f"Entry {entry_idx} missing required field: {field}")
                    return False

            # Kiểm tra type hợp lệ
            if entry.get("type") not in ["scan", "email"]:
                logger.error(f"Entry {entry_idx} has invalid type: {entry.get('type')}")
                return False

            # Kiểm tra documents là list
            if not isinstance(entry.get("documents"), list):
                logger.error(f"Entry {entry_idx} has invalid documents field: must be a list")
                return False

            # Danh sách doc hợp lệ (luôn dùng sau cùng)
            valid_docs = []

            for doc_idx, doc in enumerate(entry.get("documents", [])):
                if not isinstance(doc, dict):
                    logger.error(f"Document {doc_idx} in entry {entry_idx} is not a dictionary.")
                    if not self.cfg.skip_invalid_docs:
                        # vẫn loại bỏ doc sai kiểu, không dừng toàn bộ
                        pass
                    continue

                # Kiểm tra field bắt buộc
                missing_fields = [f for f in required_doc_fields if f not in doc]
                if missing_fields:
                    logger.error(f"Document {doc_idx} in entry {entry_idx} missing fields: {missing_fields}")
                    if not self.cfg.skip_invalid_docs:
                        # vẫn loại bỏ doc thiếu field, không dừng toàn bộ
                        pass
                    continue

                # Kiểm tra format
                doc_format = doc.get("documentFormat", "").lower()
                if doc_format not in self.cfg.supported_formats:
                    doc_path_raw = doc.get("documentPath", "")
                    logger.warning(
                        f"Skipping unsupported format in entry {entry_idx} doc {doc_idx}: {Path(doc_path_raw).name}"
                    )
                    # luôn bỏ qua doc không hỗ trợ
                    continue

                # Chuẩn hoá đường dẫn
                doc_path_raw = doc.get("documentPath")
                resolved = resolve_doc_path(doc_path_raw, self.cfg)
                doc["documentPath"] = resolved

                # Nếu file không tồn tại → luôn bỏ
                if not resolved or not os.path.exists(resolved):
                    msg = (
                        f"Document {doc_idx} in entry {entry_idx} path not found. "
                        f"Given='{doc_path_raw}' -> Resolved='{resolved}'"
                    )
                    if not self.cfg.skip_invalid_docs:
                        logger.error(msg)
                        raise ValueError(msg)
                    else:
                        logger.error(msg)
                    continue

                valid_docs.append(doc)

            # QUAN TRỌNG: luôn gán lại danh sách đã lọc (kể cả khi skip_invalid_docs=False)
            entry["documents"] = valid_docs

        return True


    # ===== ZIP handling =====
    def _process_zip_file(self, zip_path: str, customer_id: str, document_id: str, document_type: str):
        """Giải nén tệp zip và tạo danh sách tài liệu con."""
        documents = []

        zip_path = resolve_doc_path(zip_path, self.cfg)
        if not zip_path or not os.path.exists(zip_path):
            logger.error(f"Invalid or non-existent zip path: {zip_path}")
            return documents

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                extract_dir = (self.cfg.unzip_root / customer_id / document_id).resolve()
                os.makedirs(extract_dir, exist_ok=True)
                zip_ref.extractall(extract_dir)

                for file_name in zip_ref.namelist():
                    file_path = (extract_dir / file_name).resolve()
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        ext_no_dot = ext[1:] if ext.startswith(".") else ext

                        if ext_no_dot not in self.cfg.supported_formats:
                            logger.warning(f"Skipping unsupported file in zip: {file_name}")
                            continue

                        child_doc_id = f"{document_id}-{Path(file_name).stem}"

                        try:
                            if ext_no_dot == "pdf":
                                is_text_based, like_watermark, like_signature = self._is_pdf_text_based(str(file_path))
                                requires_ocr = not is_text_based
                            else:
                                is_text_based = False
                                like_watermark = False
                                like_signature = False
                                requires_ocr = ext_no_dot in ["jpg", "png", "tiff"]

                            doc_info = {
                                "documentID": child_doc_id,
                                "documentType": document_type,
                                "documentPath": str(file_path),
                                "requiresOCR": requires_ocr,
                                "metadata": {
                                    "likeWatermark": like_watermark,
                                    "likeSignature": like_signature
                                },
                            }
                            documents.append(doc_info)

                            logger.info(
                                f"[ZIP] Processed child doc | customerID={customer_id} | parentID={document_id} | "
                                f"file={file_path.name} | format={ext_no_dot} | requiresOCR={requires_ocr} | "
                                f"likeWatermark={like_watermark} | likeSignature={like_signature}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[ZIP] Error processing child file '{file_path}': {e}",
                                exc_info=True
                            )
        except zipfile.BadZipFile:
            logger.error(f"Invalid zip file: {zip_path}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing zip {zip_path}: {e}", exc_info=True)

        return documents

    # ===== Core pipeline =====
    def process(self, input_data):
        """Xử lý dữ liệu đầu vào và trả về dữ liệu đầu ra (list entries)."""
        if not self._validate_input(input_data):
            raise ValueError("Invalid input data.")

        output_data = []

        for entry_idx, entry in enumerate(input_data):
            customer_id = entry.get("customerID")
            if not customer_id:
                logger.error(f"Entry {entry_idx} has empty customerID.")
                if not self.cfg.skip_invalid_docs:
                    raise ValueError(f"Invalid customerID in entry {entry_idx}")
                continue

            transaction_id = entry.get("transactionID")
            if not transaction_id:
                transaction_id = str(uuid.uuid4())

            documents = []
            for doc_idx, doc in enumerate(entry.get("documents", [])):
                doc_id = doc.get("documentID")
                doc_type = doc.get("documentType")
                doc_format = doc.get("documentFormat", "").lower()
                doc_path = doc.get("documentPath")

                # Log bắt đầu xử lý từng document
                logger.info(
                    f"Start doc | entry={entry_idx} | idx={doc_idx} | customerID={customer_id} | "
                    f"docID={doc_id} | type={doc_type} | format={doc_format} | path='{doc_path}'"
                )

                if not all([doc_id, doc_type, doc_format, doc_path]):
                    msg = f"Document {doc_idx} in entry {entry_idx} has missing fields."
                    logger.error(msg)
                    if self.cfg.skip_invalid_docs:
                        continue
                    raise ValueError(msg)

                try:
                    if doc_format == "zip":
                        child_docs = self._process_zip_file(doc_path, customer_id, doc_id, doc_type)
                        documents.extend(child_docs)
                        logger.info(
                            f"Finish doc (ZIP) | customerID={customer_id} | docID={doc_id} | "
                            f"child_count={len(child_docs)}"
                        )
                    else:
                        if doc_format == "pdf":
                            is_text_based, like_watermark, like_signature = self._is_pdf_text_based(doc_path)
                            requires_ocr = not is_text_based
                        else:
                            is_text_based = False
                            like_watermark = False
                            like_signature = False
                            requires_ocr = doc_format in ["jpg", "png", "tiff"]

                        doc_info = {
                            "documentID": doc_id,
                            "documentType": doc_type,
                            "documentPath": doc_path,
                            "requiresOCR": requires_ocr,
                            "metadata": {
                                "likeWatermark": like_watermark,
                                "likeSignature": like_signature
                            },
                        }
                        documents.append(doc_info)

                        logger.info(
                            f"Finish doc | customerID={customer_id} | docID={doc_id} | "
                            f"requiresOCR={requires_ocr} | likeWatermark={like_watermark} | likeSignature={like_signature}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing doc | customerID={customer_id} | docID={doc_id} | error={e}",
                        exc_info=True
                    )
                    if not self.cfg.skip_invalid_docs:
                        raise

            if documents:
                output_data.append(
                    {
                        "customerID": customer_id,
                        "transactionID": transaction_id,
                        "documents": documents,
                    }
                )

        return output_data

    def process_file(self, input_path: str, output_path: str):
        """Xử lý tệp JSON đầu vào và lưu kết quả vào tệp đầu ra."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        logger.info(f"Begin processing input='{input_path}' -> output='{output_path}'")

        output_data = self.process(input_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully processed '{input_path}' to '{output_path}'")