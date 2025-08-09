# main.py
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

from config import ProcessorConfig
from document_processor import DocumentProcessor

def setup_logging(cfg: ProcessorConfig) -> Path:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Tạo log file theo timestamp cho mỗi lần chạy
    log_file = cfg.log_dir / f"processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger()
    logger.handlers.clear()  # bỏ mọi handler cũ
    logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    # Chỉ ghi file, không ghi ra console
    # Có thể chọn RotatingFileHandler hoặc FileHandler; ở đây dùng Rotating cho an toàn
    fh = RotatingFileHandler(
        filename=str(log_file),
        mode="a",
        maxBytes=cfg.log_rotate_megabytes * 1024 * 1024,
        backupCount=cfg.log_backup_count,
        encoding="utf-8",
        delay=False,
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Tắt propagate thừa
    logging.getLogger("document_processor").propagate = True

    return log_file

def main():
    parser = argparse.ArgumentParser(description="Process documents and determine OCR requirements.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=False, help="Path to output JSON file")
    parser.add_argument("--skip-invalid-docs", action="store_true", help="Skip invalid documents instead of raising error")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output).resolve() if args.output else (input_path.parent / "output.json")

    # Khởi tạo cấu hình và áp dụng flag CLI (nếu có)
    cfg = ProcessorConfig()
    if args.skip_invalid_docs:
        cfg.skip_invalid_docs = True

    # Thiết lập logging ra file
    log_file = setup_logging(cfg)

    processor = DocumentProcessor(cfg)
    processor.process_file(str(input_path), str(output_path))

    # In ra đường dẫn log để bạn tiện mở (chỉ in một dòng cuối cùng sau khi hoàn tất)
    # Nếu tuyệt đối không muốn in ra màn hình, hãy comment 2 dòng dưới.
    print(f"Log written to: {log_file}")

if __name__ == "__main__":
    main()