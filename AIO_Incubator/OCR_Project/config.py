from dataclasses import dataclass, field
from pathlib import Path

def _here() -> Path:
    return Path(__file__).resolve().parent

@dataclass
class ProcessorConfig:
    # ====== IO / Paths ======
    base_dir: Path = field(default_factory=_here)
    docs_root: Path = field(default_factory=lambda: _here() / "docs")
    unzip_root: Path = field(default_factory=lambda: _here() / "data" / "unzipped")

    # ====== Formats / Logic ======
    supported_formats: set = field(default_factory=lambda: {"pdf", "jpg", "png", "tiff", "zip"})
    text_threshold: int = 100

    # Thêm từ khóa watermark Tiếng Việt + tiếng Anh
    watermark_patterns: list = field(
        default_factory=lambda: [
            "confidential", "draft", "sample", "watermark",
            "bản nháp", "lưu hành nội bộ", "bản sao", "không sao chép"
        ]
    )

    skip_invalid_docs: bool = False

    # ====== Signature / Digital seal heuristics ======
    signature_keywords: list = field(
        default_factory=lambda: [
            r"ký\s*bởi", r"ký\s*ngày", r"đã\s*ký", r"chữ\s*ký\s*số", r"tem\s*số",
            r"signed\s*by", r"digitally\s*signed\s*by", r"reason", r"location",
            r"certificate", r"serial\s*number", r"issuer"
        ]
    )

    signature_bottom_ratio: float = 0.35
    signature_right_ratio: float = 0.55
    signature_area_ratio: float = 0.01
    min_coverage_ratio: float = 0.02
    min_blocks_non_sign: int = 2

    # ====== Logging ======
    log_dir: Path = field(default_factory=lambda: _here() / "logs")
    log_level: str = "INFO"
    log_rotate_megabytes: int = 10
    log_backup_count: int = 5