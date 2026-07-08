from omegaconf import DictConfig
from liteparse import LiteParse
from pathlib import Path


def document_parser(configs: DictConfig):
    docs_path = configs.ingest.path.docs
    ocr_enabled = configs.ingest.parse.ocr_enabled
    output_format = configs.ingest.parse.output_format
    liteparser = LiteParse(ocr_enabled=ocr_enabled, output_format=output_format)
    files = [str(p) for p in Path(docs_path).glob("*") if p.is_file()]
    return [(file, liteparser.parse(file)) for file in files]
