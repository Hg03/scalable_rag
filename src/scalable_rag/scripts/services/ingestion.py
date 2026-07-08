from scalable_rag.scripts.ingest.parser import document_parser
from scalable_rag.scripts.ingest.chunk import document_chunker
from scalable_rag.scripts.ingest.storage import push_to_qdrant
from scalable_rag.scripts.ingest.embed import chunk_embed
from hydra import compose, initialize
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()


class IngestionService:
    def __init__(self):
        self.config: DictConfig = self.initialize_configs()

    def initialize_configs(self) -> DictConfig:
        with initialize(
            config_path="../../conf", job_name="ingestion", version_base="1.1"
        ):
            return compose(config_name="config")

    def trigger(self):
        parsed_files = document_parser(self.config)
        chunks = document_chunker(self.config, parsed_files)
        embeddings = chunk_embed(self.config, chunks)
        push_to_qdrant(self.config, embeddings)


# if __name__ == "__main__":
#     IngestionService().trigger()
