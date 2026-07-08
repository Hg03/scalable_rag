from scalable_rag.scripts.generate.retrieve import retriever
from scalable_rag.scripts.generate.rewrite import query_rewriter
from scalable_rag.scripts.generate.ask import answer_generator
from scalable_rag.scripts.generate.rerank import rerank_results
from hydra import compose, initialize
from omegaconf import DictConfig


class GenerationService:
    def __init__(self, query: str):
        self.config: DictConfig = self.initialize_configs()
        self.query = query

    def initialize_configs(self) -> DictConfig:
        with initialize(config_path="../../conf", job_name="generation"):
            return compose(config_name="config")

    def trigger(self):
        rewritten_query = query_rewriter(self.config, self.query)
        retrieved_results = retriever(config=self.config, query=rewritten_query)
        reranked_context = rerank_results(
            config=self.config,
            query=rewritten_query,
            retrieved_chunks=retrieved_results,
        )
        return answer_generator(
            config=self.config, query=rewritten_query, context_chunks=reranked_context
        )


if __name__ == "__main__":
    GenerationService(query="Share me some Transformers Learning Resources ?").trigger()
