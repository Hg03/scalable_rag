from scalable_rag.scripts.generate.retrieve import retriever
from scalable_rag.scripts.generate.rewrite import query_rewriter
from scalable_rag.scripts.generate.rerank import rerank_results
from scalable_rag.scripts.generate.ask import answer_generator
from scalable_rag.scripts.generate.cache import (
    initialize_cache,
    get_cached_answer,
    cache_answer,
)
from hydra import initialize, compose
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()


class GenerationService:
    def __init__(self, enable_cache: bool = True):
        self.config: DictConfig = self.initialize_configs()
        self.enable_cache = enable_cache

        if self.enable_cache:
            self.cache = initialize_cache(self.config)
        else:
            self.cache = None

    def initialize_configs(self) -> DictConfig:
        with initialize(
            config_path="../../conf", job_name="generation", version_base="1.1"
        ):
            return compose(config_name="config")

    def trigger(self, query: str) -> dict:
        """RAG pipeline with semantic cache"""

        print(f"\n{'=' * 60}")
        print(f"Question: {query}")
        print(f"{'=' * 60}")

        # Step 0: Check semantic cache
        if self.enable_cache and self.cache:
            cached_answer = get_cached_answer(self.cache, query)
            if cached_answer:
                return {
                    "original_query": query,
                    "answer": cached_answer,
                    "source": "cache",
                }

        # Step 1: Rewrite query
        print("\n📝 Rewriting query...")
        rewritten_query = query_rewriter(self.config, query)

        # Step 2: Retrieve
        print("\n🔍 Retrieving chunks...")
        retrieved_chunks = retriever(self.config, rewritten_query)

        # Step 3: Rerank
        print("\n🎯 Reranking chunks...")
        reranked_chunks = rerank_results(
            self.config, rewritten_query, retrieved_chunks, top_k=5
        )

        # Step 4: Generate answer
        print("\n✨ Generating answer...")
        answer = answer_generator(self.config, rewritten_query, reranked_chunks)

        # Step 5: Cache answer
        if self.enable_cache and self.cache:
            cache_answer(self.cache, query, answer)

        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "source": "retrieval",
        }


# if __name__ == "__main__":
#     result = GenerationService(enable_cache=True).trigger('What do you mean by Transformers?')
#     print(f"\nAnswer: {result['answer']}\nSource: {result['source']}")
