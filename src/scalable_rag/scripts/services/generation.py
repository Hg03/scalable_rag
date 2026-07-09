from scalable_rag.scripts.generate.retrieve import retriever
from scalable_rag.scripts.generate.rewrite import query_rewriter
from scalable_rag.scripts.generate.rerank import rerank_results
from scalable_rag.scripts.generate.ask import answer_generator
from scalable_rag.scripts.generate.cache import (
    initialize_cache,
    get_cached_answer,
    cache_answer,
)
from scalable_rag.scripts.generate import memory
from hydra import initialize, compose
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()


class GenerationService:
    def __init__(
        self,
        enable_cache: bool = True,
        enable_memory: bool = True,
        session_id: str = "default-session",
    ):
        self.config: DictConfig = self.initialize_configs()
        self.enable_cache = enable_cache
        self.enable_memory = enable_memory

        if self.enable_cache:
            self.cache = initialize_cache(self.config)
        else:
            self.cache = None

        if self.enable_memory:
            memory.initialize_memory(self.config, session_id=session_id)

    def initialize_configs(self) -> DictConfig:
        with initialize(
            config_path="../../conf", job_name="generation", version_base="1.1"
        ):
            return compose(config_name="config")

    def trigger(self, query: str) -> dict:
        """RAG pipeline with semantic cache and conversation memory"""

        print(f"\n{'=' * 60}")
        print(f"Question: {query}")
        print(f"{'=' * 60}")

        # Step 0: Check semantic cache
        if self.enable_cache and self.cache:
            cached_answer = get_cached_answer(self.cache, query)
            if cached_answer:
                if self.enable_memory:
                    memory.add_user_message(query)
                    memory.add_assistant_message(cached_answer)
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

        # Step 4: Get conversation history
        history = []
        if self.enable_memory:
            history = memory.get_session_history()

        # Step 5: Generate answer
        print("\n✨ Generating answer...")
        answer = answer_generator(
            self.config, rewritten_query, reranked_chunks, history=history
        )

        # Step 6: Store in cache and memory
        if self.enable_cache and self.cache:
            cache_answer(self.cache, query, answer)

        if self.enable_memory:
            memory.add_user_message(query)
            memory.add_assistant_message(answer)

        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "source": "retrieval",
        }

    def start_conversation(self):
        """Interactive multi-turn conversation"""
        print(
            "🤖 RAG Chatbot Started (type 'exit' to quit, 'history' to see conversation)"
        )

        while True:
            query = input("\nYou: ").strip()

            if query.lower() == "exit":
                print("Goodbye!")
                break
            elif query.lower() == "history":
                hist = memory.get_session_history()
                for msg in hist[-4:]:
                    print("{msg['role'].upper()}: {msg['content'][:100]}...")
                continue
            elif not query:
                continue

            result = self.trigger(query)
            print(f"\nAssistant: {result['answer']}")
            print(f"[Source: {result['source']}]")


if __name__ == "__main__":
    service = GenerationService(
        enable_cache=True, enable_memory=True, session_id="user-1"
    )

    # First query
    result1 = service.trigger("What is Transformers?")
    print(f"Answer 1: {result1['answer']}\nSource: {result1['source']}")

    # Second query (same topic) - should hit cache
    result2 = service.trigger("What is Transformers?")
    print(f"Answer 2: {result2['answer']}\nSource: {result2['source']}")

    # Third query (different) - uses conversation context
    result3 = service.trigger("Tell me more about attention mechanism")
    print(f"Answer 3: {result3['answer']}\nSource: {result3['source']}")
