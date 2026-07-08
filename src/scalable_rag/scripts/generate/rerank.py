from groq import Groq
from omegaconf import DictConfig
from typing import List
from dotenv import load_dotenv

load_dotenv()


def rerank_results(
    config: DictConfig, query: str, retrieved_chunks: List, top_k: int = 5
):
    """Rerank using LLM relevance scoring"""

    client = Groq(api_key=config.generate.secrets.GROQ_API_KEY)

    # Create a simple relevance scoring prompt
    chunk_list = "\n\n".join(
        [
            f"[Chunk {i + 1}]:\n{chunk.get_content()[:300]}..."
            for i, chunk in enumerate(retrieved_chunks)
        ]
    )

    message = client.chat.completions.create(
        model=config.generate.llms.model,
        messages=[
            {
                "role": "system",
                "content": "Rank the chunks by relevance to the query. Return ONLY the chunk numbers in order (1,3,2,5,4), most relevant first.",
            },
            {"role": "user", "content": f"Query: {query}\n\nChunks:\n{chunk_list}"},
        ],
        max_tokens=config.generate.llms.rerank.max_tokens,
        temperature=config.generate.llms.rerank.temperature,
    )

    # Parse ranking
    ranking = message.choices[0].message.content.strip()
    try:
        ranked_indices = [int(x.strip()) - 1 for x in ranking.split(",")]
        reranked = [
            retrieved_chunks[i] for i in ranked_indices if i < len(retrieved_chunks)
        ]
        return reranked[:top_k]
    except (ValueError, AttributeError, IndexError):
        return retrieved_chunks[:top_k]  # Fallback
