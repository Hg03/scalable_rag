from groq import Groq
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()


def query_rewriter(config: DictConfig, query: str) -> str:
    """Rewrite query using Groq for optimization"""

    client = Groq(api_key=config.generate.secrets.GROQ_API_KEY)

    message = client.chat.completions.create(
        model=config.generate.llms.model,  # Fast and free
        messages=[
            {
                "role": "system",
                "content": """You are an expert search engine optimization agent. Rewrite the user's query to optimize it for vector database retrieval.

Rules:
1. Strip conversational filler words (e.g., "please", "can you tell me").
2. Fix typos or grammatical mistakes.
3. Output ONLY the optimized search string.""",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=config.generate.llms.rewriter.max_tokens,
        temperature=config.generate.llms.rewriter.temperature,
    )

    rewritten = message.choices[0].message.content.strip()
    print(f"Original: {query}\nRewritten: {rewritten}")
    return rewritten
