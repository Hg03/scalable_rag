from groq import Groq
from omegaconf import DictConfig
from typing import List
from dotenv import load_dotenv

load_dotenv()


def answer_generator(
    config: DictConfig, query: str, context_chunks: List, history: List[dict] = None
) -> str:

    if history is None:
        history = []

    client = Groq(api_key=config.generate.secrets.GROQ_API_KEY)

    # Format context from retrieved chunks
    context_text = "\n".join([chunk.get_content() for chunk in context_chunks])

    messages = [
        {
            "role": "system",
            "content": "You are a technical expert assistant. Answer the question using ONLY the provided context. If the context doesn't contain enough information, say so clearly. Keep answers concise and relevant.",
        },
        *history,
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
    ]

    response = client.chat.completions.create(
        model=config.generate.llms.answer,
        messages=messages,
        max_tokens=config.generate.llms.answer.max_tokens,
        temperature=config.generate.llms.answer.temperature,
    )

    answer = response.choices[0].message.content.strip()
    print("\nAnswer generated successfully")
    return answer
