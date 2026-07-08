from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from omegaconf import DictConfig


def chunk_embed(config: DictConfig, chunks):
    embed_model = HuggingFaceEmbedding(
        model_name=config.ingest.embed.model, device="cpu"
    )
    Settings.embed_model = embed_model

    texts = [chunk.get_content() for chunk in chunks]
    embeddings = embed_model.get_text_embedding_batch(texts, batch_size=32)

    # Assign embeddings to chunk objects
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    print(f"Embedded {len(chunks)} nodes")
    return chunks
