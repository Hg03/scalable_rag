from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from qdrant_client import QdrantClient
from omegaconf import DictConfig


def retriever(config: DictConfig, query: str):
    """Retrieve from existing Qdrant index with hybrid search"""
    embed_model = HuggingFaceEmbedding(
        model_name=config.ingest.embed.model, device="cpu"
    )

    top_k = config.generate.retrieve.top_k
    sparse_top_k = config.generate.retrieve.sparse_top_k
    # Connect to Qdrant
    client = QdrantClient(
        url=config.ingest.secrets.QDRANT_CLUSTER_ENDPOINT,
        api_key=config.ingest.secrets.QDRANT_API_KEY,
        https=True,
    )

    # Connect to existing vector store
    vector_store = QdrantVectorStore(
        collection_name=config.ingest.vector_store.collection_name,
        client=client,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )

    # Load existing index (don't create new one)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    # Hybrid retrieval with sparse + dense
    query_engine = index.as_retriever(
        similarity_top_k=top_k,
        sparse_top_k=sparse_top_k,
        vector_store_query_mode="hybrid",
    )

    # Get nodes
    retrieval_result = query_engine.retrieve(query)

    print(f"Retrieved {len(retrieval_result)} nodes")
    return retrieval_result
