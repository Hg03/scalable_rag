from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from omegaconf import DictConfig
from qdrant_client import QdrantClient, AsyncQdrantClient


def push_to_qdrant(config: DictConfig, embeddings):
    """Store embeddings with hybrid search (dense + sparse BM25)"""

    # Initialize both sync and async clients
    client = QdrantClient(
        url=config.ingest.secrets.QDRANT_CLUSTER_ENDPOINT,
        api_key=config.ingest.secrets.QDRANT_API_KEY,
        https=True,
    )

    aclient = AsyncQdrantClient(
        url=config.ingest.secrets.QDRANT_CLUSTER_ENDPOINT,
        api_key=config.ingest.secrets.QDRANT_API_KEY,
        https=True,
    )

    # Create vector store with hybrid indexing
    vector_store = QdrantVectorStore(
        collection_name=config.ingest.vector_store.collection_name,
        client=client,
        aclient=aclient,
        enable_hybrid=True,  # Enable hybrid search
        fastembed_sparse_model="Qdrant/bm25",  # Sparse model for BM25
        batch_size=20,  # Batch size for encoding
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Configure settings
    Settings.chunk_size = config.ingest.chunk.max_tokens
    Settings.chunk_overlap = config.ingest.chunk.overlap

    # Create index
    index = VectorStoreIndex(
        nodes=embeddings,
        storage_context=storage_context,
    )

    print(f"Stored {len(embeddings)} chunks in Qdrant with hybrid search enabled")
    print(index)
