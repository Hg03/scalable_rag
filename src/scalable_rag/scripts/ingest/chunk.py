from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import Document
from omegaconf import DictConfig
from pathlib import Path


def document_chunker(config: DictConfig, parsed_files):
    max_tokens = config.ingest.chunk.max_tokens
    overlap = config.ingest.chunk.overlap
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[
            max_tokens * 2,
            max_tokens,
        ],  # Parent: 2x tokens, Child: regular tokens
        chunk_overlap=overlap,
    )

    all_nodes = []

    for file_path, parse_result in parsed_files:
        # Extract text from all pages
        full_text = "\n".join([page.text for page in parse_result.pages])

        # Create LlamaIndex Document
        doc = Document(
            text=full_text,
            metadata={
                "source": Path(file_path).name,
                "file_path": file_path,
                "page_count": len(parse_result.pages),
            },
        )

        # Parse into hierarchical nodes
        nodes = parser.get_nodes_from_documents([doc])
        all_nodes.extend(nodes)

    print(f"Total Chunks: {len(all_nodes)}")
    return all_nodes
