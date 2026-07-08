from langcache import LangCache
from omegaconf import DictConfig


def initialize_cache(config: DictConfig):
    """Initialize semantic cache with LangCache"""

    api_key = config.generate.semantic_cache.LANGCACHE_API_KEY
    server_url = config.generate.semantic_cache.LANGCACHE_SERVER_URL
    cache_id = config.generate.semantic_cache.LANGCACHE_CACHE_ID

    cache = LangCache(
        server_url=server_url,
        cache_id=cache_id,
        api_key=api_key,
    )

    return cache


def get_cached_answer(cache, query: str):
    """Search for similar query in cache"""

    try:
        search_response = cache.search(prompt=query)

        if search_response and search_response.get("responses"):
            # If similar query found
            cached_answer = search_response["responses"][0]["response"]
            print(f"✅ Cache hit for: {query}")
            return cached_answer
    except Exception as e:
        print(f"Cache search failed: {e}")

    return None


def cache_answer(cache, query: str, answer: str):
    """Store Q&A pair in semantic cache"""

    try:
        save_response = cache.set(prompt=query, response=answer)
        print(f"💾 Cached: {query[:50]}...")
        return save_response
    except Exception as e:
        print(f"Cache storage failed: {e}")
        return None
