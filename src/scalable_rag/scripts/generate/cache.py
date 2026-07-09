from redisvl.extensions.cache.llm import LangCacheSemanticCache
from omegaconf import DictConfig


def initialize_cache(config: DictConfig):
    """Initialize semantic cache with LangCache via RedisVL"""

    api_key = config.generate.semantic_cache.LANGCACHE_API_KEY
    server_url = config.generate.semantic_cache.LANGCACHE_SERVER_URL
    cache_id = config.generate.semantic_cache.LANGCACHE_CACHE_ID

    cache = LangCacheSemanticCache(
        name="scalable_rag_cache",
        server_url=server_url,
        cache_id=cache_id,
        api_key=api_key,
        ttl=3600,
    )

    return cache


def get_cached_answer(cache, query: str):
    """Search for similar query in cache"""

    try:
        results = cache.check(prompt=query)

        if results and len(results) > 0:
            # If similar query found
            cached_answer = results[0]["response"]
            print(f"✅ Cache hit for: {query}")
            return cached_answer
    except Exception as e:
        print(f"Cache search failed: {e}")

    return None


def cache_answer(cache, query: str, answer: str):
    """Store Q&A pair in semantic cache"""

    try:
        cache.store(prompt=query, response=answer)
        print(f"💾 Cached: {query[:50]}...")
    except Exception as e:
        print(f"Cache storage failed: {e}")
