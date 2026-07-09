import os
from redisvl.extensions.cache.llm import LangCacheSemanticCache
from dotenv import load_dotenv

load_dotenv()

CACHE_ID = os.getenv("LANGCACHE_CACHE_ID")
API_KEY = os.getenv("LANGCACHE_API_KEY")
SERVER_URL = os.getenv("LANGCACHE_SERVER_URL")

print(f"Cache ID: {CACHE_ID}")
print(f"Server: {SERVER_URL}\n")

try:
    # Initialize
    print("1️⃣ Initializing cache...")
    cache = LangCacheSemanticCache(
        name="scalable_rag_cache",
        server_url=SERVER_URL,
        cache_id=CACHE_ID,
        api_key=API_KEY,
        ttl=3600,
    )
    print("✅ Cache initialized\n")

    # Store
    print("2️⃣ Storing entry...")
    cache.store(
        prompt="What is Transformers?",
        response="Transformers are neural network architectures",
    )
    print("✅ Entry stored\n")

    # Check (retrieve)
    print("3️⃣ Checking cache...")
    result = cache.check(prompt="What is Transformers?")
    print(f"✅ Check result: {result}\n")

    # Search similar
    print("4️⃣ Searching similar query...")
    result2 = cache.check(prompt="Tell me about Transformers")
    print(f"✅ Similar search: {result2}\n")

except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
