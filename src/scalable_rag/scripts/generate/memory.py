from redis_agent_memory import AgentMemory, models
from omegaconf import DictConfig
import time
from typing import List, Dict

_agent_memory = None
_session_id = "default-session"


def initialize_memory(config: DictConfig, session_id: str = "default-session"):
    """Initialize AgentMemory connection"""
    global _agent_memory, _session_id

    api_key = config.generate.secrets.REDIS_AGENT_MEMORY_API_KEY
    server_url = config.generate.memory.SERVER_URL
    store_id = config.generate.memory.STORE_ID
    _session_id = session_id

    _agent_memory = AgentMemory(
        server_url,
        store_id=store_id,
        api_key=api_key,
    )

    return _agent_memory


def add_user_message(query: str, actor_id: str = "user-1"):
    """Add user query to session memory"""

    if not _agent_memory:
        return

    try:
        _agent_memory.add_session_event(
            session_id=_session_id,
            actor_id=actor_id,
            role=models.MessageRole.USER,
            content=[{"text": query}],
            created_at=int(time.time() * 1000),
        )
        print("✅ Added user message to memory")
    except Exception as e:
        print(f"Failed to add user message: {e}")


def add_assistant_message(answer: str, actor_id: str = "assistant-1"):
    """Add assistant response to session memory"""

    if not _agent_memory:
        return

    try:
        _agent_memory.add_session_event(
            session_id=_session_id,
            actor_id=actor_id,
            role=models.MessageRole.ASSISTANT,
            content=[{"text": answer}],
            created_at=int(time.time() * 1000),
        )
        print("✅ Added assistant message to memory")
    except Exception as e:
        print(f"Failed to add assistant message: {e}")


def get_session_history() -> List[Dict]:
    """Retrieve conversation history for context"""

    if not _agent_memory:
        return []

    try:
        session = _agent_memory.get_session_memory(session_id=_session_id)

        # Handle the actual response structure
        messages = []

        # The response object might be an object, not a dict
        if hasattr(session, "events"):
            events = session.events
        else:
            events = session.get("events", []) if isinstance(session, dict) else []

        for event in events:
            role = event.get("role") if isinstance(event, dict) else event.role
            content = event.get("content") if isinstance(event, dict) else event.content

            messages.append(
                {
                    "role": str(role).lower(),
                    "content": content[0]["text"] if content else "",
                }
            )

        return messages
    except Exception as e:
        print(f"Failed to retrieve session history: {e}")
        return []


def store_long_term_memory(memories: List[Dict]):
    """Store facts/knowledge for long-term retrieval"""

    if not _agent_memory:
        return

    try:
        _agent_memory.bulk_create_long_term_memories(memories=memories)
        print(f"💾 Stored {len(memories)} long-term memories")
    except Exception as e:
        print(f"Failed to store long-term memories: {e}")


def search_long_term_memory(query: str) -> List[Dict]:
    """Search long-term memory for relevant facts"""

    if not _agent_memory:
        return []

    try:
        results = _agent_memory.search_long_term_memory(request={"text": query})
        return results.get("results", [])
    except Exception as e:
        print(f"Failed to search long-term memory: {e}")
        return []


def clear_session():
    """Clear current session"""
    global _session_id
    _session_id = f"session-{int(time.time())}"
    print(f"🗑️ Session cleared, new session: {_session_id}")
