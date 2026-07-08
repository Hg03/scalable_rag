from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from scalable_rag.scripts.services.generation import GenerationService
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI(title="Scalable RAG API", version="1.0.0")

# Store active sessions
_sessions: dict = {}


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    original_query: str
    rewritten_query: Optional[str]
    answer: str
    source: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query through the RAG pipeline"""

    try:
        # Create or retrieve session
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in _sessions:
            _sessions[session_id] = GenerationService(
                enable_cache=True, enable_memory=True, session_id=session_id
            )

        service = _sessions[session_id]
        result = service.trigger(request.query)

        return QueryResponse(
            session_id=session_id,
            original_query=result["original_query"],
            rewritten_query=result.get("rewritten_query"),
            answer=result["answer"],
            source=result["source"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: QueryRequest):
    """Chat endpoint that maintains conversation"""

    try:
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in _sessions:
            _sessions[session_id] = GenerationService(
                enable_cache=True, enable_memory=True, session_id=session_id
            )

        service = _sessions[session_id]
        result = service.trigger(request.query)

        return {
            "session_id": session_id,
            "query": request.query,
            "answer": result["answer"],
            "source": result["source"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/sessions")
async def get_sessions():
    """Get active sessions"""
    return {"active_sessions": len(_sessions), "session_ids": list(_sessions.keys())}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a session"""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
