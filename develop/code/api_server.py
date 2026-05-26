"""
의약품 RAG QnA — FastAPI 서버
React 프론트에서 rag_qna_multi.MedicalChatbot 과 동일한 대화 흐름 제공
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Literal

# 예/아니요, 네/아니요, (예/아니요로 답해주세요) 등 질문 내 안내 문구
_YES_NO_HINT = re.compile(r"(?:예|네)\s*/\s*아니[요](?:\s*로)?", re.I)

AnswerMode = Literal["yes_no"]

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="의약품 RAG QnA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, Any] = {}
_MedicalChatbot: type | None = None


def _get_bot_class():
    global _MedicalChatbot
    if _MedicalChatbot is None:
        from rag_qna_multi import MedicalChatbot

        _MedicalChatbot = MedicalChatbot
    return _MedicalChatbot


def _get_bot(session_id: str):
    if session_id not in _sessions:
        Bot = _get_bot_class()
        _sessions[session_id] = Bot()
    return _sessions[session_id]


def _answer_mode(bot, reply: str) -> AnswerMode | None:
    if not _YES_NO_HINT.search(reply):
        return None
    from rag_qna_multi import is_final_recommendation

    if is_final_recommendation(reply):
        return None
    if bot._pending_subject or bot.state._in_flow_a_clarify:
        return "yes_no"
    return None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    pending_subject: str | None = None
    answer_mode: AnswerMode | None = None


class SessionResponse(BaseModel):
    session_id: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    _get_bot(session_id)
    return SessionResponse(session_id=session_id)


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    session_id = body.session_id or str(uuid.uuid4())
    bot = _get_bot(session_id)

    try:
        reply = bot.chat(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pending = bot._pending_subject if bot._pending_subject else None

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        pending_subject=pending,
        answer_mode=_answer_mode(bot, reply),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
