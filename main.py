"""
Caseer / Cyber Execs AI Chatbot — Backend
Version 2.0

Architecture:
  User message
    → rapidfuzz FAQ match  (fast, free, accurate)
    → if no match → AI fallback (OpenRouter)

Upgrades from v1:
  - Pydantic v2 compatible
  - rapidfuzz replaces difflib (much better fuzzy matching)
  - Conversation memory per session (last N turns sent to AI)
  - DB logging actually implemented in /chat
  - /admin/logs    → view all chat logs
  - /admin/unanswered → view AI-fallback queries (unmatched FAQs)
  - /admin/faq     → view loaded FAQ data
  - async HTTP (httpx) instead of blocking requests
  - PostgreSQL-ready (set DATABASE_URL in env)
"""

import os
import json
import time
import asyncio
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import httpx
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz, process
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Text, Float, DateTime, Table, MetaData, Boolean, select, desc
)
from sqlalchemy.orm import sessionmaker

# --------------------------------------------------
# ENV
# --------------------------------------------------
env_path = find_dotenv(".env")
load_dotenv(env_path, override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
MATCH_THRESHOLD    = float(os.getenv("MATCH_THRESHOLD", "72"))   # 0–100 scale (rapidfuzz)
MAX_HISTORY_TURNS  = int(os.getenv("MAX_HISTORY_TURNS", "6"))    # pairs of user/bot kept per session
ADMIN_SECRET       = os.getenv("ADMIN_SECRET", "changeme")       # basic admin gate
FAQ_FILE           = os.getenv("FAQ_FILE", "cyber_data.json")
BOT_NAME           = os.getenv("BOT_NAME", "Cyber Execs AI Assistant")
SYSTEM_PROMPT      = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful cybersecurity assistant for Cyber Execs Network. "
    "Answer questions clearly and professionally. "
    "If a question is outside cybersecurity, politely redirect."
)

print(f"[BOOT] Model: {OPENROUTER_MODEL}")
print(f"[BOOT] FAQ file: {FAQ_FILE}")
print(f"[BOOT] Match threshold: {MATCH_THRESHOLD}")

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(title="Cyber Execs Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_logs.db")

# Render PostgreSQL fix: replace postgres:// with postgresql://
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DB_URL else {}
engine = create_engine(DB_URL, connect_args=connect_args)
metadata_db = MetaData()

chat_logs = Table(
    "chat_logs",
    metadata_db,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(128), index=True),
    Column("client_id", String(128), index=True),
    Column("user_message", Text),
    Column("bot_reply", Text),
    Column("source", String(32)),          # "faq" or "ai"
    Column("matched_question", Text, nullable=True),
    Column("score", Float, nullable=True),
    Column("answered", Boolean, default=True),
    Column("timestamp", DateTime, default=datetime.utcnow),
)

try:
    metadata_db.create_all(engine)
    print("[DB] Tables created/verified OK")
except Exception as e:
    print(f"[DB WARNING] Could not connect at startup: {e}")
    print("[DB WARNING] App will still start. Fix DATABASE_URL to enable logging.")
SessionLocal = sessionmaker(bind=engine)

def log_to_db(
    session_id: str,
    client_id: str,
    user_message: str,
    bot_reply: str,
    source: str,
    matched_question: Optional[str] = None,
    score: Optional[float] = None,
):
    """Write a chat turn to the database."""
    answered = source == "faq"
    db = SessionLocal()
    try:
        db.execute(
            chat_logs.insert().values(
                session_id=session_id,
                client_id=client_id,
                user_message=user_message,
                bot_reply=bot_reply,
                source=source,
                matched_question=matched_question,
                score=score,
                answered=answered,
                timestamp=datetime.utcnow(),
            )
        )
        db.commit()
    except Exception as e:
        print(f"[DB ERROR] {e}")
        db.rollback()
    finally:
        db.close()

# --------------------------------------------------
# FAQ DATA
# --------------------------------------------------
FAQ_PATH = Path(FAQ_FILE)
if not FAQ_PATH.exists():
    FAQ_PATH.write_text(json.dumps({"default": []}, indent=2))

with FAQ_PATH.open("r", encoding="utf-8") as f:
    FAQ_DATA: Dict[str, List[dict]] = json.load(f)

# Pre-build question → answer maps per client_id for rapidfuzz
FAQ_INDEX: Dict[str, Dict[str, str]] = {}
for client_id, entries in FAQ_DATA.items():
    FAQ_INDEX[client_id] = {e["question"]: e["answer"] for e in entries}

print(f"[FAQ] Loaded clients: {list(FAQ_INDEX.keys())}")


def find_best_faq(client_id: str, message: str):
    """
    Use rapidfuzz to find the best FAQ match.
    Returns (answer, matched_question, score) or (None, None, 0).
    Score is 0–100.
    """
    qa_map = FAQ_INDEX.get(client_id) or FAQ_INDEX.get("default", {})
    if not qa_map:
        return None, None, 0.0

    result = process.extractOne(
        message,
        qa_map.keys(),
        scorer=fuzz.WRatio,   # WRatio handles partial + token matching well
    )

    if result is None:
        return None, None, 0.0

    matched_question, score, _ = result
    if score >= MATCH_THRESHOLD:
        return qa_map[matched_question], matched_question, score

    return None, None, score


# --------------------------------------------------
# CONVERSATION MEMORY (in-process)
# Structure: { session_id: [ {"role": ..., "content": ...}, ... ] }
# Capped at MAX_HISTORY_TURNS * 2 messages
# --------------------------------------------------
conversation_store: Dict[str, List[dict]] = defaultdict(list)


def get_history(session_id: str) -> List[dict]:
    return conversation_store[session_id]


def append_history(session_id: str, role: str, content: str):
    conversation_store[session_id].append({"role": role, "content": content})
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(conversation_store[session_id]) > max_msgs:
        conversation_store[session_id] = conversation_store[session_id][-max_msgs:]


# --------------------------------------------------
# AI FALLBACK  (OpenRouter — async)
# --------------------------------------------------
async def ai_fallback(session_id: str, user_message: str) -> str:
    if not OPENROUTER_API_KEY:
        return (
            "I don't have a specific answer for that in my knowledge base. "
            "Please contact us directly for more details."
        )

    history = get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cyberexecs.com",
        "X-Title": BOT_NAME,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# --------------------------------------------------
# PYDANTIC MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "message": "What is phishing?",
            "client_id": "default",
            "session_id": "user-abc-123"
        }
    })

    message: str
    client_id: Optional[str] = "default"
    session_id: Optional[str] = "anonymous"


class ChatResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "reply": "Phishing is a social engineering attack...",
            "source": "faq",
            "matched_question": "What is phishing?",
            "score": 95.0
        }
    })

    reply: str
    source: str                          # "faq" | "ai" | "error"
    matched_question: Optional[str] = None
    score: Optional[float] = None


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": time.time(),
        "model": OPENROUTER_MODEL,
        "faq_clients": list(FAQ_INDEX.keys()),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message    = req.message.strip()
    client_id  = req.client_id or "default"
    session_id = req.session_id or "anonymous"

    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    # 1️⃣ FAQ MATCH
    answer, matched_q, score = find_best_faq(client_id, message)
    if answer:
        asyncio.create_task(asyncio.to_thread(
            log_to_db, session_id, client_id, message,
            answer, "faq", matched_q, score
        ))
        return ChatResponse(
            reply=answer,
            source="faq",
            matched_question=matched_q,
            score=score,
        )

    # 2️⃣ AI FALLBACK
    try:
        reply = await ai_fallback(session_id, message)

        # Store in conversation memory so next turn has context
        append_history(session_id, "user", message)
        append_history(session_id, "assistant", reply)

        asyncio.create_task(asyncio.to_thread(
            log_to_db, session_id, client_id, message,
            reply, "ai", None, score
        ))

        return ChatResponse(reply=reply, source="ai", score=score)

    except Exception as e:
        print(f"[AI ERROR] {e}")
        fallback_msg = (
            "I couldn't find a specific answer right now. "
            "Please reach out to our team directly."
        )
        asyncio.create_task(asyncio.to_thread(
            log_to_db, session_id, client_id, message,
            fallback_msg, "error", None, score
        ))
        return ChatResponse(reply=fallback_msg, source="error")


# --------------------------------------------------
# ADMIN ROUTES  (require ?secret=ADMIN_SECRET header)
# --------------------------------------------------
def verify_admin(secret: str):
    if secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/admin/logs")
def admin_logs(
    secret: str = Query(...),
    limit: int = Query(50, le=500),
    source: Optional[str] = Query(None),
):
    """Return recent chat logs. Filter by source=faq|ai|error."""
    verify_admin(secret)
    db = SessionLocal()
    try:
        stmt = select(chat_logs).order_by(desc(chat_logs.c.timestamp)).limit(limit)
        if source:
            stmt = stmt.where(chat_logs.c.source == source)
        rows = db.execute(stmt).fetchall()
        return {
            "count": len(rows),
            "logs": [dict(r._mapping) for r in rows]
        }
    finally:
        db.close()


@app.get("/admin/unanswered")
def admin_unanswered(
    secret: str = Query(...),
    limit: int = Query(50, le=500),
):
    """Return all questions that went to AI (not matched by FAQ)."""
    verify_admin(secret)
    db = SessionLocal()
    try:
        stmt = (
            select(chat_logs)
            .where(chat_logs.c.source.in_(["ai", "error"]))
            .order_by(desc(chat_logs.c.timestamp))
            .limit(limit)
        )
        rows = db.execute(stmt).fetchall()
        return {
            "count": len(rows),
            "tip": "These are gaps in your FAQ. Add them to improve coverage.",
            "unanswered": [
                {
                    "user_message": r.user_message,
                    "timestamp": str(r.timestamp),
                    "session_id": r.session_id,
                }
                for r in rows
            ],
        }
    finally:
        db.close()


@app.get("/admin/faq")
def admin_faq(secret: str = Query(...)):
    """View currently loaded FAQ data."""
    verify_admin(secret)
    return {
        "clients": list(FAQ_INDEX.keys()),
        "total_questions": sum(len(v) for v in FAQ_INDEX.values()),
        "data": FAQ_DATA,
    }


@app.get("/admin/stats")
def admin_stats(secret: str = Query(...)):
    """Quick stats: total chats, FAQ hit rate, AI usage."""
    verify_admin(secret)
    db = SessionLocal()
    try:
        from sqlalchemy import func
        total = db.execute(select(func.count()).select_from(chat_logs)).scalar()
        faq_count = db.execute(
            select(func.count()).select_from(chat_logs).where(chat_logs.c.source == "faq")
        ).scalar()
        ai_count = db.execute(
            select(func.count()).select_from(chat_logs).where(chat_logs.c.source == "ai")
        ).scalar()

        return {
            "total_messages": total,
            "faq_hits": faq_count,
            "ai_fallbacks": ai_count,
            "faq_hit_rate": f"{(faq_count / total * 100):.1f}%" if total else "N/A",
        }
    finally:
        db.close()

# --------------------------------------------------
# STATIC FILES + CHAT INTERFACE
# --------------------------------------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/")
def serve_index():
    return FileResponse("chat.html")

app.mount("/static", StaticFiles(directory="."), name="static")
