# Cyber Execs AI Chatbot — v2.0

FAQ-first hybrid chatbot. Uses fuzzy matching to answer from your knowledge base first.
Only calls AI when no match is found. Logs everything to a database.

```
User message
  → rapidfuzz FAQ match  (fast, free, business-accurate)
  → fallback → AI (OpenRouter)  (only when needed = low cost)
```

---

## Project Structure

```
chatbot/
├── main.py            Backend API (FastAPI)
├── requirements.txt   Dependencies
├── render.yaml        Render.com deployment config
├── .env.example       Environment variable template
├── cyber_data.json    Cybersecurity FAQ knowledge base
├── faq_data.json      Generic business FAQ (backup)
├── index.html         Embeddable widget page
├── style.css          Widget styles
├── script.js          Widget logic
└── chat.html          Full standalone chat interface
```

---

## Local Setup

```bash
# 1. Clone / download the project
cd chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your OPENROUTER_API_KEY and ADMIN_SECRET

# 5. Run
uvicorn main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## Deploy to Render (Free Tier)

1. Push project to a **GitHub** repo (private is fine)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Set these environment variables manually in the Render dashboard:
   - `OPENROUTER_API_KEY` → your key
   - `ADMIN_SECRET` → a strong secret password
6. Click **Deploy**

> ⚠️ **SQLite warning**: Render free tier has ephemeral disk storage.
> Chat logs saved to SQLite will be lost on every redeploy.
> **Solution**: Add a free PostgreSQL database on Render and set `DATABASE_URL`.
> The app automatically detects PostgreSQL vs SQLite.

---

## API Reference

### `POST /chat`
Send a message and get a reply.

**Request:**
```json
{
  "message": "What is phishing?",
  "client_id": "default",
  "session_id": "user-abc-123"
}
```

**Response:**
```json
{
  "reply": "Phishing is a form of social engineering...",
  "source": "faq",
  "matched_question": "What is phishing?",
  "score": 97.0
}
```

- `source`: `faq` (matched from knowledge base) | `ai` (AI fallback) | `error`
- `score`: 0–100 match confidence (rapidfuzz)

### `GET /health`
Check server status and loaded FAQ clients.

### `GET /admin/logs?secret=YOUR_SECRET`
View all chat logs. Add `&source=ai` to filter AI fallbacks only.

### `GET /admin/unanswered?secret=YOUR_SECRET`
View questions that didn't match any FAQ — use this to grow your knowledge base.

### `GET /admin/stats?secret=YOUR_SECRET`
Quick stats: total messages, FAQ hit rate, AI usage count.

### `GET /admin/faq?secret=YOUR_SECRET`
View currently loaded FAQ data.

---

## Embed Widget on Any Website

Add these 3 things to any HTML page:

```html
<!-- In <head> -->
<link rel="stylesheet" href="https://your-app.onrender.com/style.css" />

<!-- In <body> -->
<div id="chat-button">💬</div>
<div id="chat-popup" class="hidden">
  <div id="chat-header">
    <div class="header-dot"></div>
    <div class="header-title">Your Bot Name</div>
    <div class="header-sub">● Online</div>
  </div>
  <div id="chat-box"></div>
  <div id="chat-input-container">
    <input id="user-input" type="text" placeholder="Ask me anything…" />
    <button id="send-btn">Send</button>
  </div>
</div>

<!-- Before </body> -->
<script>
  window.CHATBOT_API_URL = "https://your-app.onrender.com/chat";
  window.CHATBOT_CLIENT_ID = "your_client_id";
</script>
<script src="https://your-app.onrender.com/script.js"></script>
```

---

## Adding FAQs for a New Client

Edit `cyber_data.json` and add a new key:

```json
{
  "default": [...],
  "acme_corp": [
    {
      "question": "What is your refund policy?",
      "answer": "We offer full refunds within 14 days of purchase."
    }
  ]
}
```

Then send requests with `"client_id": "acme_corp"` to use that client's FAQ.
The system falls back to `"default"` if the client ID isn't found.

---

## Tuning the Matching Threshold

`MATCH_THRESHOLD` (default: 72) controls how strict FAQ matching is.

| Value | Behavior |
|-------|----------|
| 60–65 | Lenient — matches more questions, higher risk of wrong answers |
| 70–75 | **Recommended** — good balance |
| 80–85 | Strict — fewer false matches, more AI fallbacks |

---

## Roadmap (Level 3)

- [ ] Admin web dashboard (React)
- [ ] FAQ upload via API endpoint
- [ ] Semantic search with sentence embeddings (faster, smarter matching)
- [ ] Multi-tenant: each client gets their own FAQ namespace
- [ ] WhatsApp / Telegram integration via webhook
- [ ] Analytics charts: top questions, unanswered trends
