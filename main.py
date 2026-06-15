import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────
# Load .env from project root
# ──────────────────────────────────────────────
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-3.5-turbo"

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "❌ OPENROUTER_API_KEY environment variable is not set.\n"
        "   Set it with:  $env:OPENROUTER_API_KEY=\"sk-or-...\"  (PowerShell)\n"
        "   or add it to your .env file."
    )

# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(title="Askly Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# OpenRouter Agent
# ──────────────────────────────────────────────
async def ask_openrouter(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Sends the user prompt to OpenRouter and returns the AI response text.
    """

    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Optional metadata that helps OpenRouter route requests
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Askly, an advanced AI conversational assistant. "
                    "You provide clear, accurate, and helpful answers. "
                    "Be concise but thorough. Format code blocks with proper markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"OpenRouter API error: {response.text}",
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected response format from OpenRouter: {data}",
        )


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.post("/generate")
async def generate(data: dict):
    """
    Receives a prompt and returns a response from the OpenRouter AI agent.
    Request body: { "prompt": "your question here" }
    Response:     { "response": "AI answer here" }
    """
    prompt = data.get("prompt", "").strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    # Optional: override the model per-request
    model = data.get("model", DEFAULT_MODEL)

    ai_response = await ask_openrouter(prompt, model)

    return {"response": ai_response}


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok", "model": DEFAULT_MODEL}

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
