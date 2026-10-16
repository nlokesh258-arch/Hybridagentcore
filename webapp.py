
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
AGENT_URL = "http://127.0.0.1:8080/invocations"

app = FastAPI(title="AI Business Process Assistant")

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)


class ChatRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        response = requests.get(
            "http://127.0.0.1:8080/",
            timeout=3,
        )
        return {
            "frontend": "running",
            "agent_http_status": response.status_code,
        }
    except requests.RequestException:
        return {
            "frontend": "running",
            "agent": "Check that AgentCore dev is running on port 8080.",
        }


@app.post("/api/chat")
def chat(request: ChatRequest):
    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Please enter a message.",
        )

    try:
        response = requests.post(
            AGENT_URL,
            json={"prompt": prompt},
            timeout=180,
        )
        response.raise_for_status()

        data = response.json()

        return {
            "reply": data.get(
                "result",
                data.get("message", str(data)),
            )
        }

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The agent took too long to respond. Please try again.",
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not contact the local AI agent. "
                "Check the AgentCore development server. "
                f"Details: {exc}"
            ),
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="The agent returned an unexpected response.",
        )
