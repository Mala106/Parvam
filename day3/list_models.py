import os
from pathlib import Path

import httpx
from google import genai

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
elif Path(".env").exists():
    env_updates = {}
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_updates[key.strip()] = value.strip().strip("'\"")

    for key, value in env_updates.items():
        os.environ.setdefault(key, value)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI API KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to your environment or install python-dotenv so the .env file can be loaded."
    )

client = genai.Client(api_key=API_KEY)

print("Available models that support generateContent:")
try:
    for model in client.models.list():
        if 'generateContent' in model.supported_actions:
            print(f"  {model.name}")
except httpx.ConnectError as exc:
    raise RuntimeError(
        "Could not connect to the Gemini API. Check your internet connection or proxy settings and try again."
    ) from exc
