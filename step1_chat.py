"""Step 1: talk to Granite through Ollama. No agent yet, just one LLM call.

Run:  python step1_chat.py "What is IBM Granite?"
"""
import json
import os
import sys
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "granite4:micro")


def chat(messages, tools=None):
    """Send the conversation to Ollama and return the assistant's message dict."""
    body = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "In one sentence, what is an AI agent?"
    reply = chat([{"role": "user", "content": question}])
    print(reply["content"])
