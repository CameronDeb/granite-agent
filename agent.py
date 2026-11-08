"""Step 2: a tool-using agent, built from scratch.

The whole idea of an "agent" is this loop:
    1. Send the conversation + a list of tools to the LLM.
    2. If the LLM asks to call a tool, run it and append the result.
    3. Repeat until the LLM answers in plain text (or we hit a step limit).

Run:  python agent.py "What is 17.5% of 2340, and what time is it?"
"""
import ast
import json
import operator
import sys
from datetime import datetime

from step1_chat import chat

# ---------------------------------------------------------------- tools ---
# A tool is just a Python function + a JSON schema describing it to the LLM.

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic. Never use eval() on LLM output."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {expression}")
    return str(ev(ast.parse(expression, mode="eval").body))


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


NOTES = {
    "granite": "IBM Granite is IBM's family of open-source (Apache 2.0) LLMs for enterprise use.",
    "watsonx": "watsonx is IBM's AI platform: watsonx.ai (models), watsonx.data, watsonx.governance.",
    "beeai": "BeeAI is an open-source agent framework from IBM Research, now under the Linux Foundation.",
}


def search_notes(query: str) -> str:
    """A tiny stand-in for retrieval (RAG): keyword lookup over local notes."""
    hits = [text for key, text in NOTES.items() if key in query.lower()]
    return "\n".join(hits) or "No matching notes."


# Read-only "lab" tools over fake data, so the agent can triage machines
# without being able to change anything.
DISKS = {"lab-srv01": (450, 500), "lab-srv02": (120, 500), "lab-srv04": (990, 1000)}
LOGS = {"tape01": ["INFO mount complete"], "tape02": ["ERROR drive timeout", "ERROR checksum mismatch"]}


def check_disk(host: str) -> str:
    if host not in DISKS:
        return f"No such host: {host}"
    used, total = DISKS[host]
    return f"{host} is at {used * 100 // total}% disk usage"


def read_logs(machine: str) -> str:
    return "\n".join(LOGS.get(machine, ["No logs found."]))


TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_notes": search_notes,
    "check_disk": check_disk,
    "read_logs": read_logs,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '2340 * 0.175'.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_notes",
        "description": "Search internal notes about IBM products (granite, watsonx, beeai).",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "check_disk",
        "description": "Get disk usage for a lab host, e.g. 'lab-srv01'. Read-only.",
        "parameters": {"type": "object",
                       "properties": {"host": {"type": "string"}},
                       "required": ["host"]}}},
    {"type": "function", "function": {
        "name": "read_logs",
        "description": "Read recent log lines for a machine, e.g. 'tape02'. Read-only.",
        "parameters": {"type": "object",
                       "properties": {"machine": {"type": "string"}},
                       "required": ["machine"]}}},
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools when they help: the calculator "
    "for any math, get_current_time for dates/times, search_notes for IBM products, "
    "check_disk and read_logs for lab machines. "
    "When you have enough information, answer the user directly and concisely."
)

# ----------------------------------------------------------- the loop ---


def run_tool(name, args):
    """Run one tool call. Errors go back to the model instead of crashing."""
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'"
    try:  # inside the try, so bad JSON from the model can't crash the agent
        if isinstance(args, str):  # some models send arguments as a JSON string
            args = json.loads(args or "{}")
        return str(TOOLS[name](**(args or {})))  # str() in case a tool returns a number
    except Exception as e:
        return f"Error: {e}"


def run_agent(question, llm=chat, max_steps=6, verbose=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):
        reply = llm(messages, tools=TOOL_SCHEMAS)
        messages.append(reply)

        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:           # no tool requested: this is the final answer
            return reply.get("content", "")

        for call in tool_calls:      # act, then observe
            name = call["function"]["name"]
            args = call["function"].get("arguments", {})
            result = run_tool(name, args)
            if verbose:
                print(f"  [step {step + 1}] {name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_name": name, "content": result})

    return "Stopped: hit the step limit before reaching an answer."


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is 17.5% of 2340, and what time is it?"
    print(f"Q: {q}")
    print(f"A: {run_agent(q)}")
