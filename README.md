# granite-agent

A tool-using AI agent built from scratch in plain Python, running IBM's open-source Granite model locally through Ollama. No agent framework, no pip installs. I wanted to understand what frameworks like LangGraph actually do under the hood, so I wrote the loop myself.

Example run (the model's wording varies from run to run):

```
$ python agent.py "Anything wrong in the lab? Check lab-srv04 and the tape02 logs."
Q: Anything wrong in the lab? Check lab-srv04 and the tape02 logs.
  [step 1] check_disk({'host': 'lab-srv04'}) -> lab-srv04 is at 99% disk usage
  [step 1] read_logs({'machine': 'tape02'}) -> ERROR drive timeout
ERROR checksum mismatch
A: lab-srv04 is almost out of disk space (99%), and tape02 is logging drive timeouts and checksum errors.
```

## How it works

An agent is a loop around an LLM:

1. Send the conversation plus a list of tools (as JSON schemas) to the model.
2. If the model asks for a tool, run it and add the result to the conversation.
3. Repeat until the model answers in plain text, or stop at the step limit.

```
question ──► LLM ──► wants a tool? ──► run it ──► add result ──┐
              ▲                                                │
              └────────────────────────────────────────────────┘
             plain text? ──► final answer      (max_steps stops runaways)
```

All of the agent's memory is the `messages` list. Each tool is an ordinary Python function plus a schema that tells the model its name, purpose, and arguments.

## Tools

| Tool | What it does |
|---|---|
| `calculator` | Evaluates arithmetic by walking the Python AST. It never calls `eval()` on model output. |
| `get_current_time` | Returns the local date and time. |
| `search_notes` | Keyword lookup over a few notes, a stand-in for retrieval (RAG). |
| `check_disk` | Disk usage for a lab host (fake data, read-only). |
| `read_logs` | Recent log lines for a machine (fake data, read-only). |

## Guardrails

- **Step limit:** `max_steps=6`, so a confused model can't loop forever.
- **Errors go back to the model:** an unknown tool, bad JSON arguments, or a tool exception comes back as text so the model can correct itself instead of crashing the program.
- **No code execution:** the calculator only allows numbers and arithmetic operators.
- **Read-only tools:** the lab tools can look but not change anything. Anything that writes should need a human to approve it.

## Tests

The tests use a scripted fake LLM, so they check the agent's logic without depending on what a real model happens to say, and they run without Ollama.

```bash
python -m unittest test_agent.py
```

They cover a tool call followed by an answer, unknown tools, the step limit, the calculator rejecting code, bad JSON arguments, and a two-tool triage run.

## Run it

```bash
# install Ollama from https://ollama.com, then:
ollama pull granite4:micro
python step1_chat.py "What is an AI agent?"   # one plain LLM call
python agent.py "What is 17.5% of 2340, and what time is it?"
```

Use another model with `MODEL=granite3.3:8b python agent.py "..."`. Only the standard library is used.

## What I'd add next

- Real retrieval: embeddings and a vector store in place of `search_notes`
- Argument validation against each tool's schema before running it
- Logging and tracing of every step, plus an eval set to measure task success, tool-call accuracy, steps, and latency
- Exposing the tools over MCP so other agents can use them
