"""Tests that run without Ollama, using a scripted fake LLM.

Run:  python -m unittest test_agent.py
"""
import unittest

from agent import calculator, check_disk, read_logs, run_agent, run_tool


def fake_llm(script):
    """Return an llm() that replays canned replies and records what it was sent."""
    seen = []

    def llm(messages, tools=None):
        seen.append(list(messages))
        return script[len(seen) - 1]
    llm.seen = seen
    return llm


class AgentTests(unittest.TestCase):
    def test_tool_then_answer(self):
        llm = fake_llm([
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "calculator", "arguments": {"expression": "2340*0.175"}}}]},
            {"role": "assistant", "content": "It's 409.5."},
        ])
        self.assertEqual(run_agent("17.5% of 2340?", llm=llm, verbose=False), "It's 409.5.")
        tool_msg = llm.seen[1][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["content"], "409.5")

    def test_unknown_tool_reported_to_model(self):
        llm = fake_llm([
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "hack_mainframe", "arguments": {}}}]},
            {"role": "assistant", "content": "Sorry, I can't."},
        ])
        run_agent("x", llm=llm, verbose=False)
        self.assertIn("unknown tool", llm.seen[1][-1]["content"])

    def test_step_limit(self):
        loop = {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_current_time", "arguments": {}}}]}
        llm = fake_llm([loop] * 3)
        self.assertIn("step limit", run_agent("x", llm=llm, max_steps=3, verbose=False))

    def test_calculator_rejects_code(self):
        self.assertEqual(calculator("2**10"), "1024")
        with self.assertRaises(ValueError):
            calculator("__import__('os').system('ls')")

    def test_bad_json_args_dont_crash(self):
        self.assertIn("Error", run_tool("calculator", "{not json"))

    def test_lab_tools(self):
        self.assertEqual(check_disk("lab-srv04"), "lab-srv04 is at 99% disk usage")
        self.assertIn("No such host", check_disk("nope"))
        self.assertIn("checksum mismatch", read_logs("tape02"))
        self.assertEqual(read_logs("tape99"), "No logs found.")

    def test_triage_two_tools(self):
        llm = fake_llm([
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "check_disk", "arguments": {"host": "lab-srv04"}}},
                {"function": {"name": "read_logs", "arguments": '{"machine": "tape02"}'}}]},
            {"role": "assistant", "content": "lab-srv04 is nearly full and tape02 has drive errors."},
        ])
        answer = run_agent("Anything wrong in the lab?", llm=llm, verbose=False)
        self.assertIn("nearly full", answer)
        results = [m["content"] for m in llm.seen[1] if m["role"] == "tool"]
        self.assertEqual(results[0], "lab-srv04 is at 99% disk usage")
        self.assertIn("drive timeout", results[1])


if __name__ == "__main__":
    unittest.main()
