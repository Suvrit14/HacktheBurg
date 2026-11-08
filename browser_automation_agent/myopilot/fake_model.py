"""A scripted stand-in for the Bedrock model, used with --fake.

It drives the NHS 'find a pharmacy' task with simple rules so the whole pipeline
(graph, browser, safety, interrupts, event log) can be tested on the live site
without an AWS model. It is NOT the agent - the real agent is the LLM.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _ref(page: str, pattern: str) -> int | None:
    m = re.search(r"\[(\d+)\] " + pattern, page)
    return int(m.group(1)) if m else None


class ScriptedPharmacyModel(BaseChatModel):
    postcode: str = "NW1 2DB"
    model_id: str = "scripted-fake"

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedPharmacyModel":  # noqa: ARG002
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        msg = self._decide(messages)
        chars = sum(len(str(m.content)) for m in messages)
        msg.usage_metadata = {"input_tokens": chars // 4, "output_tokens": 40, "total_tokens": chars // 4 + 40}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        done = [tc["name"] + ":" + str(tc["args"].get("explain", tc["args"].get("question", "")))
                for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        # Latest page snapshot = the last message that contains one.
        page = next((str(m.content) for m in reversed(messages)
                     if isinstance(m, ToolMessage | HumanMessage) and "URL:" in str(m.content)), "")
        url_m = re.search(r"URL: (\S+)", page)
        url = url_m.group(1) if url_m else ""

        def call(name: str, **args: Any) -> AIMessage:
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "t" + uuid.uuid4().hex[:8]}])

        cookie = _ref(page, r'button "Reject (analytics|additional) cookies"')
        if cookie and not any(d.startswith("click:Rejecting") for d in done):
            return call("click", ref=cookie, explain="Rejecting analytics cookies")
        if not any(d.startswith("ask_choice") for d in done):
            return call("ask_choice", question=f"Find pharmacies near your home postcode ({self.postcode})?",
                        options=["Yes, near home", "Somewhere else"])
        box = _ref(page, r'input\[text\] "Enter a town, city or postcode')
        if box and "/results" not in url and not any(d.startswith("type_text") for d in done):
            return call("type_text", ref=box, text=self.postcode, press_enter=True, explain="Searching near your postcode")
        details = _ref(page, r'a "View pharmacy details for [^"]+"')
        if details and "/results" in url and not any(d.startswith("click:Opening") for d in done):
            return call("click", ref=details, explain="Opening the nearest pharmacy")
        title = re.search(r"TITLE: (.+)", page)
        phone = re.search(r"tel:\+?(\d+)", page) or re.search(r"(0\d{3} ?\d{3} ?\d{4})", page)
        answer = f"Nearest pharmacy page: {title.group(1) if title else 'unknown'}."
        if phone:
            answer += f" Phone: {phone.group(1)}."
        return call("finish", answer=answer, follow_ups=["Show opening times", "Find another pharmacy", "Back to home"])
