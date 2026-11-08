"""The MyoPilot task agent as a LangGraph state machine.

    START -> setup -> agent --(tool calls)--> act --(ask/confirm pending)--> human -> agent
                        |                      |--(finish / max steps)--> END
                        |--(plain text reply)--> END                       '--> agent

* `act` runs browser tools through the SafetyGuard and returns the fresh page.
* `human` pauses the graph with `interrupt()` until the user answers with gestures
  (or voice for free text). The driver resumes it with `Command(resume=...)`.
  The node has no side effects before `interrupt()`, so LangGraph's re-run on
  resume is safe.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Annotated, Any, Awaitable, Callable, TypedDict
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from .browser import Browser, BrowserError
from .config import Settings
from .events import EventLog
from .profile import flatten_values, profile_for_prompt
from .provenance import AskFirst, ProvenanceGuard
from .safety import SafetyBlock, SafetyGuard
from .tools import ALL_TOOLS, BROWSER_TOOLS, HUMAN_TOOLS

SNAPSHOT_ELEMENTS = 150

SYSTEM_PROMPT = """You are MyoPilot, a web assistant acting for a person with limited hand movement.
They control you with four gestures (scroll_down, scroll_up, forward, back) and cannot type. You operate a real
web browser for them on live websites and complete everyday tasks end to end.

HOW YOU SEE THE PAGE
After every action you receive the page as: URL, headings, a numbered list of interactive
ELEMENTS like `[12] input[text] "Postcode" required`, and the PAGE TEXT. Act on elements
by their [ref] number. Refs change after every page update: always use refs from the latest page.
PAGE TEXT already contains the whole page (not just what is on screen), and you can click
"(off-screen)" elements directly - so do not scroll just to read or reach something.

RULES
0. Always act through the tool-calling interface. Never write a tool call as text or XML.
1. One browser action per turn. Give every action a short `explain` caption in plain English
   (it is shown to the user, e.g. "Typing your postcode").
2. Cookie banners: always choose the option that rejects analytics/additional cookies.
3. Fill forms from the USER PROFILE below whenever a field matches. NEVER invent anything:
   not personal data, and not details of the request the user has not given you (which day,
   which time, where to, why, which option). If the TASK or PROFILE does not say it, ASK.
   If a value is NOT KNOWN in the profile, prefer an "I don't know" / "skip" option; otherwise ask.
4. Asking the user costs them effort. Ask only when you must:
   - ask_choice for anything that can be offered as 2-4 short options (which result, yes/no, a time).
     For a dropdown or radio group, offer the page's own options (up to 4 most relevant).
   - ask_text only for genuinely free-form answers (it uses voice dictation), e.g. a description,
     a destination or a reason. Type the user's words exactly.
5. Never type passwords, PINs, card or bank details, and never sign in or create accounts.
   If a site needs that, stop and tell the user.
6. Before pressing any final submit/send/confirm/pay button, call confirm() with a short summary.{no_submit_rule}
7. Stay on these sites: {domains}. If a page shows an error, read it and fix the field.
8. When the task is done (or cannot be done), call finish(). Its `answer` must contain the actual
   information the user asked for (names, times, numbers, dates), copied from the page - never
   "the details have been provided". Add 2-4 useful follow-up actions. Fewer steps is better.

USER PROFILE (fictional demo persona)
{profile}
"""

NO_SUBMIT_RULE = (
    "\n   DEMO MODE: on live websites the final submit is blocked. Fill everything, stop on the"
    "\n   review/check-your-answers page, and finish() telling the user it is ready to submit."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    goal: str
    start_url: str
    steps: int
    done: bool
    answer: str
    follow_ups: list[str]
    pending_human: list[dict[str, Any]]


InterruptHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class TaskAgent:
    def __init__(self, *, browser: Browser, guard: SafetyGuard, log: EventLog, profile: dict[str, Any],
                 llm: BaseChatModel, settings: Settings, model_name: str = "", screenshots: bool = True):
        self.browser = browser
        self.guard = guard
        self.log = log
        self.profile = profile
        self.profile_values = flatten_values(profile)
        self.provenance = ProvenanceGuard(profile, enabled=settings.strict_provenance)
        self.settings = settings
        self.model_name = model_name or getattr(llm, "model_id", "") or type(llm).__name__
        self.llm = llm.bind_tools(ALL_TOOLS)
        self.screenshots = screenshots
        self.system = SYSTEM_PROMPT.format(
            no_submit_rule=NO_SUBMIT_RULE if settings.live_no_submit else "",
            domains=", ".join(settings.allowed_domains),
            profile=profile_for_prompt(profile),
        )
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("setup", self._setup)
        g.add_node("agent", self._agent)
        g.add_node("act", self._act)
        g.add_node("human", self._human)
        g.add_edge(START, "setup")
        g.add_edge("setup", "agent")
        g.add_conditional_edges("agent", self._route_agent, {"act": "act", "end": END})
        g.add_conditional_edges("act", self._route_act, {"human": "human", "agent": "agent", "end": END})
        g.add_edge("human", "agent")
        return g.compile(checkpointer=InMemorySaver())

    @staticmethod
    def _route_agent(state: AgentState) -> str:
        if state.get("done"):
            return "end"
        last = state["messages"][-1]
        return "act" if isinstance(last, AIMessage) and last.tool_calls else "end"

    @staticmethod
    def _route_act(state: AgentState) -> str:
        if state.get("pending_human"):
            return "human"
        return "end" if state.get("done") else "agent"

    # ------------------------------------------------------------------ nodes
    async def _setup(self, state: AgentState) -> dict[str, Any]:
        start_url = state.get("start_url")
        if start_url:
            self.guard.check_url(start_url)
            await self.browser.banner(None)
            await self.browser.goto(start_url)
        snap = await self.browser.snapshot()
        await self._shot("start")
        content = (f"TASK: {state['goal']}\n\nCURRENT PAGE:\n{snap.render(max_elements=SNAPSHOT_ELEMENTS)}")
        return {"messages": [HumanMessage(content=content)], "steps": 0, "done": False, "pending_human": []}

    async def _agent(self, state: AgentState) -> dict[str, Any]:
        steps = state.get("steps", 0)
        if steps >= self.settings.max_steps:
            msg = f"I stopped after {steps} steps without finishing, to avoid running on forever."
            self.log.emit("result", answer=msg, follow_ups=[], stopped=True)
            return {"done": True, "answer": msg, "follow_ups": []}

        prompt = [SystemMessage(content=self.system), *self._trim(state["messages"])]
        t0 = time.monotonic()
        resp: AIMessage = await self.llm.ainvoke(prompt)
        latency = time.monotonic() - t0
        if not resp.tool_calls:
            resp = _recover_text_tool_call(resp)
        usage = resp.usage_metadata or {}
        details = usage.get("input_token_details") or {}
        self.log.emit(
            "llm_call", model=self.model_name, latency_s=round(latency, 3),
            input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=details.get("cache_read", 0),
            tool_calls=[tc["name"] for tc in resp.tool_calls], text=_text(resp)[:300],
        )
        update: dict[str, Any] = {"messages": [resp]}
        if not resp.tool_calls:  # a plain reply is treated as the final answer
            answer = _text(resp) or "Done."
            self.log.emit("result", answer=answer, follow_ups=[])
            update.update(done=True, answer=answer, follow_ups=[])
        return update

    async def _act(self, state: AgentState) -> dict[str, Any]:
        last: AIMessage = state["messages"][-1]
        out: list[ToolMessage] = []
        pending: list[dict[str, Any]] = []
        update: dict[str, Any] = {}
        steps = state.get("steps", 0)
        for call in last.tool_calls:
            name, args, cid = call["name"], call.get("args", {}), call["id"]
            if name in BROWSER_TOOLS:
                steps += 1
                out.append(ToolMessage(content=await self._run_browser_tool(name, args, steps), tool_call_id=cid))
            elif name in HUMAN_TOOLS:
                pending.append(call)
            elif name == "finish":
                answer = str(args.get("answer", "")).strip()
                # Some models (e.g. Nova) write the real answer as text next to the call and put a
                # vague summary in `answer`. Keep whichever actually carries more information.
                spoken = _text(last)
                if spoken and len(spoken) > len(answer):
                    answer = spoken
                follow_ups = list(args.get("follow_ups") or [])[:4]
                await self.browser.banner(None)
                self.log.emit("result", answer=answer, follow_ups=follow_ups)
                out.append(ToolMessage(content="Task finished.", tool_call_id=cid))
                update.update(done=True, answer=answer, follow_ups=follow_ups)
            else:
                out.append(ToolMessage(content=f"ERROR: unknown tool {name}", tool_call_id=cid, status="error"))
        update.update(messages=out, steps=steps, pending_human=pending)
        if update.get("done") and pending:  # finish wins; answer the pending asks as skipped
            update["messages"] += [ToolMessage(content="Skipped: task finished.", tool_call_id=c["id"]) for c in pending]
            update["pending_human"] = []
        return update

    async def _human(self, state: AgentState) -> dict[str, Any]:
        # No side effects before interrupt(): this node re-runs from the top on every resume.
        out: list[ToolMessage] = []
        for call in state.get("pending_human", []):
            name, args = call["name"], call.get("args", {})
            if name == "ask_choice":
                options = [str(o) for o in (args.get("options") or [])][:4]
                reply = interrupt({"kind": "choice", "question": args.get("question", ""), "options": options})
                content = "User made the back gesture (cancelled)." if reply in (None, "back") else f"User chose: {reply}"
            elif name == "ask_text":
                reply = interrupt({"kind": "text", "question": args.get("question", "")})
                content = "User cancelled." if not reply or reply == "back" else f"User said: {reply}"
            else:  # confirm
                reply = interrupt({"kind": "confirm", "question": args.get("summary", "")})
                approved = reply is True or str(reply).upper() in ("APPROVED", "YES", "FORWARD")
                if approved:
                    self.guard.submit_approved = True
                content = "APPROVED" if approved else "REJECTED by the user. Do not submit."
            out.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {"messages": out, "pending_human": []}

    # ------------------------------------------------------------------ tools
    async def _run_browser_tool(self, name: str, args: dict[str, Any], step: int) -> str:
        explain = str(args.get("explain", "")).strip()
        b, snap = self.browser, self.browser.last_snapshot
        ref = args.get("ref")
        el = snap.element(int(ref)) if (snap and ref is not None) else None
        url_before = b.page.url if b.page else ""
        t0 = time.monotonic()
        ok, blocked, needs_input, error, note = True, False, False, "", ""
        await b.banner(explain or None)
        try:
            if name == "navigate":
                self.guard.check_url(args["url"])
                await b.goto(args["url"])
            elif name == "click":
                self.guard.check_click(el, snap)
                if el and el.get("type") in ("radio", "checkbox") and not el.get("checked"):
                    self.provenance.check_choice(el.get("name", ""), el.get("question", ""))
                await b.click(int(ref))
            elif name == "type_text":
                self.guard.check_type(el)
                self.provenance.check_value(str(args["text"]), _field(el))
                await b.type_text(int(ref), str(args["text"]), bool(args.get("press_enter", False)))
            elif name == "select_option":
                self.provenance.check_value(str(args["option"]), _field(el))
                await b.select_option(int(ref), str(args["option"]))
            elif name == "set_checkbox":
                if args.get("checked") and el:
                    self.provenance.check_choice(el.get("name", ""), el.get("question", ""))
                await b.set_checked(int(ref), bool(args["checked"]))
            elif name == "scroll":
                await b.scroll(str(args["direction"]))
            elif name == "go_back":
                await b.go_back()
            # A click may have led off the allowed sites (e.g. a maps link): come back.
            if b.page and b.page.url.startswith("http"):
                try:
                    self.guard.check_url(b.page.url)
                except SafetyBlock as exc:
                    await b.go_back()
                    raise SafetyBlock(f"That link left the allowed sites, so I went back. {exc}") from exc
        except SafetyBlock as exc:
            ok, blocked, error = False, True, str(exc)
            self.log.emit("safety_block", tool=name, args=args, reason=error, url=url_before)
        except AskFirst as exc:
            ok, needs_input, error = False, True, str(exc) + _suggest_ask(el, snap)
            self.log.emit("ask_first", tool=name, args=args, reason=error, url=url_before)
        except (BrowserError, KeyError, ValueError) as exc:
            ok, error = False, str(exc)

        new = await b.snapshot()
        # Navigation wipes the caption; re-show it so viewers (and screenshots) see what happened.
        await b.banner(("⚠ " + error[:140]) if blocked else (explain or None))
        shot = await self._shot(f"{step:02d}-{name}")
        if name == "type_text" and ok:
            note = "from_profile" if str(args.get("text", "")).strip().lower() in self.profile_values else "not_from_profile"
        self.log.emit(
            "agent_step", step=step, tool=name, args=args, explain=explain, ok=ok, error=error,
            url_before=url_before, url=new.url, page_title=new.title,
            element=(el or {}).get("name"), latency_s=round(time.monotonic() - t0, 3),
            screenshot=shot, note=note,
        )
        self.log.totals["safety_blocks"] = self.log.totals.get("safety_blocks", 0) + int(blocked)
        head = ("OK." if ok else f"BLOCKED BY SAFETY RULE: {error}" if blocked
                else f"NEEDS USER INPUT: {error}" if needs_input else f"ERROR: {error}")
        return f"{head}\n\nPAGE NOW:\n{new.render(max_elements=SNAPSHOT_ELEMENTS)}"

    # ------------------------------------------------------------------ helpers
    def _trim(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Keep only the latest page snapshots in full; older ones become one line.
        Only the prompt is trimmed - the stored state keeps everything for the log."""
        keep = self.settings.keep_full_snapshots
        snap_idx = [i for i, m in enumerate(messages)
                    if isinstance(m, (ToolMessage, HumanMessage)) and "PAGE" in str(m.content)]
        old = set(snap_idx[:-keep]) if keep else set(snap_idx)
        out: list[AnyMessage] = []
        for i, m in enumerate(messages):
            if i in old:
                text = str(m.content)
                first = text.split("\n", 1)[0]
                url = next((ln for ln in text.splitlines() if ln.startswith("URL:")), "")
                short = f"{first} [older page snapshot omitted] {url}".strip()
                m = (ToolMessage(content=short, tool_call_id=m.tool_call_id, status=m.status)
                     if isinstance(m, ToolMessage) else HumanMessage(content=short))
            out.append(m)
        return out

    async def _shot(self, label: str) -> str | None:
        return await self.browser.screenshot(label) if self.screenshots else None

    # ------------------------------------------------------------------ driver
    async def run(self, goal: str, on_interrupt: InterruptHandler, start_url: str | None = None) -> dict[str, Any]:
        """Run one task to completion, pausing for the user whenever the graph interrupts."""
        config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 400}
        before, t_start = dict(self.log.totals), time.monotonic()
        self.provenance.reset(goal)
        self.log.emit("task_start", goal=goal, start_url=start_url, model=self.model_name)
        inp: Any = {"goal": goal, "start_url": start_url or ""}
        while True:
            async for _ in self.graph.astream(inp, config, stream_mode="updates"):
                pass
            snap = await self.graph.aget_state(config)
            if snap.interrupts:
                payload = snap.interrupts[0].value
                self.log.emit("ask", **payload)
                reply = await on_interrupt(payload)
                self.log.emit("answer", kind=payload.get("kind"), reply=reply)
                self.provenance.add_answer(reply)
                inp = Command(resume=reply)
                continue
            break
        values = snap.values
        # Per-task numbers (the log's totals are cumulative across a whole session).
        delta = {k: v - before.get(k, 0) for k, v in self.log.totals.items()}
        summary = {
            "goal": goal, "answer": values.get("answer", ""), "follow_ups": values.get("follow_ups", []),
            "steps": values.get("steps", 0), "elapsed_s": round(time.monotonic() - t_start, 1), **delta,
            "total_tokens": delta.get("input_tokens", 0) + delta.get("output_tokens", 0),
            "user_asks": sum(1 for m in values.get("messages", []) if isinstance(m, AIMessage)
                             for tc in m.tool_calls if tc["name"] in HUMAN_TOOLS),
            "model": self.model_name, "final_url": self.browser.page.url if self.browser.page else "",
        }
        self.log.emit("task_end", **summary)
        return summary


_TOOL_NAMES = {t.name for t in ALL_TOOLS}
_PSEUDO_CALL = re.compile(r"<(\w+)\s+(.*?)/?>", re.S)
_ATTR = re.compile(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\[.*?\]|\d+|true|false)', re.S)


def _recover_text_tool_call(resp: AIMessage) -> AIMessage:
    """Some models (seen with Amazon Nova) occasionally write a tool call as text,
    e.g. `<finish answer="..." follow_ups=[...] />`, instead of a real tool call.
    Turn that back into a proper tool call so the graph handles it normally."""
    text = _text(resp)
    m = _PSEUDO_CALL.search(text)
    if not m or m.group(1) not in _TOOL_NAMES:
        return resp
    args: dict[str, Any] = {}
    for key, raw in _ATTR.findall(m.group(2)):
        try:
            args[key] = json.loads(raw)
        except json.JSONDecodeError:
            args[key] = raw.strip('"')
    call = {"name": m.group(1), "args": args, "id": "txt" + uuid.uuid4().hex[:10], "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call], usage_metadata=resp.usage_metadata,
                     response_metadata=resp.response_metadata)


PLACEHOLDER = re.compile(r"^(choose|select|please|--|pick)", re.I)


def _suggest_ask(el: dict[str, Any] | None, snap: Any) -> str:
    """Turn a provenance refusal into a ready-made question, so the model asks instead of retrying."""
    if not el:
        return ""
    question = el.get("question") or el.get("name") or "this field"
    if el.get("tag") == "select":
        opts = [o for o in el.get("options", []) if o and not PLACEHOLDER.search(o)][:4]
    elif el.get("type") == "radio" and snap:
        opts = [e["name"] for e in snap.elements if e.get("group") and e.get("group") == el.get("group")][:4]
    elif el.get("type") == "checkbox":
        question, opts = el.get("name") or question, ["Yes", "No"]
    else:
        return f"\nNEXT STEP: call ask_text(question={json.dumps(str(question))})."
    if not opts:
        return ""
    return f"\nNEXT STEP: call ask_choice(question={json.dumps(str(question))}, options={json.dumps(opts)})."


def _field(el: dict[str, Any] | None) -> str:
    if not el:
        return ""
    q, n = el.get("question", ""), el.get("name", "")
    return f"{q} - {n}" if q and q != n else n


def _text(msg: AIMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c.strip()
    return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text").strip()


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""
