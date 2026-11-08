"""A local, unbranded multi-page form ("Book community transport") served on localhost.

Purpose: show the full fill -> check answers -> APPROVE -> submit path, which the
SafetyGuard blocks on live sites. It is deliberately fictional (no real organisation)
and exercises every typing layer:
  L1 profile autofill  name, date of birth (day/month/year), email, phone, address
  L2 choices           wheelchair radio, assistance checkbox, date dropdown, time radio
  L3 voice             "where do you need to go and why" free-text
It also returns validation errors so the agent has to read and fix them.
"""
from __future__ import annotations

import html
import random
from datetime import date, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/mock/transport")
STORE: dict[str, str] = {}

CSS = """
body{margin:0;font:19px/1.5 Arial,Helvetica,sans-serif;color:#0b0c0c;background:#fff}
header{background:#1d3557;color:#fff;padding:14px 24px;font-weight:700;font-size:20px}
header small{display:block;font-weight:400;font-size:14px;opacity:.85}
main{max-width:720px;margin:28px auto;padding:0 20px}
h1{font-size:34px;margin:0 0 20px}
label,legend{display:block;font-weight:700;margin:0 0 6px}
.hint{color:#505a5f;margin:-2px 0 8px;font-size:17px}
.group{margin-bottom:26px}
.group--error{border-left:5px solid #d4351c;padding-left:14px}
.error-message{color:#d4351c;font-weight:700;margin:0 0 6px}
input[type=text],input[type=email],input[type=tel],select,textarea{font:inherit;padding:6px;border:2px solid #0b0c0c;width:100%;max-width:420px}
textarea{max-width:100%;min-height:110px}
.date{display:flex;gap:14px}.date div{width:90px}.date input{max-width:90px}.date .y{width:120px}.date .y input{max-width:120px}
.radio,.check{display:flex;align-items:center;gap:10px;margin:8px 0}.radio label,.check label{font-weight:400;margin:0}
input[type=radio],input[type=checkbox]{width:26px;height:26px}
button,.button{font:inherit;font-weight:700;background:#00703c;color:#fff;border:0;padding:10px 18px;box-shadow:0 2px 0 #002d18;cursor:pointer;text-decoration:none;display:inline-block}
.error-summary{border:5px solid #d4351c;padding:16px;margin-bottom:26px}.error-summary h2{margin:0 0 10px;font-size:24px}
dl{display:grid;grid-template-columns:1fr 2fr auto;gap:10px 16px;border-top:1px solid #b1b4b6;padding-top:10px}
dt{font-weight:700}dd{margin:0}
.panel{background:#00703c;color:#fff;padding:26px;text-align:center}.panel strong{font-size:30px;display:block}
footer{margin-top:60px;padding:20px 24px;color:#505a5f;font-size:15px;border-top:1px solid #b1b4b6}
"""


def page(title: str, body: str, errors: dict[str, str] | None = None) -> HTMLResponse:
    summary = ""
    if errors:
        items = "".join(f"<li>{html.escape(m)}</li>" for m in errors.values())
        summary = f'<div class="error-summary" role="alert"><h2>There is a problem</h2><ul>{items}</ul></div>'
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)} – Community transport (demo)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>{CSS}</style></head>
<body><header>Community transport booking<small>MyoPilot demo service – not a real service</small></header>
<main id="main-content">{summary}<h1>{html.escape(title)}</h1>{body}</main>
<footer>Demo form for the MyoPilot hackathon project. No data leaves this computer.</footer></body></html>"""
    return HTMLResponse(doc)


def text_field(name: str, label: str, errors: dict, kind: str = "text", hint: str = "", autocomplete: str = "") -> str:
    err = errors.get(name, "")
    v = html.escape(STORE.get(name, ""))
    hint_html = f'<div class="hint" id="{name}-hint">{html.escape(hint)}</div>' if hint else ""
    err_html = f'<p class="error-message" id="{name}-error">{html.escape(err)}</p>' if err else ""
    described = " ".join(x for x in (f"{name}-hint" if hint else "", f"{name}-error" if err else "") if x)
    ac = f' autocomplete="{autocomplete}"' if autocomplete else ""
    return (f'<div class="group{" group--error" if err else ""}"><label for="{name}">{html.escape(label)}</label>{hint_html}{err_html}'
            f'<input type="{kind}" id="{name}" name="{name}" value="{v}"{ac}'
            f'{f" aria-describedby=\"{described}\"" if described else ""}></div>')


def form(action: str, inner: str, button: str = "Continue") -> str:
    return f'<form method="post" action="{action}" novalidate>{inner}<button type="submit">{button}</button></form>'


def require(data: dict, rules: dict[str, str]) -> dict[str, str]:
    return {k: msg for k, msg in rules.items() if not str(data.get(k, "")).strip()}


@router.get("", response_class=HTMLResponse)
async def start() -> HTMLResponse:
    STORE.clear()
    return page("Book community transport", """
<p>Use this service to book a door-to-door accessible minibus for a medical appointment or essential trip.</p>
<p>It takes around 5 minutes. You will need your address and details of your journey.</p>
<a class="button" role="button" href="/mock/transport/about-you">Start now</a>""")


# ---------------- about you ----------------
@router.get("/about-you", response_class=HTMLResponse)
async def about_you(errors: dict | None = None) -> HTMLResponse:
    errors = errors or {}
    dob_err = errors.get("dob", "")
    dob = f"""<div class="group{' group--error' if dob_err else ''}"><fieldset style="border:0;padding:0;margin:0" role="group" aria-describedby="dob-hint">
<legend>Date of birth</legend><div class="hint" id="dob-hint">For example, 27 3 1985</div>
{f'<p class="error-message">{html.escape(dob_err)}</p>' if dob_err else ''}
<div class="date"><div><label for="dob-day">Day</label><input type="text" inputmode="numeric" id="dob-day" name="dob-day" value="{html.escape(STORE.get('dob-day',''))}" autocomplete="bday-day"></div>
<div><label for="dob-month">Month</label><input type="text" inputmode="numeric" id="dob-month" name="dob-month" value="{html.escape(STORE.get('dob-month',''))}" autocomplete="bday-month"></div>
<div class="y"><label for="dob-year">Year</label><input type="text" inputmode="numeric" id="dob-year" name="dob-year" value="{html.escape(STORE.get('dob-year',''))}" autocomplete="bday-year"></div></div></fieldset></div>"""
    inner = (text_field("full-name", "Full name", errors, autocomplete="name") + dob
             + text_field("email", "Email address", errors, "email", "We will send your booking confirmation here", "email")
             + text_field("phone", "Mobile phone number", errors, "tel", "The driver will call this number when they arrive", "tel"))
    return page("About you", form("/mock/transport/about-you", inner), errors)


@router.post("/about-you")
async def about_you_post(request: Request):
    data = dict(await request.form())
    STORE.update({k: str(v) for k, v in data.items()})
    errors = require(data, {"full-name": "Enter your full name", "email": "Enter your email address",
                            "phone": "Enter your mobile phone number"})
    try:
        date(int(data.get("dob-year", "")), int(data.get("dob-month", "")), int(data.get("dob-day", "")))
    except ValueError:
        errors["dob"] = "Date of birth must be a real date"
    if data.get("email") and "@" not in str(data["email"]):
        errors["email"] = "Enter an email address in the correct format, like name@example.com"
    if errors:
        return await about_you(errors)
    return RedirectResponse("/mock/transport/address", status_code=303)


# ---------------- address ----------------
@router.get("/address", response_class=HTMLResponse)
async def address(errors: dict | None = None) -> HTMLResponse:
    errors = errors or {}
    inner = (text_field("address-line1", "Address line 1", errors, autocomplete="address-line1")
             + text_field("address-line2", "Address line 2 (optional)", errors, autocomplete="address-line2")
             + text_field("town", "Town or city", errors, autocomplete="address-level2")
             + text_field("postcode", "Postcode", errors, autocomplete="postal-code"))
    return page("Where should we pick you up?", form("/mock/transport/address", inner), errors)


@router.post("/address")
async def address_post(request: Request):
    data = dict(await request.form())
    STORE.update({k: str(v) for k, v in data.items()})
    errors = require(data, {"address-line1": "Enter address line 1", "town": "Enter a town or city",
                            "postcode": "Enter a postcode"})
    if errors:
        return await address(errors)
    return RedirectResponse("/mock/transport/needs", status_code=303)


# ---------------- access needs ----------------
@router.get("/needs", response_class=HTMLResponse)
async def needs(errors: dict | None = None) -> HTMLResponse:
    errors = errors or {}
    sel = STORE.get("wheelchair", "")
    radios = "".join(
        f'<div class="radio"><input type="radio" id="wc-{i}" name="wheelchair" value="{v}"{" checked" if sel == v else ""}>'
        f'<label for="wc-{i}">{v}</label></div>' for i, v in enumerate(["Yes", "No", "Sometimes"]))
    err = errors.get("wheelchair", "")
    help_checked = " checked" if STORE.get("assistance") else ""
    inner = f"""<div class="group{' group--error' if err else ''}"><fieldset style="border:0;padding:0;margin:0"><legend>Do you use a wheelchair?</legend>
{f'<p class="error-message">{html.escape(err)}</p>' if err else ''}{radios}</fieldset></div>
<div class="group"><fieldset style="border:0;padding:0;margin:0"><legend>Help you need</legend>
<div class="check"><input type="checkbox" id="assistance" name="assistance" value="yes"{help_checked}>
<label for="assistance">I need help getting in and out of the vehicle</label></div></fieldset></div>"""
    return page("Your access needs", form("/mock/transport/needs", inner), errors)


@router.post("/needs")
async def needs_post(request: Request):
    data = dict(await request.form())
    STORE["wheelchair"] = str(data.get("wheelchair", ""))
    STORE["assistance"] = "yes" if data.get("assistance") else ""
    errors = require(data, {"wheelchair": "Select if you use a wheelchair"})
    if errors:
        return await needs(errors)
    return RedirectResponse("/mock/transport/journey", status_code=303)


# ---------------- journey ----------------
def _days() -> list[str]:
    today = date.today()
    return [(today + timedelta(days=i)).strftime("%A %d %B") for i in range(1, 8)]


@router.get("/journey", response_class=HTMLResponse)
async def journey(errors: dict | None = None) -> HTMLResponse:
    errors = errors or {}
    day_sel = STORE.get("day", "")
    opts = '<option value="">Choose a day</option>' + "".join(
        f'<option{" selected" if d == day_sel else ""}>{d}</option>' for d in _days())
    t_sel = STORE.get("time", "")
    times = "".join(
        f'<div class="radio"><input type="radio" id="t-{i}" name="time" value="{v}"{" checked" if t_sel == v else ""}>'
        f'<label for="t-{i}">{v}</label></div>' for i, v in enumerate(["Morning (9am to 12pm)", "Afternoon (12pm to 5pm)"]))
    reason_err = errors.get("reason", "")
    inner = f"""<div class="group{' group--error' if errors.get('day') else ''}"><label for="day">Which day do you need to travel?</label>
{f'<p class="error-message">{html.escape(errors["day"])}</p>' if errors.get('day') else ''}<select id="day" name="day">{opts}</select></div>
<div class="group{' group--error' if errors.get('time') else ''}"><fieldset style="border:0;padding:0;margin:0"><legend>What time of day?</legend>
{f'<p class="error-message">{html.escape(errors["time"])}</p>' if errors.get('time') else ''}{times}</fieldset></div>
<div class="group{' group--error' if reason_err else ''}"><label for="reason">Where do you need to go, and why?</label>
<div class="hint" id="reason-hint">For example, the name and address of the clinic and what the appointment is for</div>
{f'<p class="error-message" id="reason-error">{html.escape(reason_err)}</p>' if reason_err else ''}
<textarea id="reason" name="reason" aria-describedby="reason-hint{' reason-error' if reason_err else ''}">{html.escape(STORE.get('reason',''))}</textarea></div>"""
    return page("Your journey", form("/mock/transport/journey", inner), errors)


@router.post("/journey")
async def journey_post(request: Request):
    data = dict(await request.form())
    STORE.update({k: str(v) for k, v in data.items()})
    errors = require(data, {"day": "Choose the day you need to travel", "time": "Select a time of day",
                            "reason": "Tell us where you need to go"})
    if errors:
        return await journey(errors)
    return RedirectResponse("/mock/transport/check", status_code=303)


# ---------------- check answers + submit ----------------
@router.get("/check", response_class=HTMLResponse)
async def check() -> HTMLResponse:
    rows = [
        ("Name", STORE.get("full-name"), "about-you"),
        ("Date of birth", f"{STORE.get('dob-day')}/{STORE.get('dob-month')}/{STORE.get('dob-year')}", "about-you"),
        ("Email", STORE.get("email"), "about-you"), ("Phone", STORE.get("phone"), "about-you"),
        ("Pick-up address", ", ".join(x for x in (STORE.get("address-line1"), STORE.get("address-line2"),
                                                 STORE.get("town"), STORE.get("postcode")) if x), "address"),
        ("Wheelchair user", STORE.get("wheelchair"), "needs"),
        ("Help getting in and out", "Yes" if STORE.get("assistance") else "No", "needs"),
        ("Day", STORE.get("day"), "journey"), ("Time", STORE.get("time"), "journey"),
        ("Where and why", STORE.get("reason"), "journey"),
    ]
    dl = "".join(f"<dt>{k}</dt><dd>{html.escape(v or '')}</dd><dd><a href='/mock/transport/{p}'>Change<span style='position:absolute;left:-9999px'> {k.lower()}</span></a></dd>"
                 for k, v, p in rows)
    body = f"<dl>{dl}</dl><p>By submitting you confirm the details are correct.</p>" + form("/mock/transport/submit", "", "Submit booking")
    return page("Check your answers before booking", body)


@router.post("/submit", response_class=HTMLResponse)
async def submit() -> HTMLResponse:
    ref = f"CT-{random.randint(100000, 999999)}"
    STORE["reference"] = ref
    return page("Booking complete", f"""<div class="panel"><strong>Booking complete</strong>Your reference is <b>{ref}</b></div>
<p>We have sent a confirmation to {html.escape(STORE.get('email', 'your email'))}. The driver will call {html.escape(STORE.get('phone', 'you'))} on arrival.</p>""")
