"""Runtime configuration. Everything is overridable via environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent   # browser_automation_agent/ (profiles/, runs/)
REPO_ROOT = ROOT.parent                          # repo root: the single .env for the whole project
load_dotenv(REPO_ROOT / ".env")
# An empty `AWS_PROFILE=` line in .env makes boto3 look for a profile named "" and fail.
# Treat empty AWS variables as unset so the default credential chain is used.
for _var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"):
    if os.environ.get(_var, None) == "":
        del os.environ[_var]

# Domains the agent may visit. Subdomains are allowed (e.g. *.service.gov.uk).
DEFAULT_ALLOWED_DOMAINS = (
    "nhs.uk",
    "gov.uk",
    "nationalrail.co.uk",
    "bbc.co.uk",
    "wikipedia.org",
    "localhost",
    "127.0.0.1",
)


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- LLM (AWS Bedrock). Model is deliberately left empty: the team picks it. ---
    model_id: str = field(default_factory=lambda: os.getenv("MYOPILOT_MODEL_ID", "").strip())
    options_model_id: str = field(default_factory=lambda: os.getenv("MYOPILOT_OPTIONS_MODEL_ID", "").strip())
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "").strip())
    aws_profile: str = field(default_factory=lambda: os.getenv("AWS_PROFILE", "").strip())
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MYOPILOT_MAX_TOKENS", "4096")))

    # --- Browser: "chrome" drives the installed Google Chrome; "chromium" uses Playwright's bundled one ---
    browser: str = field(default_factory=lambda: os.getenv("MYOPILOT_BROWSER", "chrome").strip().lower())

    @property
    def browser_channel(self) -> str | None:
        return None if self.browser in ("", "chromium") else self.browser

    # --- EMG (Suvrit's classifier): which NinaPro DB5 subject's model and recording to use.
    # EMG_SUBJECT in the root .env is shared with the notebook (m1.ipynb). ---
    emg_subject: int = field(default_factory=lambda: int(os.getenv("EMG_SUBJECT", "1")))

    # --- Agent behaviour ---
    max_steps: int = field(default_factory=lambda: int(os.getenv("MYOPILOT_MAX_STEPS", "40")))
    keep_full_snapshots: int = 2  # older page snapshots are shortened before being sent to the LLM
    allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS
    # Only type/choose values that came from the profile, the task or the user's answers.
    strict_provenance: bool = field(default_factory=lambda: _bool("MYOPILOT_STRICT_PROVENANCE", True))
    # Demo safety: on live (non-localhost) sites, never press the final submit button.
    live_no_submit: bool = field(default_factory=lambda: _bool("MYOPILOT_LIVE_NO_SUBMIT", True))

    # --- Paths ---
    profile_path: Path = field(default_factory=lambda: ROOT / os.getenv("MYOPILOT_PROFILE", "profiles/demo_user.json"))
    runs_dir: Path = ROOT / "runs"


settings = Settings()
