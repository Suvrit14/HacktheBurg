"""Bedrock chat-model factory. The model id is chosen by the team (MYOPILOT_MODEL_ID)."""
from __future__ import annotations

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


class ModelNotConfigured(RuntimeError):
    pass


def get_chat_model(s: Settings, model_id: str | None = None) -> BaseChatModel:
    model_id = (model_id or s.model_id).strip()
    if not model_id:
        raise ModelNotConfigured(
            "No Bedrock model chosen yet. Set MYOPILOT_MODEL_ID in .env (e.g. a Bedrock model or "
            "inference-profile id enabled in your account), plus AWS_REGION, then re-run. "
            "Use --fake to test the pipeline without a model."
        )
    if not s.aws_region:
        raise ModelNotConfigured("Set AWS_REGION in .env (the region where the Bedrock model is enabled).")
    kwargs = dict(model_id=model_id, region_name=s.aws_region, max_tokens=s.max_tokens)
    if s.aws_profile:
        kwargs["credentials_profile_name"] = s.aws_profile
    return ChatBedrockConverse(**kwargs)
