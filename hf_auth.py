"""Hugging Face authentication helpers."""

from huggingface_hub import get_token


class HuggingFaceTokenError(RuntimeError):
    """Raised when no Hugging Face token can be resolved."""


def resolve_hf_token() -> str:
    """Resolve a Hugging Face token through the Hub library's normal sources."""
    token = get_token()
    if token and token.strip():
        return token.strip()
    raise HuggingFaceTokenError(
        "\nNo Hugging Face token found.\n\n"
        "Accept the Ideogram4 license at:\n"
        "    https://huggingface.co/ideogram-ai/ideogram-4-nf4\n\n"
        "Then run `huggingface-cli login`, or set HF_TOKEN / HF_TOKEN_PATH.\n"
    )
