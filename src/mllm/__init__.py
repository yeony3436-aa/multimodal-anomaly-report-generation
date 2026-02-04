"""MLLM clients for MMAD evaluation."""
from .echo import EchoMLLM
from .base import BaseLLMClient, INSTRUCTION

__all__ = [
    "EchoMLLM",
    "BaseLLMClient",
    "INSTRUCTION",
]

# Lazy imports for API clients (avoid import errors if dependencies not installed)
def get_gpt4_client(*args, **kwargs):
    from .openai_client import GPT4Client
    return GPT4Client(*args, **kwargs)

def get_claude_client(*args, **kwargs):
    from .claude_client import ClaudeClient
    return ClaudeClient(*args, **kwargs)
