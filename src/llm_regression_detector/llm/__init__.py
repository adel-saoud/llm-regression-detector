"""LLM client abstractions — provider-agnostic via litellm Router with fallbacks."""

from llm_regression_detector.llm.client import LLMClient, LLMResponse, build_client
from llm_regression_detector.llm.mock import MockLLMClient

__all__ = ["LLMClient", "LLMResponse", "MockLLMClient", "build_client"]
