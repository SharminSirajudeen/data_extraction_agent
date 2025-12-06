"""Provider Factory - Creates LLM instances for various providers.

Supports: Groq, Together AI, OpenRouter, HuggingFace, Google, Anthropic
"""

import os
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel


class Provider(str, Enum):
    """Supported LLM providers."""

    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # For local models if resources allow


# Model configurations per provider
PROVIDER_MODELS = {
    Provider.GROQ: {
        "default": "llama-3.1-8b-instant",
        "fast": "llama-3.1-8b-instant",
        "balanced": "llama-3.3-70b-versatile",
        "reasoning": "qwen-qwq-32b",
        "extraction": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    Provider.TOGETHER: {
        "default": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "fast": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "balanced": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "extraction": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    },
    Provider.OPENROUTER: {
        "default": "google/gemini-2.0-flash-exp:free",
        "fast": "google/gemini-2.0-flash-exp:free",
        "balanced": "deepseek/deepseek-chat",
        "reasoning": "deepseek/deepseek-reasoner",
        "extraction": "anthropic/claude-3.5-sonnet",
    },
    Provider.HUGGINGFACE: {
        "default": "Qwen/Qwen2.5-7B-Instruct",
        "fast": "mistralai/Mistral-7B-Instruct-v0.3",
        "balanced": "Qwen/Qwen2.5-72B-Instruct",
        "reasoning": "Qwen/Qwen2.5-72B-Instruct",
        "extraction": "Qwen/Qwen2.5-72B-Instruct",
    },
    Provider.GOOGLE: {
        "default": "gemini-2.0-flash-exp",
        "fast": "gemini-2.0-flash-exp",
        "balanced": "gemini-1.5-pro",
        "reasoning": "gemini-1.5-pro",
        "extraction": "gemini-2.0-flash-exp",
    },
    Provider.ANTHROPIC: {
        "default": "claude-3-5-haiku-latest",
        "fast": "claude-3-5-haiku-latest",
        "balanced": "claude-sonnet-4-20250514",
        "reasoning": "claude-sonnet-4-20250514",
        "extraction": "claude-sonnet-4-20250514",
    },
    Provider.OLLAMA: {
        "default": "llama3.2",
        "fast": "llama3.2",
        "balanced": "llama3.1:70b",
        "reasoning": "qwen2.5:32b",
        "extraction": "llama3.1:8b",
    },
}

# Environment variable names per provider
PROVIDER_ENV_KEYS = {
    Provider.GROQ: "GROQ_API_KEY",
    Provider.TOGETHER: "TOGETHER_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
    Provider.HUGGINGFACE: "HUGGINGFACEHUB_API_TOKEN",
    Provider.GOOGLE: "GOOGLE_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OLLAMA: None,  # No API key needed for local
}


class ProviderFactory:
    """Factory for creating LLM instances from various providers."""

    @staticmethod
    def is_available(provider: Provider) -> bool:
        """Check if a provider is available (API key configured)."""
        env_key = PROVIDER_ENV_KEYS.get(provider)
        if env_key is None:  # Ollama doesn't need a key
            return True
        return bool(os.getenv(env_key))

    @staticmethod
    def get_available_providers() -> list[Provider]:
        """Get list of all providers with configured API keys."""
        return [p for p in Provider if ProviderFactory.is_available(p)]

    @staticmethod
    def create(
        provider: Provider,
        model_type: str = "default",
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an LLM instance for the specified provider.

        Args:
            provider: The LLM provider to use
            model_type: Type of model (default, fast, balanced, reasoning, extraction)
            model_id: Specific model ID (overrides model_type)
            **kwargs: Additional arguments passed to the model

        Returns:
            Configured LangChain chat model

        Raises:
            ValueError: If provider is not available or not supported
        """
        if not ProviderFactory.is_available(provider):
            env_key = PROVIDER_ENV_KEYS.get(provider, "API_KEY")
            raise ValueError(f"Provider {provider} not available. Set {env_key}.")

        # Get model ID
        models = PROVIDER_MODELS.get(provider, {})
        final_model_id = model_id or models.get(model_type, models.get("default"))

        # Create provider-specific model
        if provider == Provider.GROQ:
            from langchain_groq import ChatGroq

            return ChatGroq(model=final_model_id, **kwargs)

        elif provider == Provider.TOGETHER:
            from langchain_together import ChatTogether

            return ChatTogether(model=final_model_id, **kwargs)

        elif provider == Provider.OPENROUTER:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=final_model_id,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                **kwargs,
            )

        elif provider == Provider.HUGGINGFACE:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            llm = HuggingFaceEndpoint(
                repo_id=final_model_id,
                task="text-generation",
                **kwargs,
            )
            return ChatHuggingFace(llm=llm)

        elif provider == Provider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=final_model_id, **kwargs)

        elif provider == Provider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=final_model_id, **kwargs)

        elif provider == Provider.OLLAMA:
            from langchain_ollama import ChatOllama

            return ChatOllama(model=final_model_id, **kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")


def get_provider(provider_name: str) -> Provider:
    """Get Provider enum from string name."""
    try:
        return Provider(provider_name.lower())
    except ValueError:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {[p.value for p in Provider]}")


def get_model(
    provider: str = "groq",
    model_type: str = "default",
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Convenience function to get a configured LLM.

    Args:
        provider: Provider name (groq, together, openrouter, etc.)
        model_type: Model type (default, fast, balanced, reasoning, extraction)
        model_id: Specific model ID (overrides model_type)
        **kwargs: Additional model arguments

    Returns:
        Configured LangChain chat model
    """
    return ProviderFactory.create(
        provider=get_provider(provider),
        model_type=model_type,
        model_id=model_id,
        **kwargs,
    )


def list_available_providers() -> dict[str, list[str]]:
    """List all available providers and their models."""
    result = {}
    for provider in ProviderFactory.get_available_providers():
        models = PROVIDER_MODELS.get(provider, {})
        result[provider.value] = list(models.values())
    return result
