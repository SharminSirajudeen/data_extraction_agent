"""Provider Factory - Creates LLM instances for various providers.

Supports: Groq, Together AI, OpenRouter, HuggingFace, Google, Anthropic,
          Cerebras, SambaNova, Fireworks, Ollama
"""

import os
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel


class Provider(str, Enum):
    """Supported LLM providers - Open Source First!"""

    # FREE TIER PROVIDERS (Priority 1)
    GROQ = "groq"  # Free tier, 750 tok/s
    CEREBRAS = "cerebras"  # Free tier, 1800 tok/s - FASTEST!
    SAMBANOVA = "sambanova"  # $5 free credit = 30M tokens

    # BUDGET PROVIDERS (Priority 2)
    TOGETHER = "together"  # $0.20/M tokens
    FIREWORKS = "fireworks"  # Lowest latency
    OPENROUTER = "openrouter"  # Aggregator, some free models

    # OPEN SOURCE LOCAL (Priority 3)
    OLLAMA = "ollama"  # Local/Colab - completely free
    HUGGINGFACE = "huggingface"  # Free tier available

    # COMMERCIAL FALLBACK (Priority 4)
    GOOGLE = "google"  # Gemini
    ANTHROPIC = "anthropic"  # Claude - expensive but best quality


# Model configurations per provider
PROVIDER_MODELS = {
    # === FREE TIER PROVIDERS ===
    Provider.GROQ: {
        "default": "llama-3.1-8b-instant",
        "fast": "llama-3.1-8b-instant",
        "balanced": "llama-3.3-70b-versatile",
        "reasoning": "qwen-qwq-32b",
        "extraction": "llama-3.3-70b-versatile",
    },
    Provider.CEREBRAS: {
        "default": "llama3.1-8b",
        "fast": "llama3.1-8b",  # 1800 tok/s!
        "balanced": "llama3.1-70b",  # 450 tok/s
        "reasoning": "llama3.1-70b",
        "extraction": "llama3.1-8b",
    },
    Provider.SAMBANOVA: {
        "default": "Meta-Llama-3.1-8B-Instruct",
        "fast": "Meta-Llama-3.1-8B-Instruct",
        "balanced": "Meta-Llama-3.1-70B-Instruct",
        "reasoning": "Meta-Llama-3.1-405B-Instruct",
        "extraction": "Meta-Llama-3.1-70B-Instruct",
    },
    # === BUDGET PROVIDERS ===
    Provider.TOGETHER: {
        "default": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "fast": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "balanced": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "extraction": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    },
    Provider.FIREWORKS: {
        "default": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "fast": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "balanced": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "reasoning": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "extraction": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    },
    Provider.OPENROUTER: {
        "default": "google/gemini-2.0-flash-exp:free",
        "fast": "google/gemini-2.0-flash-exp:free",
        "balanced": "deepseek/deepseek-chat",
        "reasoning": "deepseek/deepseek-reasoner",
        "extraction": "meta-llama/llama-3.1-70b-instruct",
    },
    # === LOCAL/SELF-HOSTED ===
    Provider.OLLAMA: {
        "default": "llama3.2",
        "fast": "llama3.2:1b",
        "balanced": "llama3.1:8b",
        "reasoning": "qwen2.5:14b",
        "extraction": "llama3.1:8b",
    },
    Provider.HUGGINGFACE: {
        "default": "Qwen/Qwen2.5-7B-Instruct",
        "fast": "mistralai/Mistral-7B-Instruct-v0.3",
        "balanced": "Qwen/Qwen2.5-72B-Instruct",
        "reasoning": "Qwen/Qwen2.5-72B-Instruct",
        "extraction": "Qwen/Qwen2.5-72B-Instruct",
    },
    # === COMMERCIAL FALLBACK ===
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
}

# Environment variable names per provider
PROVIDER_ENV_KEYS = {
    Provider.GROQ: "GROQ_API_KEY",
    Provider.CEREBRAS: "CEREBRAS_API_KEY",
    Provider.SAMBANOVA: "SAMBANOVA_API_KEY",
    Provider.TOGETHER: "TOGETHER_API_KEY",
    Provider.FIREWORKS: "FIREWORKS_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
    Provider.OLLAMA: None,  # No API key needed for local
    Provider.HUGGINGFACE: "HUGGINGFACEHUB_API_TOKEN",
    Provider.GOOGLE: "GOOGLE_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}

# Provider metadata for smart routing
PROVIDER_INFO = {
    Provider.GROQ: {
        "tier": "free",
        "speed_tok_s": 750,
        "cost_per_m_8b": 0.0,
        "cost_per_m_70b": 0.0,
    },
    Provider.CEREBRAS: {
        "tier": "free",
        "speed_tok_s": 1800,  # FASTEST!
        "cost_per_m_8b": 0.0,  # Free tier
        "cost_per_m_70b": 0.0,  # Free tier
    },
    Provider.SAMBANOVA: {
        "tier": "free",
        "speed_tok_s": 500,
        "cost_per_m_8b": 0.0,  # $5 credit
        "cost_per_m_70b": 0.0,
    },
    Provider.TOGETHER: {
        "tier": "budget",
        "speed_tok_s": 86,
        "cost_per_m_8b": 0.20,
        "cost_per_m_70b": 0.88,
    },
    Provider.FIREWORKS: {
        "tier": "budget",
        "speed_tok_s": 200,
        "cost_per_m_8b": 0.20,
        "cost_per_m_70b": 0.90,
    },
    Provider.OPENROUTER: {
        "tier": "budget",
        "speed_tok_s": 100,
        "cost_per_m_8b": 0.0,  # Free models available
        "cost_per_m_70b": 0.50,
    },
    Provider.OLLAMA: {
        "tier": "free",
        "speed_tok_s": 30,  # Depends on hardware
        "cost_per_m_8b": 0.0,
        "cost_per_m_70b": 0.0,
    },
    Provider.HUGGINGFACE: {
        "tier": "free",
        "speed_tok_s": 20,
        "cost_per_m_8b": 0.0,
        "cost_per_m_70b": 0.0,
    },
    Provider.GOOGLE: {
        "tier": "commercial",
        "speed_tok_s": 150,
        "cost_per_m_8b": 0.075,
        "cost_per_m_70b": 0.30,
    },
    Provider.ANTHROPIC: {
        "tier": "commercial",
        "speed_tok_s": 100,
        "cost_per_m_8b": 0.25,  # Haiku
        "cost_per_m_70b": 3.00,  # Sonnet
    },
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
    def get_free_providers() -> list[Provider]:
        """Get providers with free tiers."""
        return [
            p for p in ProviderFactory.get_available_providers()
            if PROVIDER_INFO.get(p, {}).get("tier") == "free"
        ]

    @staticmethod
    def get_fastest_provider() -> Optional[Provider]:
        """Get the fastest available provider."""
        available = ProviderFactory.get_available_providers()
        if not available:
            return None
        return max(available, key=lambda p: PROVIDER_INFO.get(p, {}).get("speed_tok_s", 0))

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

        elif provider == Provider.CEREBRAS:
            # Cerebras uses OpenAI-compatible API
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=final_model_id,
                openai_api_base="https://api.cerebras.ai/v1",
                openai_api_key=os.getenv("CEREBRAS_API_KEY"),
                **kwargs,
            )

        elif provider == Provider.SAMBANOVA:
            # SambaNova uses OpenAI-compatible API
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=final_model_id,
                openai_api_base="https://api.sambanova.ai/v1",
                openai_api_key=os.getenv("SAMBANOVA_API_KEY"),
                **kwargs,
            )

        elif provider == Provider.TOGETHER:
            from langchain_together import ChatTogether
            return ChatTogether(model=final_model_id, **kwargs)

        elif provider == Provider.FIREWORKS:
            # Fireworks uses OpenAI-compatible API
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=final_model_id,
                openai_api_base="https://api.fireworks.ai/inference/v1",
                openai_api_key=os.getenv("FIREWORKS_API_KEY"),
                **kwargs,
            )

        elif provider == Provider.OPENROUTER:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=final_model_id,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                **kwargs,
            )

        elif provider == Provider.OLLAMA:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=final_model_id, **kwargs)

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
        provider: Provider name (groq, cerebras, together, etc.)
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


def get_cheapest_provider() -> Optional[Provider]:
    """Get the cheapest available provider for 8B models."""
    free_providers = ProviderFactory.get_free_providers()
    if free_providers:
        return free_providers[0]

    available = ProviderFactory.get_available_providers()
    if not available:
        return None

    return min(
        available,
        key=lambda p: PROVIDER_INFO.get(p, {}).get("cost_per_m_8b", float("inf"))
    )
