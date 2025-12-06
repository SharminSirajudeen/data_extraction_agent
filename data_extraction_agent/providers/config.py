"""Provider configurations.

This file contains ALL provider configurations. To add a new provider:
1. Add a ProviderConfig below
2. That's it! No other code changes needed.

The registry pattern means you NEVER touch business logic to add providers.
"""

from data_extraction_agent.providers.base import (
    ProviderConfig,
    ProviderRegistry,
    ProviderTier,
)


# ============================================================================
# FREE TIER PROVIDERS
# ============================================================================

ProviderRegistry.register(ProviderConfig(
    name="groq",
    env_key="GROQ_API_KEY",
    base_url=None,  # Uses native client
    tier=ProviderTier.FREE,
    speed_tokens_per_sec=750,
    client_type="groq",
    models={
        "default": "llama-3.1-8b-instant",
        "fast": "llama-3.1-8b-instant",
        "balanced": "llama-3.3-70b-versatile",
        "reasoning": "qwen-qwq-32b",
        "extraction": "llama-3.3-70b-versatile",
    },
))

ProviderRegistry.register(ProviderConfig(
    name="cerebras",
    env_key="CEREBRAS_API_KEY",
    base_url="https://api.cerebras.ai/v1",
    tier=ProviderTier.FREE,
    speed_tokens_per_sec=1800,  # FASTEST!
    client_type="openai_compatible",
    models={
        "default": "llama3.1-8b",
        "fast": "llama3.1-8b",
        "balanced": "llama3.1-70b",
        "reasoning": "llama3.1-70b",
        "extraction": "llama3.1-8b",
    },
))

ProviderRegistry.register(ProviderConfig(
    name="sambanova",
    env_key="SAMBANOVA_API_KEY",
    base_url="https://api.sambanova.ai/v1",
    tier=ProviderTier.FREE,  # $5 free credit
    speed_tokens_per_sec=500,
    client_type="openai_compatible",
    models={
        "default": "Meta-Llama-3.1-8B-Instruct",
        "fast": "Meta-Llama-3.1-8B-Instruct",
        "balanced": "Meta-Llama-3.1-70B-Instruct",
        "reasoning": "Meta-Llama-3.1-405B-Instruct",
        "extraction": "Meta-Llama-3.1-70B-Instruct",
    },
))

# ============================================================================
# BUDGET PROVIDERS
# ============================================================================

ProviderRegistry.register(ProviderConfig(
    name="together",
    env_key="TOGETHER_API_KEY",
    base_url=None,  # Uses native client
    tier=ProviderTier.BUDGET,
    speed_tokens_per_sec=86,
    client_type="together",
    models={
        "default": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "fast": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "balanced": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "extraction": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    },
))

ProviderRegistry.register(ProviderConfig(
    name="fireworks",
    env_key="FIREWORKS_API_KEY",
    base_url="https://api.fireworks.ai/inference/v1",
    tier=ProviderTier.BUDGET,
    speed_tokens_per_sec=200,
    client_type="openai_compatible",
    models={
        "default": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "fast": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "balanced": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "reasoning": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "extraction": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    },
))

ProviderRegistry.register(ProviderConfig(
    name="openrouter",
    env_key="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1",
    tier=ProviderTier.BUDGET,
    speed_tokens_per_sec=100,
    client_type="openai_compatible",
    models={
        "default": "google/gemini-2.0-flash-exp:free",
        "fast": "google/gemini-2.0-flash-exp:free",
        "balanced": "deepseek/deepseek-chat",
        "reasoning": "deepseek/deepseek-reasoner",
        "extraction": "meta-llama/llama-3.1-70b-instruct",
    },
))

# ============================================================================
# LOCAL / SELF-HOSTED PROVIDERS
# ============================================================================

ProviderRegistry.register(ProviderConfig(
    name="ollama",
    env_key=None,  # No API key needed
    base_url="http://localhost:11434",
    tier=ProviderTier.FREE,
    speed_tokens_per_sec=30,  # Depends on hardware
    client_type="ollama",
    models={
        "default": "llama3.2",
        "fast": "llama3.2:1b",
        "balanced": "llama3.1:8b",
        "reasoning": "qwen2.5:14b",
        "extraction": "llama3.1:8b",
    },
))

ProviderRegistry.register(ProviderConfig(
    name="huggingface",
    env_key="HUGGINGFACEHUB_API_TOKEN",
    base_url=None,
    tier=ProviderTier.FREE,
    speed_tokens_per_sec=20,
    client_type="huggingface",
    models={
        "default": "Qwen/Qwen2.5-7B-Instruct",
        "fast": "mistralai/Mistral-7B-Instruct-v0.3",
        "balanced": "Qwen/Qwen2.5-72B-Instruct",
        "reasoning": "Qwen/Qwen2.5-72B-Instruct",
        "extraction": "Qwen/Qwen2.5-72B-Instruct",
    },
))

# ============================================================================
# COMMERCIAL PROVIDERS (Fallback)
# ============================================================================

ProviderRegistry.register(ProviderConfig(
    name="google",
    env_key="GOOGLE_API_KEY",
    base_url=None,
    tier=ProviderTier.STANDARD,
    speed_tokens_per_sec=150,
    client_type="google",
    models={
        "default": "gemini-2.0-flash-exp",
        "fast": "gemini-2.0-flash-exp",
        "balanced": "gemini-1.5-pro",
        "reasoning": "gemini-1.5-pro",
        "extraction": "gemini-2.0-flash-exp",
    },
))

ProviderRegistry.register(ProviderConfig(
    name="anthropic",
    env_key="ANTHROPIC_API_KEY",
    base_url=None,
    tier=ProviderTier.PREMIUM,
    speed_tokens_per_sec=100,
    client_type="anthropic",
    models={
        "default": "claude-3-5-haiku-latest",
        "fast": "claude-3-5-haiku-latest",
        "balanced": "claude-sonnet-4-20250514",
        "reasoning": "claude-sonnet-4-20250514",
        "extraction": "claude-sonnet-4-20250514",
    },
))


# ============================================================================
# EASY EXTENSION: Add your own providers!
# ============================================================================
#
# To add a new provider, just copy-paste and modify:
#
# ProviderRegistry.register(ProviderConfig(
#     name="my_provider",
#     env_key="MY_PROVIDER_API_KEY",
#     base_url="https://api.myprovider.com/v1",  # For OpenAI-compatible
#     tier=ProviderTier.BUDGET,
#     speed_tokens_per_sec=100,
#     client_type="openai_compatible",  # Most APIs are OpenAI-compatible now
#     models={
#         "default": "my-model-8b",
#         "fast": "my-model-8b",
#         "balanced": "my-model-70b",
#     },
# ))
#
# That's it! No other code changes needed.
# ============================================================================
