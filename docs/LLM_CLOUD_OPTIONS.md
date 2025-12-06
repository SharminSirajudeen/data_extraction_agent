# LLM Cloud Options: Beyond Groq Free Tier

A comprehensive guide to self-hosted and cloud-based LLM options when you outgrow Groq's free tier, prioritizing **uncompromising performance at minimal cost**.

## Quick Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHICH OPTION IS RIGHT FOR YOU?                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Budget: FREE                                                       │
│  ├── Experimenting? → Google Colab + Ollama                        │
│  ├── Light usage? → Groq Free Tier (stay here!)                    │
│  └── Need more? → SambaNova Free ($5 credit = 30M tokens)          │
│                                                                     │
│  Budget: $10-50/month                                               │
│  ├── Best value → Together AI ($0.20/M tokens)                     │
│  ├── Fastest → Cerebras ($0.10-0.60/M tokens, 1800 tok/s)          │
│  └── Most models → OpenRouter (aggregator)                         │
│                                                                     │
│  Budget: $50-200/month                                              │
│  ├── Production ready → Fireworks AI                               │
│  ├── Self-hosted → RunPod vLLM ($0.44/hr A40)                      │
│  └── Enterprise → AWS Bedrock / Google Vertex                      │
│                                                                     │
│  Budget: Scale (>100M tokens/month)                                │
│  ├── API-based → Together AI / Fireworks (volume discounts)        │
│  └── Self-hosted → RunPod/Vast.ai vLLM (break-even ~$0.02/M)       │
│                                                                     │
│  Budget: Massive Scale (>1B tokens/month)                          │
│  └── Self-hosted H100 reserved instances ($0.013/1K tokens)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tier 1: FREE Options

### 1. Groq Free Tier (Current - Stay Here If Possible!)
- **Cost**: FREE
- **Limits**: ~14,400 requests/day, 6,000 tokens/min
- **Speed**: 750 tokens/sec (Llama 3.1 8B)
- **Models**: Llama 3.1/3.3, Mixtral, Gemma
- **Best for**: Development, prototyping, low-volume production

### 2. Google Colab + Ollama (Self-Hosted Free)
- **Cost**: FREE (Colab free tier)
- **GPU**: NVIDIA T4 (limited sessions)
- **Setup**: 15 minutes
- **Models**: Any Ollama-supported model that fits in ~15GB VRAM

**Setup Steps**:
```python
# In Google Colab notebook
!sudo apt-get install -y pciutils
!curl https://ollama.ai/install.sh | sh

# Start Ollama server
import subprocess
subprocess.Popen(['ollama', 'serve'])

# Pull a model
!ollama pull llama3.2:1b

# Expose via ngrok (for external access)
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(11434)
print(f"Ollama API: {public_url}")
```

**Limitations**:
- Session timeouts (90 min inactive, 12 hr max)
- Not for production
- Network latency

### 3. SambaNova Free Tier
- **Cost**: $5 free credit (= 30M+ tokens on Llama 8B)
- **Speed**: 10x faster than GPUs (claimed)
- **Models**: Llama 3.1 8B/70B/405B, Llama 4, DeepSeek-R1, Qwen3
- **Expiry**: 90 days
- **Best for**: Testing before committing to paid tier

### 4. Cerebras Free Tier
- **Cost**: FREE with generous limits
- **Speed**: 1,800 tokens/sec (Llama 8B) - FASTEST!
- **Models**: Llama 3.1 8B, 70B
- **Best for**: Speed-critical applications

---

## Tier 2: Budget Options ($10-100/month)

### 1. Cerebras Developer Tier ⭐ RECOMMENDED FOR SPEED
- **Llama 3.1 8B**: $0.10/M tokens
- **Llama 3.1 70B**: $0.60/M tokens
- **Llama 3.1 405B**: $6/$12 per M tokens (input/output)
- **Speed**: 1,800 tok/s (8B), 450 tok/s (70B), 969 tok/s (405B)
- **Why**: 20x faster than GPU clouds, OpenAI-compatible API

### 2. Together AI ⭐ RECOMMENDED FOR VALUE
- **Llama 3.1 8B**: $0.20/M tokens
- **Llama 3.3 70B**: $0.88/M tokens
- **DeepSeek R1**: $0.55-$2.19/M tokens
- **Speed**: 86 tok/s, 0.5s TTFT
- **Why**: Broadest open-source model catalog, excellent docs

### 3. OpenRouter (Aggregator)
- **Pricing**: Varies by model (markup over provider costs)
- **Free models**: Gemini 2.0 Flash, some Llama variants
- **Why**: Single API for 100+ models across providers
- **Best for**: Flexibility to switch models easily

### 4. Fireworks AI
- **Llama 3.1 8B**: ~$0.20/M tokens
- **Llama 3.1 70B**: ~$0.90/M tokens
- **Speed**: 0.4s TTFT (fastest time-to-first-token)
- **Why**: Lowest latency for real-time applications

### 5. Google Colab Pro ($9.99/month)
- **GPU**: Priority T4, sometimes V100
- **Sessions**: Longer timeouts
- **Best for**: Development with Ollama/vLLM

### 6. Google Colab Pro+ ($49.99/month)
- **GPU**: A100 40GB access
- **Sessions**: Background execution
- **Best for**: Running larger models (70B quantized)

---

## Tier 3: Self-Hosted Options

### 1. RunPod Serverless (vLLM) ⭐ BEST SELF-HOSTED VALUE
- **A40 (48GB)**: $0.44/hr
- **A100 80GB**: $1.19/hr
- **H100 80GB**: $2.49/hr
- **Effective cost**: ~$0.02-0.04/M tokens at scale
- **Cold starts**: 60s (10s with FlashBoot +10% cost)

**Setup**:
```bash
# RunPod provides vLLM templates
# Just specify: model_id, GPU type, and you're live
```

### 2. Modal
- **T4**: ~$300-400/month
- **Cold starts**: 2-3s
- **DX**: Excellent (Python decorators)
- **Best for**: Python-native teams

### 3. Vast.ai (Cheapest GPUs)
- **Savings**: 40-60% cheaper than AWS/GCP
- **Model**: Decentralized GPU marketplace
- **Risk**: Variable performance by host
- **Best for**: Cost-sensitive batch processing

### 4. Replicate
- **Model**: Pay-per-prediction
- **Cold starts**: Low
- **Best for**: Quick deployments, pre-built models

### 5. Lambda Labs
- **H100**: ~$2.49/hr
- **Best for**: Reserved instances, ML training + inference

---

## Tier 4: Enterprise / Massive Scale

### Meta Llama Cloud (Official API)
- **Llama 4 Scout**: $0.189/$0.62 per M tokens
- **Llama 4 Maverick**: $0.28/$0.89 per M tokens
- **Status**: Limited preview (waitlist)
- **Partners**: AWS Bedrock, Google Vertex, Azure, Groq, Together

### AWS Bedrock
- **Llama 3.1 70B**: $0.75/$1.00 per 1K tokens
- **Best for**: Enterprise compliance, existing AWS infrastructure

### Google Vertex AI
- **Llama models**: Available
- **Best for**: GCP-native applications

### Anyscale Endpoints
- **BYOC**: Bring Your Own Cloud
- **Best for**: Privacy-sensitive deployments

---

## Cost Comparison Table

| Provider | Llama 8B (per M tokens) | Llama 70B (per M tokens) | Speed (tok/s) | Notes |
|----------|-------------------------|--------------------------|---------------|-------|
| **Groq Free** | FREE | FREE | 750 | Rate limited |
| **Cerebras Free** | FREE | FREE | 1,800 | Best free speed |
| **SambaNova Free** | FREE ($5 credit) | FREE ($5 credit) | Fast | 90-day expiry |
| **Cerebras Paid** | $0.10 | $0.60 | 1,800 | Fastest paid |
| **Together AI** | $0.20 | $0.88 | 86 | Best value |
| **Fireworks** | $0.20 | $0.90 | Fast | Lowest latency |
| **OpenRouter** | Varies | Varies | Varies | Aggregator |
| **RunPod vLLM** | ~$0.02* | ~$0.04* | Good | Self-hosted |
| **Colab + Ollama** | FREE | N/A | Slow | Development only |

*RunPod costs depend on GPU utilization

---

## Recommended Strategy for Data Extraction Agent

### Phase 1: Development (FREE)
```
Primary: Groq Free Tier
Backup: Google Colab + Ollama
Testing: SambaNova Free ($5 credit)
```

### Phase 2: Low Volume Production (<10M tokens/month)
```
Primary: Groq Free (while within limits)
Overflow: Together AI ($0.20-0.88/M)
Speed-critical: Cerebras ($0.10-0.60/M)
```

### Phase 3: Medium Volume (10-100M tokens/month)
```
Primary: Together AI (volume discounts available)
Speed-critical: Cerebras or Fireworks
Fallback: OpenRouter (route to cheapest available)
```

### Phase 4: High Volume (>100M tokens/month)
```
Primary: RunPod vLLM serverless (self-hosted)
Burst capacity: Together AI / Fireworks APIs
Evaluate: Reserved H100 instances if >50% utilization
```

---

## Implementation: Adding New Providers

Our Data Extraction Agent already supports adding new providers. Here's how to add Cerebras:

```python
# In providers/factory.py

Provider.CEREBRAS = "cerebras"

PROVIDER_MODELS[Provider.CEREBRAS] = {
    "default": "llama3.1-8b",
    "fast": "llama3.1-8b",
    "balanced": "llama3.1-70b",
    "reasoning": "llama3.1-70b",
}

PROVIDER_ENV_KEYS[Provider.CEREBRAS] = "CEREBRAS_API_KEY"

# In the create method:
elif provider == Provider.CEREBRAS:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=final_model_id,
        openai_api_base="https://api.cerebras.ai/v1",
        openai_api_key=os.getenv("CEREBRAS_API_KEY"),
        **kwargs,
    )
```

---

## Ollama Cloud (Official)

Ollama now offers their own cloud service:
- **Purpose**: Run large models on powerful cloud GPUs
- **Pricing**: Usage-based (coming soon)
- **Limits**: Hourly and weekly caps to avoid capacity issues
- **Access**: Through Ollama App, CLI, and API

---

## Key Takeaways

1. **Stay on Groq Free** as long as possible - it's excellent
2. **Cerebras** = fastest inference (1,800 tok/s)
3. **Together AI** = best price/performance ratio
4. **RunPod vLLM** = best for self-hosted at scale
5. **Google Colab + Ollama** = free development environment
6. **Don't self-host** until you hit >50% GPU utilization consistently

---

## Sources

- [Cerebras Pricing](https://www.cerebras.ai/pricing)
- [Cerebras Inference Launch](https://www.cerebras.ai/press-release/cerebras-launches-the-worlds-fastest-ai-inference)
- [SambaNova Plans](https://cloud.sambanova.ai/plans)
- [Together AI Pricing](https://www.together.ai/pricing)
- [RunPod vLLM Docs](https://docs.runpod.io/serverless/vllm/get-started)
- [Modal Serverless GPUs](https://modal.com/blog/serverless-gpu-article)
- [Ollama on Colab Guide](https://medium.com/data-science-collective/unleash-the-power-of-ai-host-your-own-ollama-models-for-free-with-google-colab-0aac5f237a9f)
- [Meta Llama API Pricing](https://llamaimodel.com/price/)
- [OpenRouter](https://openrouter.ai)
