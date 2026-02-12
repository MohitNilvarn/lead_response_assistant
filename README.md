# UrbanRoof Lead Response Assistant

> AI-powered workflow that reads customer enquiries and drafts helpful, guardrail-checked replies for UrbanRoof — a home repair and waterproofing company.

---

## Architecture Overview

```
Customer Enquiry
       │
       ▼
┌──────────────────┐
│  FastAPI Endpoint │  POST /api/v1/enquiry
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────┐
│ Intent Classifier │────▶│  Groq LLM (JSON)   │
│  (Step 1)        │     │  llama-3.3-70b     │
└────────┬─────────┘     └────────────────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────┐
│ Knowledge Base   │────▶│  FAISS + HuggingFace│
│  RAG Retrieval   │     │  Embeddings        │
│  (Step 2)        │     └────────────────────┘
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────┐
│ Response Draft   │────▶│  Groq LLM          │
│  Generation      │     │  (with RAG context)│
│  (Step 3)        │     └────────────────────┘
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Guardrails       │  Regex + rule-based post-processing
│  (Step 4)        │  Removes hallucinated prices, guarantees, timelines
└────────┬─────────┘
         │
         ▼
   Final Response
   (with intent, reply, follow-ups, sources, audit trail)
```

## Project Structure

```
lead_response_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory & entry point
│   ├── config.py                # Pydantic settings (env vars)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py # LLM-based intent classification
│   │   ├── knowledge_base.py    # FAISS vector store + RAG retrieval
│   │   ├── response_generator.py# Orchestration pipeline
│   │   └── guardrails.py        # Hallucination & safety guardrails
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py # HuggingFace sentence-transformers
│   │   └── groq_service.py      # Groq API client with retries
│   └── data/
│       ├── __init__.py
│       └── urbanroof_knowledge.json  # Domain knowledge base
├── tests/
│   ├── __init__.py
│   └── test_assistant.py        # Unit + integration tests
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
└── README.md
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Modular pipeline** | Each step (intent → RAG → generation → guardrails) is independent and testable |
| **FAISS for vector search** | Fast, in-memory, zero-infrastructure similarity search |
| **HuggingFace `all-MiniLM-L6-v2`** | Lightweight (80 MB), high-quality embeddings, runs on CPU |
| **Groq `llama-3.3-70b-versatile`** | Fast inference, strong instruction following, JSON mode support |
| **Regex-based guardrails** | Deterministic, auditable, zero-latency post-processing |
| **Pydantic schemas everywhere** | Type safety, validation, and auto-generated API docs |
| **Tenacity retries on Groq** | Production resilience against transient API failures |

## Setup & Installation

### Prerequisites
- Python 3.10+
- A Groq API key ([get one free](https://console.groq.com/keys))

### Steps

```bash
# 1. Clone / navigate to the project
cd lead_response_assistant

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# 5. Add your Groq API key to .env
# Edit .env and set: GROQ_API_KEY=gsk_your_actual_key_here

# 6. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API docs are available at: **http://localhost:8000/docs**

## API Endpoints

### `POST /api/v1/enquiry` — Process Customer Enquiry

**Request:**
```json
{
  "message": "Hi, I am getting damp patches on my bedroom wall after rains. What should I do?",
  "customer_name": "Rahul"
}
```

**Response:**
```json
{
  "request_id": "a1b2c3d4e5f6",
  "timestamp": "2026-02-12T10:30:00Z",
  "original_message": "Hi, I am getting damp patches on my bedroom wall after rains. What should I do?",
  "intent_analysis": {
    "primary_intent": "leakage_seepage",
    "confidence": 0.92,
    "secondary_intents": ["water_damage", "wall_repair"],
    "urgency": "medium",
    "key_topics": ["damp patches", "bedroom wall", "rain-related moisture"]
  },
  "drafted_reply": "Hello Rahul! Thank you for reaching out to us...",
  "follow_up_questions": [
    {
      "question": "Which floor is your bedroom located on?",
      "reason": "Helps determine if the issue is from the roof/terrace or external wall"
    },
    {
      "question": "How long have you been noticing these damp patches?",
      "reason": "Duration helps assess severity and urgency of the issue"
    }
  ],
  "guardrail_check": {
    "passed": true,
    "flags": [],
    "modifications_applied": []
  },
  "sources_used": [
    "common_issues/Damp Patches on Walls",
    "services/Thermal Scanning and Inspection"
  ]
}
```

### `POST /api/v1/intent` — Intent Classification Only
### `POST /api/v1/knowledge` — Knowledge Retrieval Only
### `GET /api/v1/health` — Health Check

## How the System Ensures Accuracy & Reliability

1. **RAG (Retrieval-Augmented Generation)**: Every response is grounded in UrbanRoof's actual knowledge base — the LLM receives verified context, not just its training data.

2. **Intent Classification**: A separate LLM call with structured JSON output classifies the customer's intent before response generation, ensuring the reply is contextually appropriate.

3. **Guardrails (Post-Processing)**: Regex-based rules scan every generated reply for:
   - Fabricated pricing (₹, Rs., INR amounts)
   - False timelines ("within 3 days")
   - Absolute guarantees ("100% guarantee", "lifetime warranty")
   - Scare tactics / competitor bashing
   - AI self-references

4. **Prompt Engineering**: The system prompt contains 10 strict rules that the LLM must follow, including never diagnosing exact problems, never fabricating prices, and always recommending professional assessment.

5. **Retry Logic**: Groq API calls use exponential backoff (via `tenacity`) to handle transient failures gracefully.

## Known Limitations

- **Single-turn only**: The current API processes one message at a time. Multi-turn conversation memory is not implemented.
- **Static knowledge base**: The knowledge JSON is loaded at startup. Adding new knowledge requires a restart.
- **CPU-only embeddings**: The embedding model runs on CPU. For high-throughput, GPU acceleration would be needed.
- **No user authentication**: The API currently has no auth layer.
- **Guardrails are regex-based**: Sophisticated prompt injections or novel hallucination patterns may not be caught.
- **No conversation logging/analytics**: Enquiries are not persisted for analytics.

## What I Would Improve With More Time

1. **Multi-turn conversation**: Add session management with Redis to maintain conversation history across multiple messages.
2. **Dynamic knowledge base**: Admin API to add/update/delete knowledge entries without restart + persistent vector store (Qdrant/Pinecone).
3. **LLM-based guardrails**: Add a second LLM pass that verifies factual claims against the knowledge base (claim verification).
4. **Streaming responses**: Use SSE/WebSocket for real-time streaming of the generated reply.
5. **Analytics dashboard**: Log all enquiries, intents, guardrail flags, and response quality metrics.
6. **A/B testing**: Test different prompt variants to optimise reply quality.
7. **Multi-language support**: Detect language and respond in Hindi/regional languages.
8. **Rate limiting & auth**: JWT-based auth + per-user rate limiting for production deployment.
9. **Docker deployment**: Containerise with Docker Compose for easy deployment.
10. **Automated evaluation**: Build an eval suite with expected outputs to regression-test prompt changes.

## Running Tests

```bash
# Unit tests only (no API key needed)
pytest tests/ -v -m "not integration"

# All tests including integration (requires GROQ_API_KEY)
pytest tests/ -v
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend Framework | FastAPI |
| LLM Inference | Groq API (Llama 3.3 70B) |
| Embeddings | HuggingFace sentence-transformers (MiniLM-L6-v2) |
| Vector Store | FAISS (in-memory) |
| Validation | Pydantic v2 |
| Resilience | Tenacity (retry with backoff) |
| Testing | Pytest |

---

*Built for the UrbanRoof Lead Response Assistant assignment.*
