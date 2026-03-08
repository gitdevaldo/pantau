# Pantau Backend — Product Requirements Document

> FastAPI REST API for real-time fraud scoring, transaction management, and ML model lifecycle.
> Part of **Pantau** — AI fraud detection for illegal online gambling (judol) in Indonesian QRIS payments.

---

## 1. Executive Summary

The Pantau Backend is a FastAPI-based REST API that serves as the core intelligence layer between
the ML detection engine and downstream consumers (dashboard, B2B integrations, payment providers).
It loads trained ML models (.pkl), scores incoming QRIS transactions in real-time through the
6-layer ensemble, stores results in PostgreSQL, provides historical analytics, manages
authentication, and exposes LLM-powered explainability for flagged transactions.

**Target integration:** B2B — payment providers, banks, payment gateways, e-wallet providers.
**Demo users:** Developer (admin) and hackathon judges.

---

## 2. Problem Statement

The ML pipeline produces trained models and can score datasets in batch. However, there is no
real-time interface for:
- Scoring individual transactions as they arrive
- Storing and querying historical scored transactions
- Managing ML model versions and retraining
- Providing human-readable explanations for flagged transactions
- Integrating with external systems (banks, payment gateways) via API

---

## 3. Goals & Success Criteria

| Goal | Success Criteria |
|------|-----------------|
| Real-time scoring | < 500ms response time per transaction |
| Batch scoring | Process 1,000 transactions in < 30 seconds |
| Data persistence | All scored transactions stored in PostgreSQL |
| Model management | Hot-swap models without API restart |
| Authentication | JWT-based auth for API access |
| Explainability | LLM generates natural language explanation per flagged transaction |
| B2B ready | OpenAPI/Swagger docs auto-generated |
| Reliability | Graceful error handling, health checks, structured logging |

---

## 4. User Stories

### 4.1 Payment Provider (B2B Integration)
- **As a** payment provider, **I want to** send a QRIS transaction to the API and receive a risk
  score in real-time, **so that** I can block or flag suspicious transactions before settlement.
- **As a** payment provider, **I want to** send a batch of transactions, **so that** I can score
  historical data for compliance reporting.

### 4.2 Compliance Officer (Dashboard User)
- **As a** compliance officer, **I want to** query flagged transactions by risk level, date range,
  merchant, or user, **so that** I can investigate suspicious activity.
- **As a** compliance officer, **I want to** see a natural language explanation of why a transaction
  was flagged, **so that** I can make informed decisions without understanding ML internals.

### 4.3 System Administrator
- **As a** system admin, **I want to** check which ML model version is currently loaded, **so that**
  I know if the system is using the latest trained models.
- **As a** system admin, **I want to** trigger a model reload without restarting the API, **so that**
  new models can be deployed seamlessly.
- **As a** system admin, **I want to** monitor API health and performance, **so that** I can ensure
  uptime.

---

## 5. Technical Architecture

### 5.1 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | FastAPI | Latest |
| Server | Uvicorn | Latest |
| Database | PostgreSQL | 15+ |
| ORM | SQLAlchemy + Alembic (migrations) | 2.0+ |
| Auth | JWT (python-jose + passlib) | — |
| ML Runtime | scikit-learn, NetworkX, NumPy, Pandas | Same as ML pipeline |
| LLM | OpenAI API (GPT-4) or local model | — |
| Validation | Pydantic v2 | — |
| CORS | FastAPI CORSMiddleware | — |
| Logging | Python logging (structured JSON) | — |

### 5.2 Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, startup/shutdown, CORS
│   ├── config.py               # Settings (env vars, paths, secrets)
│   ├── dependencies.py         # Shared dependencies (DB session, auth)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py       # v1 router aggregator
│   │   │   ├── scoring.py      # POST /score, POST /score/batch
│   │   │   ├── transactions.py # GET /transactions, filters, pagination
│   │   │   ├── analytics.py    # GET /analytics/summary, /risk-distribution
│   │   │   ├── models.py       # GET /models/status, POST /models/reload
│   │   │   ├── explain.py      # POST /explain/{transaction_id}
│   │   │   └── auth.py         # POST /auth/login, /auth/register
│   │   └── health.py           # GET /health, GET /ready
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── scoring_engine.py   # Load models, run 6-layer scoring
│   │   ├── explainer.py        # LLM explainability integration
│   │   └── security.py         # JWT creation/verification, password hashing
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy engine, session factory
│   │   ├── models.py           # ORM models (Transaction, User, ModelVersion)
│   │   └── migrations/         # Alembic migrations
│   │
│   └── schemas/
│       ├── __init__.py
│       ├── transaction.py      # Pydantic request/response schemas
│       ├── scoring.py          # Score request/response
│       ├── analytics.py        # Analytics response schemas
│       ├── auth.py             # Login/register schemas
│       └── explain.py          # Explanation response schema
│
├── requirements.txt
├── Dockerfile
├── .env.example
└── PRD.md                      # This file
```

### 5.3 Database Schema

```sql
-- Scored transactions (core table)
CREATE TABLE transactions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id  VARCHAR(50) UNIQUE NOT NULL,
    user_id         VARCHAR(50) NOT NULL,
    merchant_id     VARCHAR(50) NOT NULL,
    amount          INTEGER NOT NULL,
    timestamp       TIMESTAMP NOT NULL,
    tx_type         VARCHAR(10) DEFAULT 'QRIS',
    is_round_amount BOOLEAN,
    user_city       VARCHAR(100),
    user_province   VARCHAR(100),
    merchant_city   VARCHAR(100),
    merchant_province VARCHAR(100),

    -- Scoring results
    user_score      FLOAT,
    merchant_score  FLOAT,
    network_score   FLOAT,
    temporal_score  FLOAT,
    velocity_score  FLOAT,
    flow_score      FLOAT,
    final_score     FLOAT NOT NULL,
    risk_level      VARCHAR(20) NOT NULL,
    layers_flagged  INTEGER,

    -- Metadata
    model_version   VARCHAR(50),
    scored_at       TIMESTAMP DEFAULT NOW(),
    created_at      TIMESTAMP DEFAULT NOW(),

    -- Indexes
    -- CREATE INDEX idx_transactions_risk ON transactions(risk_level);
    -- CREATE INDEX idx_transactions_merchant ON transactions(merchant_id);
    -- CREATE INDEX idx_transactions_user ON transactions(user_id);
    -- CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
    -- CREATE INDEX idx_transactions_final_score ON transactions(final_score);
);

-- API users (auth)
CREATE TABLE api_users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username    VARCHAR(100) UNIQUE NOT NULL,
    email       VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role        VARCHAR(20) DEFAULT 'viewer',  -- admin, analyst, viewer
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Model version tracking
CREATE TABLE model_versions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tag         VARCHAR(50) NOT NULL,           -- 'gan', 'parametric'
    version     VARCHAR(50) NOT NULL,
    model_path  VARCHAR(255) NOT NULL,
    metrics     JSONB,                          -- F1, AUC-ROC, etc.
    is_active   BOOLEAN DEFAULT FALSE,
    loaded_at   TIMESTAMP,
    created_at  TIMESTAMP DEFAULT NOW()
);
```

---

## 6. API Endpoints

### 6.1 Scoring

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/score` | Score a single transaction | Required |
| POST | `/api/v1/score/batch` | Score multiple transactions | Required |

**POST /api/v1/score — Request:**
```json
{
    "transaction_id": "TXN1a2b3c4d5e6f",
    "user_id": "USR1a2b3c4d5e6f",
    "merchant_id": "A1B2C3D4E5F6G7H",
    "amount": 150000,
    "timestamp": "2026-03-08T22:31:00",
    "tx_type": "QRIS",
    "is_round_amount": true,
    "user_city": "KOTA SURABAYA",
    "user_province": "JAWA TIMUR",
    "merchant_city": "KOTA SURABAYA",
    "merchant_province": "JAWA TIMUR"
}
```

**Response:**
```json
{
    "transaction_id": "TXN1a2b3c4d5e6f",
    "final_score": 82.5,
    "risk_level": "Critical",
    "layer_scores": {
        "user_behavior": 75.2,
        "merchant_behavior": 91.0,
        "network_cluster": 88.5,
        "temporal_pattern": 65.0,
        "velocity_delta": 79.3,
        "money_flow": 95.1
    },
    "layers_flagged": 5,
    "scored_at": "2026-03-08T22:31:01.234Z",
    "model_version": "gan-v1"
}
```

### 6.2 Transactions

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/v1/transactions` | List scored transactions (paginated, filtered) | Required |
| GET | `/api/v1/transactions/{id}` | Get single transaction detail | Required |

**Query Parameters:** `risk_level`, `min_score`, `max_score`, `merchant_id`, `user_id`,
`date_from`, `date_to`, `page`, `per_page`, `sort_by`, `order`

### 6.3 Analytics

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/v1/analytics/summary` | Overall stats (total, flagged, by risk level) | Required |
| GET | `/api/v1/analytics/risk-distribution` | Score distribution histogram | Required |
| GET | `/api/v1/analytics/top-merchants` | Top flagged merchants | Required |
| GET | `/api/v1/analytics/top-users` | Top flagged users | Required |
| GET | `/api/v1/analytics/timeline` | Flagged transactions over time | Required |
| GET | `/api/v1/analytics/geographic` | Risk by province/city | Required |

### 6.4 Explainability

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/explain/{transaction_id}` | LLM explanation for a flagged transaction | Required |

**Response:**
```json
{
    "transaction_id": "TXN1a2b3c4d5e6f",
    "explanation": "This transaction is flagged as Critical Risk (score: 82.5) because the merchant A1B2C3D4E5F6G7H exhibits strong judol payment gateway behavior: receiving 847 transactions from 312 unique senders in the past 7 days, with 89% being round amounts (Rp 50,000 and Rp 100,000). The transaction occurred at 22:31 WIB (prime gambling hours). Network analysis shows this merchant shares user pools with 4 other flagged merchants, forming a suspected operator ring.",
    "risk_factors": [
        {"layer": "money_flow", "score": 95.1, "reason": "High fan-in: 312 unique senders, 89% round amounts"},
        {"layer": "merchant_behavior", "score": 91.0, "reason": "847 tx/week, 10x normal merchant velocity"},
        {"layer": "network_cluster", "score": 88.5, "reason": "Connected to 4 flagged merchants via shared users"}
    ]
}
```

### 6.5 Model Management

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/v1/models/status` | Current loaded model info + metrics | Admin |
| POST | `/api/v1/models/reload` | Hot-reload models from disk | Admin |
| GET | `/api/v1/models/versions` | List all model versions | Admin |

### 6.6 Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/auth/login` | Login, returns JWT | Public |
| POST | `/api/v1/auth/register` | Register new user | Admin |
| GET | `/api/v1/auth/me` | Current user info | Required |

### 6.7 Health

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/health` | Basic health check | Public |
| GET | `/ready` | Readiness (DB + models loaded) | Public |

---

## 7. Core Components

### 7.1 Scoring Engine (`core/scoring_engine.py`)

Singleton class that:
1. **On startup:** Loads all 6 layer .pkl files from `models/{tag}/` into memory
2. **On request:** Receives transaction → engineers features → runs through 6 layers →
   combines with weights → returns final score + risk level
3. **On reload:** Atomically swaps models without downtime

Reuses existing ML code:
- `ml.models.{layer}.train()` for feature engineering
- `ml.scoring.combine_scores()` for weighted combination
- `ml.scoring.WEIGHTS` updated from `models/{tag}/weights.pkl`

### 7.2 LLM Explainer (`core/explainer.py`)

Takes a scored transaction + layer scores → constructs a structured prompt → calls LLM API →
returns natural language explanation. Uses few-shot examples calibrated to judol domain:
- References specific judol patterns (late-night, round amounts, togel timing)
- Cites numeric evidence from layer scores
- Provides actionable recommendation (review, escalate, freeze)

### 7.3 Security (`core/security.py`)

- JWT tokens with configurable expiry (default: 24h)
- Password hashing with bcrypt (passlib)
- Role-based access: `admin` (full), `analyst` (read + score), `viewer` (read only)
- API key support for B2B integrations (alternative to JWT)

---

## 8. Non-Functional Requirements

| Requirement | Target |
|------------|--------|
| Response time (single score) | < 500ms |
| Response time (batch 1000) | < 30s |
| Concurrent connections | 100+ |
| Database query performance | Indexed queries < 100ms |
| Uptime | 99.9% (production target) |
| API documentation | Auto-generated Swagger/OpenAPI |
| Logging | Structured JSON, request/response logging |
| Error handling | Consistent error response format |
| CORS | Configurable allowed origins (dashboard URL) |

---

## 9. Environment Variables

```env
# Database
DATABASE_URL=postgresql://pantau:password@localhost:5432/pantau

# JWT
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# ML Models
MODEL_TAG=gan
MODEL_DIR=../models

# LLM
LLM_API_KEY=your-openai-key
LLM_MODEL=gpt-4

# Server
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000

# Logging
LOG_LEVEL=INFO
```

---

## 10. Dependencies

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
sqlalchemy>=2.0.0
alembic>=1.13.0
psycopg2-binary>=2.9.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
networkx>=3.0
openai>=1.0.0
python-multipart>=0.0.6
httpx>=0.25.0
```

---

## 11. Milestones

| Phase | Scope |
|-------|-------|
| **Phase 1: Core Scoring** | FastAPI scaffold, scoring engine, POST /score, health checks |
| **Phase 2: Database** | PostgreSQL setup, ORM models, Alembic migrations, transaction storage |
| **Phase 3: API Endpoints** | Transactions list, analytics, filters, pagination |
| **Phase 4: Auth** | JWT auth, user management, role-based access |
| **Phase 5: Explainability** | LLM integration, POST /explain endpoint |
| **Phase 6: Polish** | Error handling, logging, Dockerfile, documentation |

---

*PRD for Pantau Backend — PIDI DIGDAYA X Hackathon 2026*
