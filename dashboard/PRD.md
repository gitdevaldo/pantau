# Pantau Dashboard — Product Requirements Document

> Next.js real-time fraud monitoring dashboard with analytics and LLM-powered explainability.
> Part of **Pantau** — AI fraud detection for illegal online gambling (judol) in Indonesian QRIS payments.

---

## 1. Executive Summary

**Problem:** The ML engine and backend API produce fraud scores, but there is no visual interface
for compliance officers, bank analysts, or hackathon judges to monitor, investigate, and understand
flagged transactions in real-time.

**Solution:** A Next.js dashboard that provides real-time scoring feed, interactive analytics,
geographic visualization, and LLM-generated natural language explanations for why transactions
are flagged — making the AI system transparent and actionable.

**Success Criteria:**

| Metric | Target |
|--------|--------|
| Page load time (LCP) | < 2.5 seconds |
| Real-time feed latency | < 3 seconds from scoring |
| Lighthouse Accessibility score | ≥ 90 |
| Dashboard usable without ML knowledge | Yes (LLM explains everything) |
| Mobile responsive | Yes (tablet + desktop priority) |

---

## 2. User Personas

### 2.1 Compliance Officer (Primary)
- Works at a bank or payment provider
- Monitors flagged transactions daily
- Needs to understand *why* something is flagged without reading ML metrics
- Escalates critical cases to investigation teams
- Uses desktop, occasionally tablet

### 2.2 Bank Analyst
- Runs periodic reports on fraud patterns
- Wants historical trends, geographic distribution, top offenders
- Exports data for internal reporting

### 2.3 Hackathon Judge (Demo)
- First-time viewer with no domain context
- Needs to understand the system's value within 5 minutes
- Impressed by: real-time feed, clear visualizations, LLM explanations

---

## 3. User Stories

### 3.1 Real-Time Monitoring
- **As a** compliance officer, **I want to** see a live feed of scored transactions as they come in,
  **so that** I can react to critical threats immediately.
  - **AC:** Feed updates within 3 seconds of scoring. Color-coded by risk level. Auto-scrolls.

- **As a** compliance officer, **I want to** filter the live feed by risk level, **so that** I can
  focus on Critical and High Risk transactions only.
  - **AC:** Toggle buttons for Normal / Suspicious / High Risk / Critical. Filters apply instantly.

### 3.2 Transaction Investigation
- **As a** compliance officer, **I want to** click on a flagged transaction and see its full detail,
  **so that** I can investigate the entity.
  - **AC:** Detail panel shows: all layer scores, merchant info, user info, amount, timestamp,
    geographic location, and risk level badge.

- **As a** compliance officer, **I want to** see an AI-generated explanation of why a transaction
  was flagged, **so that** I can understand the risk without ML expertise.
  - **AC:** "Explain" button triggers LLM call. Response displayed in plain Indonesian/English.
    Includes specific evidence (e.g., "89% round amounts", "312 unique senders").

### 3.3 Analytics & Reporting
- **As a** bank analyst, **I want to** see a summary dashboard with total transactions, flagged
  count, risk distribution, and trend lines, **so that** I can assess overall fraud exposure.
  - **AC:** Cards showing: total scored, flagged %, by risk level. Chart showing trend over time.

- **As a** bank analyst, **I want to** see a geographic heatmap of flagged transactions by
  province/city, **so that** I can identify regional fraud hotspots.
  - **AC:** Indonesia map with province-level heat coloring. Click province → city breakdown.

- **As a** bank analyst, **I want to** see top flagged merchants and users ranked by score,
  **so that** I can prioritize investigations.
  - **AC:** Sortable table. Columns: ID, score, transaction count, risk level, last flagged date.

### 3.4 Authentication
- **As a** user, **I want to** log in with my credentials, **so that** only authorized personnel
  access the dashboard.
  - **AC:** Login page with username/password. JWT stored securely. Auto-redirect on expiry.

---

## 4. Non-Goals

- **Not building:** Mobile-native app (responsive web is sufficient)
- **Not building:** Transaction blocking/approval actions (read-only dashboard for hackathon)
- **Not building:** User management UI (admin uses API directly)
- **Not building:** Multi-tenancy (single organization view for hackathon)
- **Not building:** Custom report builder (predefined analytics pages only)

---

## 5. Technical Architecture

### 5.1 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Next.js (App Router) | 15+ |
| Language | TypeScript | 5+ |
| UI Library | shadcn/ui + Tailwind CSS | Latest |
| Charts | Recharts | Latest |
| Maps | react-simple-maps or Leaflet | Latest |
| State Management | React Query (TanStack Query) | v5 |
| Real-Time | Polling (5s interval) or SSE | — |
| Auth | JWT (stored in httpOnly cookie) | — |
| HTTP Client | Axios or fetch | — |
| Icons | Lucide React | Latest |
| Package Manager | pnpm | Latest |

### 5.2 Directory Structure

```
dashboard/
├── src/
│   ├── app/
│   │   ├── layout.tsx              # Root layout (sidebar, auth guard)
│   │   ├── page.tsx                # Redirect to /dashboard
│   │   ├── login/
│   │   │   └── page.tsx            # Login page
│   │   ├── dashboard/
│   │   │   ├── page.tsx            # Overview / summary cards + charts
│   │   │   ├── live/
│   │   │   │   └── page.tsx        # Real-time scoring feed
│   │   │   ├── transactions/
│   │   │   │   ├── page.tsx        # Transaction list (filtered, paginated)
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx    # Transaction detail + LLM explain
│   │   │   ├── analytics/
│   │   │   │   └── page.tsx        # Charts, distributions, trends
│   │   │   ├── geographic/
│   │   │   │   └── page.tsx        # Indonesia map heatmap
│   │   │   ├── merchants/
│   │   │   │   └── page.tsx        # Top flagged merchants
│   │   │   └── models/
│   │   │       └── page.tsx        # Model status (admin)
│   │   └── globals.css
│   │
│   ├── components/
│   │   ├── ui/                     # shadcn/ui components
│   │   ├── layout/
│   │   │   ├── sidebar.tsx         # Navigation sidebar
│   │   │   ├── header.tsx          # Top bar with user info
│   │   │   └── auth-guard.tsx      # Redirect if not authenticated
│   │   ├── scoring/
│   │   │   ├── live-feed.tsx       # Real-time transaction feed
│   │   │   ├── score-badge.tsx     # Color-coded risk level badge
│   │   │   └── layer-breakdown.tsx # 6-layer score visualization
│   │   ├── analytics/
│   │   │   ├── summary-cards.tsx   # Total, flagged, by risk level
│   │   │   ├── trend-chart.tsx     # Time series of flagged tx
│   │   │   ├── risk-distribution.tsx # Score histogram
│   │   │   └── top-entities.tsx    # Top merchants/users table
│   │   ├── geographic/
│   │   │   └── indonesia-map.tsx   # Province-level heatmap
│   │   └── explain/
│   │       └── explanation-panel.tsx # LLM explanation display
│   │
│   ├── lib/
│   │   ├── api.ts                  # API client (Axios instance, interceptors)
│   │   ├── auth.ts                 # JWT handling, login/logout
│   │   └── utils.ts                # Formatters (Rupiah, dates, risk colors)
│   │
│   ├── hooks/
│   │   ├── use-live-feed.ts        # Polling hook for real-time feed
│   │   ├── use-transactions.ts     # React Query hook for transactions
│   │   └── use-analytics.ts        # React Query hook for analytics
│   │
│   └── types/
│       ├── transaction.ts          # Transaction, ScoreResult types
│       ├── analytics.ts            # Analytics response types
│       └── auth.ts                 # User, LoginResponse types
│
├── public/
│   └── indonesia-topo.json         # Indonesia TopoJSON for map
│
├── package.json
├── tailwind.config.ts
├── tsconfig.json
├── next.config.ts
├── Dockerfile
├── .env.example
└── PRD.md                          # This file
```

---

## 6. Pages & Components

### 6.1 Login Page (`/login`)
- Clean login form: username + password
- Error handling for invalid credentials
- Redirects to `/dashboard` on success
- Pantau logo + tagline

### 6.2 Overview Dashboard (`/dashboard`)
- **Summary Cards (top row):**
  - Total transactions scored
  - Flagged transactions (with % of total)
  - Breakdown by risk level (4 colored cards)
  - Average risk score
- **Trend Chart:** Line chart of flagged transactions over time (daily/weekly)
- **Risk Distribution:** Histogram of final scores (0-100)
- **Recent Critical:** Last 5 critical-risk transactions with quick details

### 6.3 Live Feed (`/dashboard/live`)
- Real-time scrolling feed of scored transactions
- Each row: timestamp, transaction ID, merchant, amount (formatted Rupiah), risk badge
- Filter toggles: Normal / Suspicious / High Risk / Critical
- Auto-scroll with pause button
- Click row → navigate to transaction detail
- Polling every 5 seconds (configurable)

### 6.4 Transaction List (`/dashboard/transactions`)
- Paginated table with server-side filtering
- Columns: timestamp, TX ID, user, merchant, amount, score, risk level
- Filters: risk level, date range, merchant ID, user ID, min/max score
- Sort by: timestamp, score, amount
- Click row → transaction detail page

### 6.5 Transaction Detail (`/dashboard/transactions/[id]`)
- Full transaction info card
- **Layer Score Breakdown:** Radar chart or bar chart showing all 6 layer scores
- **Risk Level Badge:** Large, color-coded
- **LLM Explanation Panel:**
  - "Explain This Transaction" button
  - Displays natural language explanation from LLM
  - Loading state while LLM generates response
  - Shows risk factors with per-layer reasoning
- **Related Transactions:** Other flagged tx from same merchant/user

### 6.6 Analytics (`/dashboard/analytics`)
- **Timeline:** Flagged transactions per day/week/month (line chart)
- **Risk Distribution:** Score histogram with threshold lines at 40/60/80
- **Top Flagged Merchants:** Table ranked by average score
- **Top Flagged Users:** Table ranked by average score
- **Layer Performance:** Which layers contribute most to flagging

### 6.7 Geographic View (`/dashboard/geographic`)
- Indonesia map colored by province-level risk density
- Click province → city-level breakdown table
- Data: count of flagged tx per province, average score per province
- Uses Indonesia TopoJSON (38 provinces)

### 6.8 Model Status (`/dashboard/models`) — Admin Only
- Current loaded model tag + version
- Model metrics: F1, AUC-ROC, PR-AUC, Precision, Recall
- Layer weights (visual bar chart)
- "Reload Models" button
- Model version history table

---

## 7. Design System

### 7.1 Risk Level Colors
| Level | Score Range | Color | Hex |
|-------|-----------|-------|-----|
| Normal | 0-40 | Green | `#22C55E` |
| Suspicious | 40-60 | Yellow | `#EAB308` |
| High Risk | 60-80 | Orange | `#F97316` |
| Critical | 80-100 | Red | `#EF4444` |

### 7.2 Theme
- Dark mode primary (professional monitoring tool aesthetic)
- Light mode toggle available
- Font: Inter (system default for shadcn/ui)
- Sidebar: dark background, icon + text navigation
- Content area: subtle gray background with white cards

### 7.3 Responsive Breakpoints
- Desktop (≥1280px): Full sidebar + content
- Tablet (≥768px): Collapsible sidebar
- Mobile (≥640px): Bottom navigation (minimal support)

---

## 8. API Integration

All API calls go through the backend at `NEXT_PUBLIC_API_URL` (default: `http://localhost:8000`).

| Dashboard Feature | Backend Endpoint |
|------------------|-----------------|
| Live feed | `GET /api/v1/transactions?sort_by=scored_at&order=desc&per_page=20` (polling) |
| Score single tx | `POST /api/v1/score` |
| Transaction list | `GET /api/v1/transactions` (with filters) |
| Transaction detail | `GET /api/v1/transactions/{id}` |
| LLM explanation | `POST /api/v1/explain/{transaction_id}` |
| Summary cards | `GET /api/v1/analytics/summary` |
| Risk distribution | `GET /api/v1/analytics/risk-distribution` |
| Top merchants | `GET /api/v1/analytics/top-merchants` |
| Top users | `GET /api/v1/analytics/top-users` |
| Timeline | `GET /api/v1/analytics/timeline` |
| Geographic | `GET /api/v1/analytics/geographic` |
| Model status | `GET /api/v1/models/status` |
| Model reload | `POST /api/v1/models/reload` |
| Login | `POST /api/v1/auth/login` |

---

## 9. Environment Variables

```env
# API
NEXT_PUBLIC_API_URL=http://localhost:8000

# Polling
NEXT_PUBLIC_LIVE_FEED_INTERVAL=5000

# App
NEXT_PUBLIC_APP_NAME=Pantau
NEXT_PUBLIC_APP_DESCRIPTION=AI Fraud Detection for QRIS
```

---

## 10. Non-Functional Requirements

| Requirement | Target |
|------------|--------|
| Largest Contentful Paint (LCP) | < 2.5s |
| First Input Delay (FID) | < 100ms |
| Cumulative Layout Shift (CLS) | < 0.1 |
| Lighthouse Performance | ≥ 80 |
| Lighthouse Accessibility | ≥ 90 |
| Bundle size (initial JS) | < 200KB gzipped |
| Browser support | Chrome, Firefox, Edge (latest 2 versions) |
| Concurrent users | 50+ |

---

## 11. Milestones

| Phase | Scope |
|-------|-------|
| **Phase 1: Foundation** | Next.js scaffold, Tailwind + shadcn/ui, layout (sidebar, header), login page |
| **Phase 2: Overview** | Dashboard overview page with summary cards, trend chart, risk distribution |
| **Phase 3: Live Feed** | Real-time scoring feed with polling, risk level filters |
| **Phase 4: Transactions** | Transaction list (paginated, filtered), transaction detail page |
| **Phase 5: Explainability** | LLM explanation panel on transaction detail, risk factor breakdown |
| **Phase 6: Analytics** | Analytics page with timeline, top entities, layer performance |
| **Phase 7: Geographic** | Indonesia map heatmap with province/city drill-down |
| **Phase 8: Polish** | Dark/light theme, responsive fixes, loading states, Dockerfile |

---

*PRD for Pantau Dashboard — PIDI DIGDAYA X Hackathon 2026*
