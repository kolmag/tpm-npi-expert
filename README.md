# TPM / NPI Expert Knowledge Worker

A production-grade RAG (Retrieval-Augmented Generation) application that functions as a senior Technical Program Manager and NPI expert — answering domain-specific questions across phase gate methodology, APQP/PPAP, RAID management, SAFe, hybrid program models, supplier readiness, and program metrics across six industries.

Built as the second application in an AI engineering portfolio series, extending the architecture established in the [CAPA/8D Expert](https://github.com/kolmag/capa-8d-expert) with a richer knowledge base, multi-industry coverage, and an integrated PM dashboard layer.

---

## Demo

**Expert Q&A tab** — ask any TPM/NPI question grounded in the knowledge base:

> *"Our Tier-2 seal supplier failed DVT sample delivery — how does this affect Gate 3 and what should the RAID log entry look like?"*

> *"What APQP artifacts are required at Gate 1 for an automotive program and who owns each?"*

> *"We are building a connected medical device with embedded firmware and a cloud SaaS backend — how do we align the hardware phase gate timeline with the CI/CD pipeline?"*

**PM Dashboard tab** — select one of six industry programs (automotive, data center, defense, semiconductor, medical, high-tech), run a phase gate analysis, and ask context-aware questions grounded in live program task data. Asana MCP-ready — swap mock data for live API calls via the `mcp_servers` parameter.

---

## Evaluation Results

Evaluated on an 80-question test set spanning 10 categories, three difficulty levels, and three question sources (internal, blind external, adversarial).

| Metric | Score |
|---|---|
| MRR@10 (retrieval quality) | **0.973** |
| Judge Overall — internal (50 q) | **7.84 / 10** |
| Judge Overall — blind external (20 q) | **7.22 / 10** |
| Judge Correctness | 7.67 / 10 |
| Judge Completeness | 6.82 / 10 |
| Judge Groundedness | 7.31 / 10 |

**By category:**

| Category | MRR@10 | Overall |
|---|---|---|
| RAID & escalation | 1.000 | 8.00 |
| Hybrid program models | 1.000 | 8.00 |
| Stakeholder management | 1.000 | 8.00 |
| Metrics & KPIs | 0.933 | 7.78 |
| APQP / PPAP | 0.864 | 7.43 |
| NPI risk / DFM / DFT | 1.000 | 7.25 |
| Agile / SAFe | 1.000 | 7.00 |
| Software delivery | 1.000 | 6.86 |
| NPI phase gate | 1.000 | 6.75 |
| TPM planning | 1.000 | 5.50 |

**Eval methodology:** MRR@10 measures whether the correct source document appears in the top 10 retrieved chunks. LLM-as-judge uses GPT-4o-mini scoring correctness, completeness, and groundedness on a 1–10 rubric. Blind external questions were generated independently by Gemini and ChatGPT without knowledge of the KB content. Adversarial questions probe genuine KB gaps (Digital Twin, CCPM, Line of Balance, ISO 26262 ADAS integration) — the system is expected to answer partially or acknowledge gaps on these.

Internal vs blind gap of 0.62 points demonstrates the KB generalises to real practitioner questions rather than being tuned to its own content.

---

## Architecture

```
User question
     │
     ▼
Query Rewriting ── Claude Haiku 4.5
     │              3 alternative phrasings with domain vocabulary injection
     │              (phase gate, APQP, PPAP, RAID, DFMEA, CSR, LLC, EVM...)
     ▼
Retrieval ── text-embedding-3-small + Chroma
     │        top-20 chunks per query × 4 queries → union + deduplicate
     │        ~80 candidate chunks
     ▼
Reranking ── BAAI/bge-reranker-v2-m3 (local, CUDA/MPS/CPU)
     │        cross-encoder scores each (question, chunk) pair
     │        returns top-15 by relevance
     ▼
Answer Generation ── GPT-4o-mini
     │                15 chunks as context, groundedness-constrained prompt
     ▼
Groundedness Check ── Claude Sonnet 4.6
                       scores answer against retrieved context (0.0–1.0)
```

**Model selection rationale:**

| Task | Model | Reason |
|---|---|---|
| Query rewriting | Claude Haiku 4.5 | Structured JSON, high-volume, cheap ($1/$5 per MTok) |
| Chunk enrichment | Claude Haiku 4.5 | Runs once at ingest — headline + summary per chunk |
| Answer generation | GPT-4o-mini | Reliable context-grounded Q&A, no rate limits ($0.15/$0.60 per MTok) |
| Groundedness check | Claude Sonnet 4.6 | Reliable structured output, catches hallucination ($3/$15 per MTok) |
| Eval judge | GPT-4o-mini | Structured rubric scoring, no rate limits |
| Embeddings | text-embedding-3-small | Best price/quality for retrieval ($0.02/MTok) |
| BGE reranker | bge-reranker-v2-m3 | Local model — free, best precision on domain-specific ranking |

**Cost per user query: ~$0.006** (under one cent)

---

## Knowledge Base

12 enriched markdown documents, 50,704 words, 232 chunks after ingest.

Each document follows the enrichment pattern: methodology reference → worked examples → decision guides → common mistakes → timing/compliance norms → standards alignment.

| Doc | Title | Key coverage |
|---|---|---|
| 01 | NPI Phase Gate Process | Phase 0–5, gate criteria, APQP mapping, LLC/CSR lifecycle, defense/aerospace, Agile within gates |
| 02 | NPI Risk — DFM, DFT, DFR | Design for manufacturability, testability, reliability; DFMEA integration; 10x rule |
| 03 | Supplier Readiness & PPAP/APQP | Supplier tiers, APQP Phase 1–5, all 18 PPAP elements, Gate 1 checklist, rejection prevention |
| 04 | TPM Core Framework | Charter, WBS, schedule baseline, CPM, resource loading, change control, status reporting |
| 05 | RAID Log & Escalation | Risk/Issue/Assumption/Dependency structure, mitigation vs contingency, escalation ladders, ROAM |
| 06 | Agile/Scrum for Hardware | Sprint cadence on physical products, hardware spikes, ESI, DoD per track, cybersecurity in Agile |
| 07 | SAFe TPM Application | ART structure, PI Planning, PI Objectives, Program Board, CDR/PDR alignment, I&A feedback loops |
| 08 | Waterfall / PMBOK | WBS decomposition, CPM, EVM (SPI/CPI/EAC/TCPI), CCB, procurement, program closure |
| 09 | Hybrid Program Models | Sequential vs parallel hybrid, transition points, change request boundaries, feedback loops |
| 10 | Stakeholder Management | Stakeholder register, RACI (single-A rule), communication plan, executive dashboard, cross-cultural |
| 11 | NPI/TPM Metrics & KPIs | SPI/CPI thresholds, contingency triggers, yield metrics, OKRs vs KPIs, industry dashboards |
| 12 | Software-Only Delivery | SaaS/mobile/cloud lifecycle, CI/CD governance, DORA metrics, SOC2, FDA SaMD, cross-domain |

**Industries covered:** Automotive (IATF 16949, APQP, PPAP), Defense (MIL-STD-1521, ITAR, CDR/PDR), Semiconductor (JEDEC, yield ramp, Cpk), Medical device (ISO 13485, FDA 21 CFR 820, DHF), Data center (Tier IV, MEP, NEC4), High-tech / SoC (SAFe, CoWoS packaging, AI inference)

**Standards covered:** IATF 16949, ISO 9001:2015, ISO 13485, AS9100D/AS9145, MIL-STD-1521, MIL-HDBK-881, ANSI/EIA-748 (EVMS), AIAG PPAP 4th Ed., AIAG-VDA FMEA, PMI PMBOK 7, SAFe 6.0, IEC 62304, ISO/SAE 21434, CMMC 2.0, SOC2, FDA SaMD guidance

---

## Key Engineering Decisions

**LLM enrichment at ingest time** is the single biggest retrieval quality lever. Claude Haiku generates a headline and 2–3 sentence summary for each chunk at ingest. The embedded text is `headline + summary + original_text` — not just raw text. This creates semantically rich embeddings that dramatically improve cosine similarity recall before BGE reranking.

**Multi-query retrieval** runs 4 queries (original + 3 rewrites) and unions the results before reranking. This recovers vocabulary mismatches between how users phrase questions and how the KB was written.

**Two-stage retrieval** separates coarse retrieval (bi-encoder cosine similarity, fast, ~80 candidates) from precise ranking (cross-encoder BGE, slower, top-15). This is the standard production pattern — cosine similarity alone produces MRR ~0.70–0.80; adding BGE pushes it to 0.97+.

**KB architecture for cross-document retrieval** — documents are written with explicit cross-references so the retriever can surface multi-doc answers. The hardest questions (supplier failure affecting gate + RAID log) require chunks from Doc 03, Doc 05, and Doc 01 simultaneously. MRR 0.973 on these confirms the approach works.

**Batch mode** loads BGE and Chroma once and reuses them across all questions — critical for eval runs and smoke tests on memory-constrained hardware.

**Colab-ready** — `CHROMA_DIR` is overridable via env var, BGE auto-detects CUDA/MPS/CPU. Full eval on 80 questions completes in ~8 minutes on a free T4 GPU.

---

## PM Dashboard

The PM Dashboard tab demonstrates agentic integration — it reads live program task data, injects it as context into the RAG pipeline, and returns phase gate analysis grounded in both the knowledge base and the program's actual task state.

Six mock programs at different phases across six industries. The mock data mirrors Asana's task structure — swapping in the Asana MCP server is a one-function change:

```python
# Current: mock data
tasks = TASKS[project_id]

# With Asana MCP (one change):
result = await anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    mcp_servers=[{"type": "url", "url": "https://mcp.asana.com/sse"}],
    messages=[{"role": "user", "content": f"Get tasks for project {project_gid}"}]
)
```

---

## Project Structure

```
tpm_npi_expert/
├── data/
│   └── knowledge_base/          # 12 enriched .md documents
├── scripts/
│   ├── ingest.py                # Chunk → enrich (Haiku) → embed → Chroma
│   ├── answer.py                # Full RAG pipeline + batch mode
│   └── app.py                   # Gradio UI (two tabs)
├── evaluation/
│   ├── eval.py                  # MRR@10 + LLM-as-judge pipeline
│   └── tests.jsonl              # 80-question test set (50 internal + 30 external/adversarial)
├── chroma_db/                   # Local vector store (gitignored)
├── smoke_tests.txt              # 13-question smoke test suite
├── pyproject.toml
└── .env                         # API keys (gitignored)
```

---

## Setup

**Prerequisites:** Python 3.11+, uv, API keys for Anthropic and OpenAI.

```bash
# Clone
git clone https://github.com/kolmag/tpm-npi-expert
cd tpm-npi-expert

# Install
uv sync

# Configure
cp .env.example .env
# Add ANTHROPIC_API_KEY and OPENAI_API_KEY to .env

# Build knowledge base (first time)
uv run scripts/ingest.py --reset

# Run app
uv run scripts/app.py
# → http://localhost:7860
```

**Ask a question via CLI:**
```bash
uv run scripts/answer.py "What APQP artifacts are required at Gate 1 for an automotive program?"
uv run scripts/answer.py --debug "Our SPI is 0.82 — what are the recovery actions?"
uv run scripts/answer.py --batch smoke_tests.txt
```

**Run evaluation:**
```bash
# Local (CPU BGE — ~45 min for 80 questions)
uv run evaluation/eval.py --tests evaluation/tests.jsonl

# Sample run for quick validation
uv run evaluation/eval.py --tests evaluation/tests.jsonl --sample 20 --no-judge

# On Google Colab (T4 GPU — ~8 min for 80 questions)
# See COLAB_SETUP.md for instructions
```

---

## Hardware Notes

BGE reranker (`bge-reranker-v2-m3`) requires ~1.5GB of GPU/CPU memory.

| Hardware | BGE device | Per-question time | Recommended for |
|---|---|---|---|
| Apple M1 8GB | CPU (force via env) | 30–50s | App demo (loads once) |
| Apple M2/M3 16GB+ | MPS | 2–5s | Development |
| Google Colab T4 | CUDA | 1–2s | Eval runs |
| Linux + A100 | CUDA | <1s | Production |

For constrained hardware, use `--reranker llm` to bypass BGE entirely and use the Claude Haiku LLM reranker fallback.

```bash
uv run scripts/answer.py --reranker llm "your question"
```

---

## Comparison with CAPA/8D Expert (App 1)

| Dimension | CAPA/8D Expert | TPM/NPI Expert |
|---|---|---|
| KB size | 12 docs, ~22K words, 96 chunks | 12 docs, ~51K words, 232 chunks |
| Domain breadth | Single domain (quality/8D) | 6 industries, 3 methodologies |
| MRR@10 | 0.947 | **0.973** |
| Judge Overall | 8.47 / 10 | 7.84 / 10 (internal) |
| Blind test | 7.55 / 10 | **7.22 / 10** |
| Cross-doc retrieval | Moderate | High — intentionally designed |
| UI | Single tab Q&A | Two tabs: Q&A + PM Dashboard |
| Integration layer | None | Asana MCP-ready PM dashboard |
| Eval set | 197 questions | 80 questions (50 + 20 blind + 10 adversarial) |

The TPM/NPI Expert has a harder retrieval problem (6× more cross-document questions) and still achieves higher MRR. The lower judge overall score reflects the adversarial test set — questions about content genuinely not in the KB (CCPM, Digital Twin, Line of Balance). This is the correct behaviour: the system answers from context or acknowledges gaps rather than hallucinating.

---

## Roadmap

**v1.1 — KB enrichment:**
- Doc 04 (TPM planning): add TCPI recovery and hidden dependency worked examples
- Doc 12 (software): add ISO 26262 ADAS firmware integration and OSS license compliance
- Target: `tpm_planning` overall from 5.50 → 7.00+

**v1.2 — Live Asana integration:**
- Connect Asana MCP server via `mcp_servers` parameter
- Replace mock program data with live project tasks
- Add OAuth flow for multi-user support

**v1.3 — Fine-tuning experiment:**
- Generate (question, chunk) pairs from the KB using GPT-4o
- Fine-tune `text-embedding-3-small` on domain-specific retrieval pairs
- Target: MRR improvement from 0.973 → 0.985+

**v2.0 — Agentic expert worker:**
- ISO 9001 / Audit Preparation Expert (App 3 in portfolio series)
- Full agentic loop: read program documents → identify gaps → generate corrective actions → update RAID log

---

## Portfolio Context

This is the second application in an AI engineering portfolio series demonstrating end-to-end RAG system design:

- **App 1:** [CAPA/8D Expert](https://github.com/kolmag/capa-8d-expert) — quality engineering domain, 197-question eval, MRR 0.947
- **App 2:** TPM/NPI Expert — this repository
- **App 3:** Auditor Expert *(planned)*
- **App 4:** SQE Expert *(planned)*
- **App 5:** ISO 9001 Agentic Expert Worker *(planned)*

All apps share the same pipeline architecture. The knowledge base design and domain enrichment methodology are the primary differentiators — demonstrating that RAG quality is as much a knowledge engineering problem as a machine learning problem.

---

## Author

**Magdalena Koleva** — NPI/TPM and quality engineering professional transitioning into AI engineering.

Domain expertise: CAPA, 8D, NPI, TPM, Agile/SAFe, APQP/PPAP, auditing, SQE — across automotive, data center, high-tech, defense, semiconductor, and medical/biomedical industries.

GitHub: [github.com/kolmag](https://github.com/kolmag)

---

## License

MIT
