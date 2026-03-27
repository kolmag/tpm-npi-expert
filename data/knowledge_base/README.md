# Knowledge Base

The knowledge base documents are not included in this repository. They represent proprietary domain content built specifically for this application.

## Structure

The knowledge base consists of 12 enriched markdown documents (~51,000 words total) covering:

| Doc | Topic | ~Words |
|---|---|---|
| 01_npi_phase_gate_process.md | NPI Phase Gate methodology (Phase 0–5), APQP mapping, LLC/CSR lifecycle | 5,700 |
| 02_npi_risk_dfm_dft.md | DFM, DFT, DFR analysis — tools, timing, integration with DFMEA | 3,500 |
| 03_npi_supplier_ppap_apqp.md | Supplier qualification, all 18 PPAP elements, Gate 1 checklist | 3,800 |
| 04_tpm_core_framework.md | Charter, WBS, schedule baseline, CPM, EVM, change control | 4,600 |
| 05_tpm_risk_issue_management.md | RAID log structure, escalation ladders, mitigation vs contingency | 3,700 |
| 06_agile_scrum_hardware.md | Sprint cadence on hardware, hardware spikes, ESI, cybersecurity | 4,000 |
| 07_safe_tpm_application.md | SAFe ART structure, PI Planning, CDR/PDR, feedback loops | 3,800 |
| 08_waterfall_pmbok.md | WBS decomposition, CPM, EVM (SPI/CPI/TCPI), CCB, program closure | 3,800 |
| 09_hybrid_program_models.md | Hybrid Agile/waterfall, transition points, change request boundaries | 4,100 |
| 10_stakeholder_management.md | Stakeholder register, RACI, communication plan, escalation | 3,900 |
| 11_npi_tpm_metrics_kpis.md | SPI/CPI thresholds, yield metrics, OKRs vs KPIs, dashboards | 4,800 |
| 12_software_only_delivery.md | SaaS/CI/CD lifecycle, DORA metrics, SOC2, FDA SaMD, cross-domain | 5,000 |

## Document Enrichment Pattern

Each document follows this structure — the same pattern used across all apps in this portfolio series:

```
## Overview — when to use / what this covers
## [Section 1]
  - Objective
  - Key Artifacts
  - What good looks like (with worked example)
  - Common mistakes
  - Timing norms
## [Section N...]
## Decision guide
## Standards and compliance alignment
```

Worked examples span all six industries: automotive (IATF 16949), defense (MIL-STD-1521), semiconductor (JEDEC), medical device (ISO 13485/FDA), data center (Tier IV), and high-tech/SoC (SAFe).

## Building Your Own Knowledge Base

To adapt this application to your domain:

1. Create markdown files following the enrichment pattern above
2. Place them in `data/knowledge_base/`
3. Run ingest to build the vector store:

```bash
uv run scripts/ingest.py --reset
```

4. Run a smoke test to validate retrieval:

```bash
uv run scripts/answer.py --debug "your domain-specific question"
```

5. Build an eval test set and measure MRR@10:

```bash
uv run evaluation/eval.py --tests evaluation/tests.jsonl --sample 20
```

## Minimum viable KB

For the pipeline to work well, each document should be:
- **500–800 tokens per logical section** — aligns with the chunking parameters (`chunk_size=500`, `overlap=200`)
- **Rich in domain vocabulary** — the query rewriter injects domain terms; documents should use the same terminology
- **Example-driven** — worked examples are the highest-value content for RAG; they create specific, retrievable chunks
- **Cross-referenced** — explicit references between documents (e.g. "see Doc 05 for RAID escalation") help the retriever surface multi-doc answers

## Coverage in this application

Industries: Automotive · Defense · Semiconductor · Medical device · Data center · High-tech/SoC

Standards: IATF 16949 · ISO 9001:2015 · ISO 13485 · AS9100D · MIL-STD-1521 · MIL-HDBK-881 · ANSI/EIA-748 · AIAG PPAP 4th Ed. · AIAG-VDA FMEA · PMI PMBOK 7 · SAFe 6.0 · IEC 62304 · ISO/SAE 21434 · CMMC 2.0 · SOC2 · FDA SaMD guidance
