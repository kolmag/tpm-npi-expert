"""
app.py — TPM/NPI Expert Knowledge Worker
Two tabs: Expert Q&A | PM Dashboard
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from answer import answer, get_collection, get_bge

load_dotenv()

# ── Pre-load heavy resources once at startup ──────────────────────────────────
print("Loading Chroma collection…")
COLLECTION = get_collection()
print("Loading BGE reranker…")
get_bge()
print("Ready.")

# ── Example questions (8 industries / topics) ─────────────────────────────────
EXAMPLES = [
    "What APQP artifacts are required at Gate 1 for an automotive program and who owns each?",
    "What is the difference between a CDR and a PDR and how do they align to SAFe PI boundaries in a defense program?",
    "Our PVT first-pass yield is 91% and Cpk on a Special Characteristic is 1.52 — what are the Gate 4 implications and recovery actions?",
    "We are building a connected medical device — what DHF deliverables are required at each phase gate and what triggers an FDA submission?",
    "How do PI Objectives map to phase gate exit criteria in a SAFe hardware NPI program?",
    "What are the three transition points in a hybrid Agile/phase-gate program and what triggers a CCB decision vs. a sprint-level change?",
    "We are building a connected medical device with embedded firmware and a cloud SaaS backend — how do we align the hardware phase gate timeline with the software CI/CD pipeline?",
    "Our Tier-2 supplier failed DVT sample delivery — how does this affect Gate 3, what goes in the RAID log, and when do we escalate?",
]

# ── Mock project data (Asana MCP-ready) ──────────────────────────────────────
PROJECTS = [
    {"id":"p1","ind":"Automotive",   "color":"#BA7517","name":"BrakeSense X — EV Brake-by-Wire Module",    "phase":"Phase 3 — DVT",        "framework":"APQP / IATF 16949",  "csr":"Stellantis SQ MS 10081","target":"2025-Q4","team":14,"risk":"High"},
    {"id":"p2","ind":"Data Center",  "color":"#185FA5","name":"Hyperscale DC — Frankfurt Tier IV Build",   "phase":"Phase 2 — Civil & MEP", "framework":"Waterfall / PMBoK",  "csr":"Uptimo SLA v2.3",       "target":"2026-Q1","team":22,"risk":"Critical"},
    {"id":"p3","ind":"High-tech",    "color":"#534AB7","name":"NovaSoC — AI Edge Inference Platform NPI", "phase":"Phase 2 — EVT",        "framework":"SAFe PI 4 / Hybrid", "csr":"OEM-A Supply Agreement","target":"2025-Q3","team":11,"risk":"High"},
    {"id":"p4","ind":"Defense",      "color":"#5F5E5A","name":"SkyShield — UAV EW Payload CDR",           "phase":"Phase CDR — Defense",  "framework":"MIL-STD-1521 / CDR", "csr":"CDRL A001–A018 / ITAR", "target":"2026-Q3","team":18,"risk":"Critical"},
    {"id":"p5","ind":"Semiconductor","color":"#0F6E56","name":"PrimeFab 3nm — Process Qual NPI",          "phase":"Phase 4 — PVT / PPAP", "framework":"APQP / JEDEC",       "csr":"TSMC PDK 3N CSR v1.1",  "target":"2025-Q2","team":9, "risk":"Medium"},
    {"id":"p6","ind":"Medical",      "color":"#993556","name":"ClearVein — Intravascular Imaging System", "phase":"Phase 1 — Definition", "framework":"ISO 13485 / FDA DHF","csr":"FDA 21 CFR 820 / MDR",  "target":"2026-Q2","team":8, "risk":"Medium"},
]

TASKS = {
    "p1":[
        {"name":"DVT build — 30 units at CM (production-intent BOM)","status":"in_progress","assignee":"Piotr K.","due":"2025-03-28","priority":"high"},
        {"name":"DFMEA final — brake actuator subsystem","status":"in_progress","assignee":"Sven L.","due":"2025-03-30","priority":"high"},
        {"name":"HALT/HASS test plan — −40°C to +125°C","status":"not_started","assignee":"Unassigned","due":"2025-04-08","priority":"high"},
        {"name":"PFMEA initiation — assembly line operations","status":"not_started","assignee":"Unassigned","due":"2025-04-10","priority":"high"},
        {"name":"CE / UN ECE R13 regulatory submission","status":"not_started","assignee":"Unassigned","due":"2025-04-20","priority":"high"},
        {"name":"Stellantis CSR gap analysis — SQ MS 10081","status":"in_progress","assignee":"Agnieszka W.","due":"2025-04-05","priority":"high"},
        {"name":"DVP&R update — traceability to SRS v2.1","status":"in_progress","assignee":"Sven L.","due":"2025-04-01","priority":"medium"},
        {"name":"Supplier qualification — Tier-1 seal supplier","status":"complete","assignee":"Dana L.","due":"2025-03-15","priority":"medium"},
        {"name":"LLC pull — prior EV brake NPI programs","status":"complete","assignee":"Magdalena W.","due":"2025-03-10","priority":"medium"},
        {"name":"Gate 3 review — preparation and booking","status":"not_started","assignee":"Magdalena W.","due":"2025-04-25","priority":"medium"},
    ],
    "p2":[
        {"name":"Civil foundation — pour schedule confirmed","status":"complete","assignee":"Erik B.","due":"2025-03-01","priority":"high"},
        {"name":"MEP design freeze — HVAC redundancy N+2","status":"in_progress","assignee":"Elena R.","due":"2025-04-01","priority":"high"},
        {"name":"UPS & generator spec — 2N architecture sign-off","status":"in_progress","assignee":"Unassigned","due":"2025-04-05","priority":"high"},
        {"name":"Critical path update — MEP lead-time slip +3 weeks","status":"in_progress","assignee":"Magdalena W.","due":"2025-03-28","priority":"high"},
        {"name":"Uptimo SLA compliance matrix — cooling latency req.","status":"not_started","assignee":"Unassigned","due":"2025-04-10","priority":"high"},
        {"name":"Risk register update — power density change impact","status":"not_started","assignee":"Magdalena W.","due":"2025-04-08","priority":"high"},
        {"name":"Commissioning test plan — Tier IV proof of concept","status":"not_started","assignee":"Unassigned","due":"2025-04-15","priority":"high"},
        {"name":"Change order — raised floor spec revision","status":"in_progress","assignee":"Erik B.","due":"2025-03-30","priority":"medium"},
        {"name":"LLC pull — Frankfurt DC Phase 1 lessons","status":"complete","assignee":"Magdalena W.","due":"2025-03-05","priority":"medium"},
    ],
    "p3":[
        {"name":"EVT silicon bring-up — 8 units from fab","status":"in_progress","assignee":"Omar S.","due":"2025-03-30","priority":"high"},
        {"name":"PI 4 planning — EVT sprint objectives vs Gate 2","status":"complete","assignee":"Magdalena W.","due":"2025-03-15","priority":"high"},
        {"name":"DFMEA — inference engine thermal failure modes","status":"in_progress","assignee":"Lena F.","due":"2025-04-05","priority":"high"},
        {"name":"DFM review — substrate packaging tolerances","status":"not_started","assignee":"Unassigned","due":"2025-04-10","priority":"high"},
        {"name":"OEM-A CSR compliance — supply agreement Annex C","status":"not_started","assignee":"Unassigned","due":"2025-04-12","priority":"high"},
        {"name":"Long-lead: CoWoS packaging slot — TSMC confirmed","status":"complete","assignee":"Dana L.","due":"2025-03-10","priority":"high"},
        {"name":"EVT test plan — power at inferencing load","status":"in_progress","assignee":"Omar S.","due":"2025-04-02","priority":"medium"},
        {"name":"SAFe system demo — EVT milestone readout","status":"not_started","assignee":"Magdalena W.","due":"2025-04-18","priority":"medium"},
    ],
    "p4":[
        {"name":"CDR package — all 18 CDRL items A001–A018","status":"in_progress","assignee":"Col. Reyes","due":"2025-04-10","priority":"high"},
        {"name":"FMECA — EW payload RF chain failure modes","status":"in_progress","assignee":"Dr. Hartmann","due":"2025-04-08","priority":"high"},
        {"name":"ITAR compliance check — all foreign national access","status":"in_progress","assignee":"Legal / DCSA","due":"2025-04-01","priority":"high"},
        {"name":"MIL-STD-810 test plan — shock, vibe, EMI","status":"not_started","assignee":"Unassigned","due":"2025-04-15","priority":"high"},
        {"name":"Reliability prediction — MIL-HDBK-217F","status":"not_started","assignee":"Unassigned","due":"2025-04-12","priority":"high"},
        {"name":"Government CDR review — site preparation","status":"not_started","assignee":"Magdalena W.","due":"2025-04-20","priority":"high"},
        {"name":"LLC pull — prior EW payload CDRs (classified)","status":"complete","assignee":"Dr. Hartmann","due":"2025-03-12","priority":"medium"},
        {"name":"Cybersecurity CMMC Level 2 audit prep","status":"not_started","assignee":"Unassigned","due":"2025-04-18","priority":"medium"},
        {"name":"Supply chain mapping — ITAR-restricted components","status":"in_progress","assignee":"Dana L.","due":"2025-04-05","priority":"high"},
    ],
    "p5":[
        {"name":"PVT pilot — 500 wafer starts at 3nm node","status":"in_progress","assignee":"Dr. Chen","due":"2025-03-28","priority":"high"},
        {"name":"Cpk study — 14 CTQ parameters (target ≥1.67)","status":"in_progress","assignee":"Sofia B.","due":"2025-03-30","priority":"high"},
        {"name":"PPAP Level 3 — JEDEC JESD47 compliance package","status":"not_started","assignee":"Unassigned","due":"2025-04-05","priority":"high"},
        {"name":"MSA — critical implant dose uniformity","status":"complete","assignee":"Sofia B.","due":"2025-03-20","priority":"high"},
        {"name":"TSMC PDK 3N CSR — customer-specific process controls","status":"in_progress","assignee":"Dr. Chen","due":"2025-04-02","priority":"high"},
        {"name":"Control Plan — all 22 critical process steps","status":"not_started","assignee":"Unassigned","due":"2025-04-08","priority":"high"},
        {"name":"Yield ramp model — 85% target by week 6","status":"in_progress","assignee":"Raj M.","due":"2025-04-01","priority":"medium"},
        {"name":"Gate 4 review — preparation","status":"not_started","assignee":"Magdalena W.","due":"2025-04-12","priority":"medium"},
    ],
    "p6":[
        {"name":"Device History File (DHF) — structure and ownership","status":"in_progress","assignee":"Dr. Varga","due":"2025-04-15","priority":"high"},
        {"name":"Design inputs — IEC 60601-1 / ISO 14971 risk mgmt","status":"in_progress","assignee":"Dr. Varga","due":"2025-04-20","priority":"high"},
        {"name":"FDA 21 CFR 820 design controls plan","status":"not_started","assignee":"Unassigned","due":"2025-04-25","priority":"high"},
        {"name":"EU MDR Article 10 compliance checklist","status":"not_started","assignee":"Unassigned","due":"2025-04-28","priority":"high"},
        {"name":"Preliminary DFMEA — catheter tip imaging module","status":"not_started","assignee":"Unassigned","due":"2025-05-02","priority":"high"},
        {"name":"Biocompatibility assessment — ISO 10993 materials","status":"in_progress","assignee":"Dr. Varga","due":"2025-04-18","priority":"medium"},
        {"name":"LLC pull — prior catheter NPI programs","status":"complete","assignee":"Magdalena W.","due":"2025-03-20","priority":"medium"},
        {"name":"Program schedule baseline — Phase 1","status":"not_started","assignee":"Magdalena W.","due":"2025-04-22","priority":"high"},
        {"name":"Clinical advisory board — imaging spec review","status":"in_progress","assignee":"Dr. Varga","due":"2025-04-30","priority":"medium"},
    ],
}

# ── RAG answer function (calls the pipeline) ─────────────────────────────────
def rag_answer(question: str, history: list, use_rewrite: bool, reranker: str):
    """Called by the Expert Q&A tab chat."""
    if not question.strip():
        return history, "", []

    hist_fmt = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        result = answer(
            question,
            use_rewrite   = use_rewrite,
            reranker_mode = reranker,
            history       = hist_fmt,
            collection    = COLLECTION,
        )
    except Exception as e:
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, "", []

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": result.answer})

    # Build source panel data
    sources = []
    for c in result.ranked_chunks[:6]:
        sources.append({
            "doc":      c["doc_name"].replace("_", " "),
            "headline": c["headline"],
            "preview":  c["text"][:280] + "…",
            "score":    round(c.get("rerank_score", 0), 3),
        })

    source_md = f"**Reranker:** {result.reranker_used}  |  "
    source_md += f"**Groundedness:** {result.checker_score or '—'}  |  "
    source_md += f"**Sources:** {', '.join(result.sources)}\n\n"
    source_md += "---\n\n"
    for i, s in enumerate(sources, 1):
        source_md += f"**{i}. {s['doc']}** · score {s['score']}\n\n"
        source_md += f"*{s['headline']}*\n\n"
        source_md += f"{s['preview']}\n\n---\n\n"

    rewrites_md = "\n".join(f"- {q}" for q in result.rewritten_queries[1:])

    return history, rewrites_md, source_md


def use_example(example: str):
    return example


# ── PM Dashboard helpers ──────────────────────────────────────────────────────
def get_project_info(proj_id: str) -> str:
    p = next((x for x in PROJECTS if x["id"] == proj_id), None)
    if not p:
        return ""
    tasks = TASKS.get(proj_id, [])
    total    = len(tasks)
    done     = sum(1 for t in tasks if t["status"] == "complete")
    inprog   = sum(1 for t in tasks if t["status"] == "in_progress")
    unassign = sum(1 for t in tasks if t["assignee"] == "Unassigned")

    md  = f"## {p['name']}\n\n"
    md += f"**Industry:** {p['ind']}  |  **Phase:** {p['phase']}  |  **Risk:** {p['risk']}\n\n"
    md += f"**Framework:** {p['framework']}  |  **CSR:** {p['csr']}  |  **Target:** {p['target']}  |  **Team:** {p['team']}\n\n"
    md += f"**Tasks:** {total} total · {done} done · {inprog} in progress"
    if unassign:
        md += f" · ⚠ {unassign} unassigned"
    md += "\n\n---\n\n"
    md += "| Status | Task | Assignee | Due | Priority |\n"
    md += "|--------|------|----------|-----|----------|\n"
    STATUS_ICON = {"complete": "✅", "in_progress": "🔵", "not_started": "⬜"}
    for t in tasks:
        icon  = STATUS_ICON.get(t["status"], "⬜")
        warn  = " ⚠" if t["assignee"] == "Unassigned" else ""
        prio  = "🔴" if t["priority"] == "high" else "🟡"
        md += f"| {icon} | {t['name']} | {t['assignee']}{warn} | {t['due']} | {prio} |\n"
    return md


def run_dashboard_analysis(proj_id: str) -> str:
    p = next((x for x in PROJECTS if x["id"] == proj_id), None)
    if not p:
        return "Select a project first."
    tasks = TASKS.get(proj_id, [])

    task_lines = "\n".join(
        f"- [{t['status'].upper()}] {t['name']} | Due: {t['due']} | Assignee: {t['assignee']} | Priority: {t['priority']}"
        for t in tasks
    )
    q = (
        f"Analyze this NPI program's phase gate health. Provide:\n"
        f"1. PHASE GATE STATUS — is this program on track for its current phase exit criteria?\n"
        f"2. CRITICAL GAPS — missing APQP/CDRL/DHF artifacts or gate deliverables at risk\n"
        f"3. TOP 3 RISKS — ranked by severity with specific impact and recommended action\n"
        f"4. RECOMMENDED NEXT ACTIONS — the 3 most important things the TPM should do this week\n\n"
        f"PROJECT: {p['name']}\nINDUSTRY: {p['ind']}\nPHASE: {p['phase']}\n"
        f"FRAMEWORK: {p['framework']}\nCSR: {p['csr']}\nTARGET: {p['target']}\n\n"
        f"TASKS:\n{task_lines}"
    )
    try:
        result = answer(q, use_rewrite=False, reranker_mode="auto", collection=COLLECTION)
        return result.answer
    except Exception as e:
        return f"Analysis error: {e}"


def dashboard_ask(question: str, proj_id: str, history: list) -> tuple:
    if not question.strip() or not proj_id:
        return history, ""
    p = next((x for x in PROJECTS if x["id"] == proj_id), None)
    if not p:
        return history, ""
    tasks = TASKS.get(proj_id, [])
    task_lines = "\n".join(
        f"- [{t['status'].upper()}] {t['name']} | Due: {t['due']} | Assignee: {t['assignee']}"
        for t in tasks
    )
    ctx_q = (
        f"PROJECT CONTEXT:\nProject: {p['name']}\nIndustry: {p['ind']}\nPhase: {p['phase']}\n"
        f"Framework: {p['framework']}\nCSR: {p['csr']}\n\nTASKS:\n{task_lines}\n\n"
        f"QUESTION: {question}"
    )
    hist_fmt = [{"role": m["role"], "content": m["content"]} for m in history]
    try:
        result = answer(ctx_q, use_rewrite=False, reranker_mode="auto",
                        history=hist_fmt, collection=COLLECTION)
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": result.answer})
    except Exception as e:
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": f"Error: {e}"})
    return history, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
SUBTITLE = "Claude Haiku · BGE Reranker (local MPS) · GPT-4o-mini · text-embedding-3-small · Chroma"

CSS = """
.gradio-container { max-width: 1400px !important; }
.source-panel { font-size: 0.85rem; line-height: 1.6; }
.example-btn { font-size: 0.80rem !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="TPM / NPI Expert") as demo:

    gr.Markdown(f"""
# TPM / NPI Expert Knowledge Worker
**{SUBTITLE}**

Expert Q&A across NPI phase gates, APQP/PPAP, RAID management, SAFe, hybrid program models, and six industries.
""")

    with gr.Tabs():

        # ── TAB 1: Expert Q&A ─────────────────────────────────────────────────
        with gr.Tab("🎓 Expert Q&A"):
            with gr.Row():

                # Left: chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="TPM / NPI Expert",
                        height=520,
                    )
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Ask about phase gates, APQP, RAID, SAFe, PPAP, DFM/DFT, metrics…",
                            show_label=False,
                            scale=5,
                            lines=2,
                        )
                        send_btn = gr.Button("Ask →", variant="primary", scale=1)

                    gr.Markdown("**Example questions:**")
                    with gr.Row():
                        ex_btns = [
                            gr.Button(e[:72] + ("…" if len(e) > 72 else ""),
                                      elem_classes="example-btn", size="sm")
                            for e in EXAMPLES
                        ]

                    with gr.Accordion("Query rewrites (debug)", open=False):
                        rewrites_box = gr.Markdown()

                    with gr.Row():
                        rewrite_toggle = gr.Checkbox(value=True, label="Query rewriting")
                        reranker_radio = gr.Radio(
                            choices=["auto", "bge", "llm"],
                            value="auto",
                            label="Reranker",
                        )
                        clear_btn = gr.Button("Clear", size="sm")

                # Right: sources
                with gr.Column(scale=2):
                    gr.Markdown("### Source chunks")
                    source_box = gr.Markdown(
                        value="*Sources will appear here after your first question.*",
                        elem_classes="source-panel",
                    )

            # Wire up Expert Q&A
            def submit_msg(question, history, use_rewrite, reranker):
                return rag_answer(question, history, use_rewrite, reranker)

            send_btn.click(
                fn=submit_msg,
                inputs=[msg_box, chatbot, rewrite_toggle, reranker_radio],
                outputs=[chatbot, rewrites_box, source_box],
            ).then(fn=lambda: "", outputs=msg_box)

            msg_box.submit(
                fn=submit_msg,
                inputs=[msg_box, chatbot, rewrite_toggle, reranker_radio],
                outputs=[chatbot, rewrites_box, source_box],
            ).then(fn=lambda: "", outputs=msg_box)

            clear_btn.click(
                fn=lambda: ([], "", "*Sources will appear here after your first question.*"),
                outputs=[chatbot, rewrites_box, source_box],
            )

            for btn, ex in zip(ex_btns, EXAMPLES):
                btn.click(fn=lambda e=ex: e, outputs=msg_box)

        # ── TAB 2: PM Dashboard ───────────────────────────────────────────────
        with gr.Tab("📊 PM Dashboard"):

            gr.Markdown("""
### Multi-Industry Program Intelligence
Select a program to view tasks, run a phase gate analysis, or ask the expert a context-aware question.
*Asana MCP-ready — swap mock data for live API calls via the `mcp_servers` parameter.*
""")
            proj_state = gr.State(None)

            with gr.Row():
                # Left: project selector
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("**Active programs**")
                    proj_btns = []
                    for p in PROJECTS:
                        btn = gr.Button(
                            f"{p['ind']} · {p['name'][:38]}…\n{p['phase']} · {p['risk']} risk",
                            size="sm",
                        )
                        proj_btns.append((btn, p["id"]))

                    gr.Markdown("---\n**Integrations**\n- 🟢 Mock data — Asana MCP ready\n- ⚪ Trello — configure API key\n- ⚪ Notion — configure token")

                # Right: main panel
                with gr.Column(scale=4):
                    with gr.Tabs() as inner_tabs:

                        with gr.Tab("Tasks"):
                            task_display = gr.Markdown("*Select a program from the left.*")

                        with gr.Tab("Phase gate analysis"):
                            analyse_btn    = gr.Button("⚡ Run phase gate analysis", variant="primary")
                            analysis_out   = gr.Markdown("*Click 'Run phase gate analysis' to generate an expert assessment.*")

                        with gr.Tab("Ask the expert"):
                            dash_chatbot = gr.Chatbot(
                                label="Context-aware expert",
                                height=400,
                            )
                            with gr.Row():
                                dash_input = gr.Textbox(
                                    placeholder="Ask about gate status, CSRs, APQP artifacts, risks, escalations…",
                                    show_label=False,
                                    scale=5,
                                )
                                dash_send = gr.Button("Ask →", variant="primary", scale=1)

                            gr.Markdown("**Suggested:**")
                            DASH_EXAMPLES = [
                                "What's missing for our next gate?",
                                "Which CSR items are at risk?",
                                "Generate a PFMEA row for the top risk",
                                "What APQP artifacts are incomplete?",
                                "Summarize top 3 escalations this week",
                            ]
                            with gr.Row():
                                for de in DASH_EXAMPLES:
                                    gr.Button(de, size="sm").click(fn=lambda e=de: e, outputs=dash_input)

            # Wire up project selection
            def select_project(pid):
                return pid, get_project_info(pid)

            for btn, pid in proj_btns:
                btn.click(
                    fn=lambda p=pid: select_project(p),
                    outputs=[proj_state, task_display],
                )

            # Wire up analysis
            analyse_btn.click(
                fn=run_dashboard_analysis,
                inputs=[proj_state],
                outputs=[analysis_out],
            )

            # Wire up dashboard chat
            dash_send.click(
                fn=dashboard_ask,
                inputs=[dash_input, proj_state, dash_chatbot],
                outputs=[dash_chatbot, dash_input],
            )
            dash_input.submit(
                fn=dashboard_ask,
                inputs=[dash_input, proj_state, dash_chatbot],
                outputs=[dash_chatbot, dash_input],
            )

    gr.Markdown(f"*{SUBTITLE} · [github.com/kolmag/tpm-npi-expert](https://github.com/kolmag/tpm-npi-expert)*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Default(), css=CSS)
