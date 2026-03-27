"""
eval.py — TPM/NPI Expert Evaluation Pipeline
MRR@10 (retrieval quality) + LLM-as-judge (answer quality)
Colab-ready: works locally or on Google Colab with uploaded chroma_db

Usage:
  uv run evaluation/eval.py --tests evaluation/tests.jsonl
  uv run evaluation/eval.py --tests evaluation/tests.jsonl --sample 20
  uv run evaluation/eval.py --tests evaluation/tests.jsonl --category apqp
  uv run evaluation/eval.py --tests evaluation/tests.jsonl --difficulty advanced
"""
import os, sys, json, argparse, time, random, re
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.progress import track

load_dotenv()
console = Console()

# ── Add scripts/ to path so we can import answer.py ──────────────────────────
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from answer import answer as rag_answer, get_collection, get_bge

JUDGE_MODEL = "gpt-4o-mini"
anthropic    = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_judge = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_SYSTEM = """You are an expert evaluator for a TPM/NPI knowledge worker RAG system.
You assess answers on three dimensions, each scored 1–10:

CORRECTNESS (1–10): Is the answer factually accurate and technically correct for the domain?
- 9–10: Completely accurate, domain-expert level precision
- 7–8: Mostly accurate, minor gaps or imprecision
- 5–6: Partially accurate, some errors or missing key points
- 3–4: Significant inaccuracies or missing critical information
- 1–2: Mostly wrong or dangerously misleading

COMPLETENESS (1–10): Does the answer cover all aspects of the question?
- 9–10: Comprehensive, covers all dimensions with appropriate depth
- 7–8: Covers main points, minor omissions
- 5–6: Covers basics but misses important aspects
- 3–4: Significant gaps in coverage
- 1–2: Barely addresses the question

GROUNDEDNESS (1–10): Is the answer grounded in the provided context, or does it hallucinate?
- 9–10: Every claim traceable to context; no hallucination
- 7–8: Mostly grounded; minor extrapolations acceptable
- 5–6: Some ungrounded claims but not harmful
- 3–4: Notable hallucination or unsupported assertions
- 1–2: Fabricates information or contradicts context

Respond ONLY with valid JSON, no preamble:
{"correctness": 8, "completeness": 7, "groundedness": 9, "overall": 8, "note": "brief explanation"}

The "overall" score is your holistic assessment (not an average).
"""


# ── MRR@10 calculation ────────────────────────────────────────────────────────
def compute_mrr(ranked_chunks: list[dict], expected_sources: list[str]) -> float:
    """MRR@10 — reciprocal rank of the first relevant chunk in top 10."""
    if not expected_sources:
        return 1.0  # blind questions — no source expectation
    for rank, chunk in enumerate(ranked_chunks[:10], 1):
        doc = chunk.get("doc_name", "")
        if any(exp in doc for exp in expected_sources):
            return 1.0 / rank
    return 0.0


# ── Retry helper ─────────────────────────────────────────────────────────────
def _call_with_retry(fn, retries=4, base_delay=8):
    """Exponential backoff for 529 overloaded and transient errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if attempt == retries - 1:
                raise
            if "529" in msg or "overloaded" in msg.lower() or "rate" in msg.lower():
                delay = base_delay * (2 ** attempt)
                console.print(f"[dim]Rate limited — retrying in {delay}s…[/dim]")
                time.sleep(delay)
            else:
                raise


def _sanitise(text: str) -> str:
    """Remove control characters that break JSON payloads."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)


# ── LLM-as-judge scoring ──────────────────────────────────────────────────────
def judge_answer(question: str, answer: str, expected_topics: list[str]) -> dict:
    topics_str = ", ".join(expected_topics) if expected_topics else "general TPM/NPI domain"
    # Sanitise to avoid OpenAI 400 on control chars
    answer_clean = _sanitise(answer[:2000])
    prompt = f"""Question: {_sanitise(question)}

Expected topics to cover: {topics_str}

Answer to evaluate:
{answer_clean}

Score this answer on correctness, completeness, and groundedness."""

    def _call():
        resp = openai_judge.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"No JSON in judge response: {raw[:80]}")

    try:
        result = _call_with_retry(_call)
        if result:
            return result
    except Exception as e:
        console.print(f"[yellow]Judge failed: {e}[/yellow]")
        time.sleep(5)  # cool down before next question
    return {"correctness": 0, "completeness": 0, "groundedness": 0, "overall": 0, "note": "judge error"}


# ── Main eval loop ────────────────────────────────────────────────────────────
def run_eval(
    tests_path: Path,
    sample:     int | None = None,
    category:   str | None = None,
    difficulty: str | None = None,
    reranker:   str = "auto",
    no_judge:   bool = False,
):
    # Load test set
    tests = []
    with open(tests_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tests.append(json.loads(line))

    # Filter
    if category:
        tests = [t for t in tests if t.get("category", "").lower() == category.lower()]
    if difficulty:
        tests = [t for t in tests if t.get("difficulty", "").lower() == difficulty.lower()]
    if sample:
        random.shuffle(tests)
        tests = tests[:sample]

    console.print(f"\n[bold]TPM/NPI Expert — Evaluation Pipeline[/bold]")
    console.print(f"Test set   : {tests_path.name}")
    console.print(f"Questions  : {len(tests)}")
    console.print(f"Reranker   : {reranker}")
    console.print(f"Judge      : {'Claude Sonnet (LLM-as-judge)' if not no_judge else 'disabled'}\n")

    # Warm up once
    col = get_collection()
    if reranker in ("bge", "auto"):
        get_bge()

    results = []
    for test in track(tests, description="Evaluating"):
        t0 = time.time()
        qid        = test.get("id", "?")
        question   = test["question"]
        exp_topics = test.get("expected_topics", [])
        exp_srcs   = test.get("expected_sources", [])
        category_  = test.get("category", "")
        difficulty_= test.get("difficulty", "")
        q_type     = test.get("question_type", "")
        source_    = test.get("source", "internal")

        try:
            result = rag_answer(
                question,
                use_rewrite   = True,
                reranker_mode = reranker,
                collection    = col,
            )
            mrr = compute_mrr(result.ranked_chunks, exp_srcs)

            scores = {}
            if not no_judge:
                scores = judge_answer(question, result.answer, exp_topics)

            elapsed = time.time() - t0
            results.append({
                "id":         qid,
                "question":   question,
                "category":   category_,
                "difficulty": difficulty_,
                "type":       q_type,
                "source":     source_,
                "mrr":        mrr,
                "sources":    result.sources,
                "reranker":   result.reranker_used,
                "checker":    result.checker_score,
                "correctness":   scores.get("correctness", None),
                "completeness":  scores.get("completeness", None),
                "groundedness":  scores.get("groundedness", None),
                "overall":       scores.get("overall", None),
                "judge_note":    scores.get("note", ""),
                "answer":        result.answer,
                "elapsed":       round(elapsed, 1),
            })
        except Exception as e:
            console.print(f"[red]Error on {qid}: {e}[/red]")
            results.append({"id": qid, "question": question, "error": str(e),
                            "mrr": 0, "overall": 0})
        time.sleep(2)  # brief pause between questions to avoid rate limits

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    valid   = [r for r in results if "error" not in r]
    mrr_avg = sum(r["mrr"] for r in valid) / len(valid) if valid else 0
    overall_scores = [r["overall"] for r in valid if r.get("overall")]
    correct_scores = [r["correctness"] for r in valid if r.get("correctness")]
    complete_scores= [r["completeness"] for r in valid if r.get("completeness")]
    ground_scores  = [r["groundedness"] for r in valid if r.get("groundedness")]
    checker_scores = [r["checker"] for r in valid if r.get("checker")]

    # ── Results table ─────────────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("ID",    width=6,  style="dim")
    table.add_column("Question",    width=44, no_wrap=False)
    table.add_column("Cat/Diff",    width=18)
    table.add_column("MRR",   width=5,  justify="center")
    table.add_column("Corr",  width=5,  justify="center")
    table.add_column("Comp",  width=5,  justify="center")
    table.add_column("Grnd",  width=5,  justify="center")
    table.add_column("Ovrl",  width=5,  justify="center")
    table.add_column("s",     width=5,  justify="right", style="dim")

    def color(val, lo=6, hi=8):
        if val is None: return "dim"
        return "green" if val >= hi else "yellow" if val >= lo else "red"

    for r in results:
        if "error" in r:
            table.add_row(r["id"], r["question"][:60], "—", "ERR", "—", "—", "—", "—", "—")
            continue
        mrr_c = "green" if r["mrr"] >= 0.5 else "yellow" if r["mrr"] > 0 else "red"
        table.add_row(
            r["id"],
            r["question"][:60],
            f"{r['category'][:10]}\n{r['difficulty'][:6]}",
            f"[{mrr_c}]{r['mrr']:.2f}[/{mrr_c}]",
            f"[{color(r.get('correctness'))}]{r.get('correctness') or '—'}[/{color(r.get('correctness'))}]",
            f"[{color(r.get('completeness'))}]{r.get('completeness') or '—'}[/{color(r.get('completeness'))}]",
            f"[{color(r.get('groundedness'))}]{r.get('groundedness') or '—'}[/{color(r.get('groundedness'))}]",
            f"[{color(r.get('overall'))}]{r.get('overall') or '—'}[/{color(r.get('overall'))}]",
            str(r.get("elapsed", "—")),
        )

    console.print(table)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(f"\n[bold]═══ Summary ═══[/bold]")
    console.print(f"  Questions evaluated : {len(valid)}/{len(results)}")
    console.print(f"  MRR@10              : [bold]{mrr_avg:.3f}[/bold]")
    if overall_scores:
        console.print(f"  Judge Overall       : [bold]{sum(overall_scores)/len(overall_scores):.2f}[/bold] / 10")
        console.print(f"  Judge Correctness   : {sum(correct_scores)/len(correct_scores):.2f} / 10")
        console.print(f"  Judge Completeness  : {sum(complete_scores)/len(complete_scores):.2f} / 10")
        console.print(f"  Judge Groundedness  : {sum(ground_scores)/len(ground_scores):.2f} / 10")
    if checker_scores:
        console.print(f"  Pipeline Grnd (avg) : {sum(checker_scores)/len(checker_scores):.2f}")

    # ── By category ───────────────────────────────────────────────────────────
    cats = {}
    for r in valid:
        c = r.get("category", "unknown")
        cats.setdefault(c, []).append(r)
    if len(cats) > 1:
        console.print(f"\n[bold]By category:[/bold]")
        for cat, rs in sorted(cats.items()):
            cat_mrr = sum(r["mrr"] for r in rs) / len(rs)
            cat_overall = [r["overall"] for r in rs if r.get("overall")]
            cat_ovg = f"{sum(cat_overall)/len(cat_overall):.2f}" if cat_overall else "—"
            console.print(f"  {cat:<30} MRR={cat_mrr:.3f}  Overall={cat_ovg}  n={len(rs)}")

    # ── By difficulty ─────────────────────────────────────────────────────────
    diffs = {}
    for r in valid:
        d = r.get("difficulty", "unknown")
        diffs.setdefault(d, []).append(r)
    if len(diffs) > 1:
        console.print(f"\n[bold]By difficulty:[/bold]")
        for diff, rs in sorted(diffs.items()):
            diff_mrr = sum(r["mrr"] for r in rs) / len(rs)
            diff_overall = [r["overall"] for r in rs if r.get("overall")]
            diff_ovg = f"{sum(diff_overall)/len(diff_overall):.2f}" if diff_overall else "—"
            console.print(f"  {diff:<15} MRR={diff_mrr:.3f}  Overall={diff_ovg}  n={len(rs)}")

    # ── Source breakdown (internal vs blind) ──────────────────────────────────
    by_source = {}
    for r in valid:
        s = r.get("source", "unknown")
        by_source.setdefault(s, []).append(r)
    if len(by_source) > 1:
        console.print(f"\n[bold]By source:[/bold]")
        for src, rs in sorted(by_source.items()):
            src_overall = [r["overall"] for r in rs if r.get("overall")]
            src_mrr = sum(r["mrr"] for r in rs) / len(rs)
            ovg = f"{sum(src_overall)/len(src_overall):.2f}" if src_overall else "—"
            console.print(f"  {src:<30} MRR={src_mrr:.3f}  Overall={ovg}  n={len(rs)}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = tests_path.parent / f"eval_results_{int(time.time())}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    console.print(f"\n[dim]Results saved to {out_path}[/dim]\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPM/NPI Expert Evaluation Pipeline")
    parser.add_argument("--tests",      required=True,  help="Path to .jsonl test set")
    parser.add_argument("--sample",     type=int,       help="Random sample N questions")
    parser.add_argument("--category",                   help="Filter by category")
    parser.add_argument("--difficulty",                 help="Filter by difficulty")
    parser.add_argument("--reranker",   default="auto", choices=["auto","bge","llm"])
    parser.add_argument("--no-judge",   action="store_true", help="Skip LLM judge (retrieval metrics only)")
    args = parser.parse_args()

    run_eval(
        tests_path = Path(args.tests),
        sample     = args.sample,
        category   = args.category,
        difficulty = args.difficulty,
        reranker   = args.reranker,
        no_judge   = args.no_judge,
    )
