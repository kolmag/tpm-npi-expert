"""
answer.py — TPM/NPI Expert RAG Pipeline
Query rewrite (Claude Haiku) → Retrieve (Chroma) → Rerank (BGE / LLM) → Answer (GPT-4o-mini) → Groundedness check (Claude Haiku)
"""
import os, re, sys, json, argparse, time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = Path(os.getenv("CHROMA_DIR", str(Path(__file__).parent.parent / "chroma_db")))
COLLECTION    = "tpm_npi_expert"
EMBED_MODEL   = "text-embedding-3-small"
REWRITE_MODEL = "claude-haiku-4-5"
ANSWER_MODEL  = "gpt-4o-mini"
BGE_MODEL     = "BAAI/bge-reranker-v2-m3"
N_REWRITES    = 3
RETRIEVAL_K   = 20
FINAL_K       = 15
ANSWER_TEMP   = 0
# ─────────────────────────────────────────────────────────────────────────────

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_cl = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ANSWER_SYSTEM = """You are a senior TPM and NPI Expert Knowledge Worker with deep expertise across:
- NPI Phase Gate methodology (Phase 0–5), APQP, PPAP, DFMEA/PFMEA, Control Plans, DFM/DFT
- TPM program management: charter, WBS, schedule baseline, RAID logs, escalation ladders, EVM
- Agile/Scrum, SAFe PI Planning, hybrid program models
- Industries: automotive (IATF 16949), defense (MIL-STD), semiconductor (JEDEC), medical (ISO 13485/FDA), data center, high-tech
- Software delivery: SaaS, CI/CD, SOC2, FDA SaMD, connected hardware-software programs
- CSR (Customer-Specific Requirements) and LLC (Lessons Learned Cards) management

Answer ONLY from the provided context chunks. If the context does not contain enough information, say so clearly — do not fill gaps with general knowledge. Be specific, direct, and practical. Reference source documents when relevant."""


@dataclass
class AnswerResult:
    question:          str
    rewritten_queries: list[str]
    ranked_chunks:     list[dict]
    answer:            str
    sources:           list[str]
    reranker_used:     str
    checker_score:     float | None = None


# ── BGE reranker (lazy load — loads ONCE, reused across batch) ────────────────
_bge_model = None

def get_bge():
    global _bge_model
    if _bge_model is None:
        try:
            from sentence_transformers import CrossEncoder
            import torch
            device = "mps" if torch.backends.mps.is_available() else \
                     "cuda" if torch.cuda.is_available() else "cpu"
            console.print(f"[dim]Loading BGE reranker on {device}…[/dim]")
            _bge_model = CrossEncoder(BGE_MODEL, device=device)
            console.print("[dim]BGE loaded.[/dim]")
        except Exception as e:
            console.print(f"[yellow]BGE load failed: {e}[/yellow]")
            _bge_model = None
    return _bge_model


# ── Chroma (lazy load — opens ONCE, reused across batch) ──────────────────────
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION)
    return _collection


# ── Query rewriting ───────────────────────────────────────────────────────────
def rewrite_query(question: str, history: list[dict] | None = None) -> list[str]:
    history_str = ""
    if history:
        last = history[-4:]
        history_str = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in last)
        history_str = f"\nConversation history:\n{history_str}\n"

    prompt = f"""{history_str}
Original question: {question}

Write {N_REWRITES} alternative phrasings of this question to improve retrieval from a TPM/NPI knowledge base.
Each phrasing should use different terminology but capture the same intent.
Use domain terms: phase gate, APQP, PPAP, RAID, WBS, DFMEA, PFMEA, CSR, LLC, EVM, SPI, CPI, mitigation, contingency, escalation, DVT, EVT, PVT.

Respond with JSON only — a list of {N_REWRITES} strings:
["rephrasing 1", "rephrasing 2", "rephrasing 3"]"""

    try:
        resp = anthropic.messages.create(
            model=REWRITE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            rewrites = json.loads(m.group())
            if isinstance(rewrites, list):
                return [question] + [r for r in rewrites if isinstance(r, str)]
    except Exception as e:
        console.print(f"[yellow]Rewrite failed: {e}[/yellow]")
    return [question]


# ── Retrieval ────────────────────────────────────────────────────────────────
def retrieve(queries: list[str], collection) -> list[dict]:
    seen, chunks = set(), []
    for q in queries:
        resp = openai_cl.embeddings.create(model=EMBED_MODEL, input=[q])
        vec  = resp.data[0].embedding
        results = collection.query(
            query_embeddings=[vec],
            n_results=RETRIEVAL_K,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            cid = meta.get("doc_name", "") + str(meta.get("chunk_index", ""))
            if cid not in seen:
                seen.add(cid)
                chunks.append({
                    "text":        doc,
                    "doc_name":    meta.get("doc_name", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "headline":    meta.get("headline", ""),
                    "summary":     meta.get("summary", ""),
                    "distance":    dist,
                })
    return chunks


# ── Reranking ────────────────────────────────────────────────────────────────
def rerank_bge(question: str, chunks: list[dict]) -> tuple[list[dict], str]:
    bge = get_bge()
    if bge is None:
        return rerank_llm(question, chunks)
    try:
        pairs  = [(question, c["text"]) for c in chunks]
        scores = bge.predict(pairs).tolist()
        for c, s in zip(chunks, scores):
            c["rerank_score"] = s
        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:FINAL_K], "bge"
    except Exception as e:
        console.print(f"[yellow]BGE rerank failed: {e} — falling back to LLM reranker[/yellow]")
        return rerank_llm(question, chunks)


def rerank_llm(question: str, chunks: list[dict]) -> tuple[list[dict], str]:
    numbered = "\n\n".join(
        f"[{i}] {c['headline']}\n{c['text'][:400]}" for i, c in enumerate(chunks[:25])
    )
    prompt = f"""Question: {question}

Rank the following chunks by relevance. Return JSON only — a list of indices from most to least relevant:
[most_relevant_index, ..., least_relevant_index]

Chunks:
{numbered}"""
    try:
        resp = anthropic.messages.create(
            model=REWRITE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            order = json.loads(m.group())
            reranked = [chunks[i] for i in order if 0 <= i < len(chunks)]
            for rank, c in enumerate(reranked):
                c["rerank_score"] = 1.0 - rank / len(reranked)
            return reranked[:FINAL_K], "llm"
    except Exception as e:
        console.print(f"[yellow]LLM rerank failed: {e} — using distance order[/yellow]")
    for i, c in enumerate(chunks):
        c["rerank_score"] = 1.0 - c.get("distance", 0)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:FINAL_K], "distance"


# ── Answer generation ─────────────────────────────────────────────────────────
def generate_answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> tuple[str, list[str]]:
    context = "\n\n---\n\n".join(
        f"[Source: {c['doc_name']}, chunk {c['chunk_index']}]\n{c['headline']}\n{c['text']}"
        for c in chunks
    )
    messages = []
    if history:
        for m in history[-6:]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
    })
    resp = openai_cl.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=ANSWER_TEMP,
        messages=[{"role": "system", "content": ANSWER_SYSTEM}] + messages,
    )
    answer  = resp.choices[0].message.content
    sources = list({c["doc_name"] for c in chunks})
    return answer, sources


# ── Groundedness check ────────────────────────────────────────────────────────
def check_groundedness(question: str, answer: str, chunks: list[dict]) -> tuple[str, float | None]:
    if len(answer) < 200:
        return answer, None
    # BGE scores are raw logits (unbounded) — do not threshold by score value.
    # Always run the check when the answer is long enough and we have chunks.
    if not chunks:
        return answer, None

    context = "\n\n".join(
        f"[{c['doc_name']}]\n{c['text'][:600]}" for c in chunks[:8]
    )
    prompt = f"""You are a groundedness checker for a RAG system.

CONTEXT (retrieved source chunks):
{context}

QUESTION: {question}

ANSWER TO CHECK:
{answer}

Does the answer only use information present in the context above?
Respond with ONLY a JSON object — no explanation, no markdown:
{{"score": 0.9}}

Score guide: 1.0=fully grounded, 0.8=mostly grounded, 0.6=some unsupported claims, 0.4=significant hallucination, 0.2=mostly fabricated."""

    try:
        resp = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        for pat in [r'"score"\s*:\s*([0-9.]+)', r':\s*([0-9.]+)', r'\b(0\.[0-9]+|1\.0)\b']:
            m = re.search(pat, raw)
            if m:
                return answer, min(1.0, max(0.0, float(m.group(1))))
        raise ValueError(f"No score in: {raw[:60]}")
    except Exception as e:
        console.print(f"[dim]Groundedness check skipped: {e}[/dim]")
        return answer, None


# ── Main answer() function ────────────────────────────────────────────────────
def answer(
    question: str,
    use_rewrite:   bool = True,
    debug:         bool = False,
    reranker_mode: str  = "auto",
    history:       list[dict] | None = None,
    collection     = None,   # pass in to avoid re-opening Chroma on every call
) -> AnswerResult:

    col = collection or get_collection()

    # 1. Query rewriting
    queries = rewrite_query(question, history) if use_rewrite else [question]
    if debug:
        console.print(f"\n[bold]Queries:[/bold] {queries}")

    # 2. Retrieval
    chunks = retrieve(queries, col)
    if debug:
        console.print(f"[bold]Retrieved:[/bold] {len(chunks)} unique chunks")

    # 3. Reranking
    if reranker_mode in ("bge", "auto"):
        ranked, reranker_used = rerank_bge(question, chunks)
    else:
        ranked, reranker_used = rerank_llm(question, chunks)

    if debug:
        console.print(f"[bold]Reranker:[/bold] {reranker_used}")
        console.print("\n[bold]Top 5 chunks:[/bold]")
        for c in ranked[:5]:
            console.print(f"  [{c['doc_name']}] {c['headline'][:60]}  score={c.get('rerank_score', 0):.3f}")

    # 4. Answer generation
    ans, sources = generate_answer(question, ranked, history)

    # 5. Groundedness check
    ans, g_score = check_groundedness(question, ans, ranked)

    if debug:
        console.print(f"\n[bold]Groundedness score:[/bold] {g_score}")
        console.print(f"\n[bold]Sources:[/bold] {sources}")

    return AnswerResult(
        question          = question,
        rewritten_queries = queries,
        ranked_chunks     = ranked,
        answer            = ans,
        sources           = sources,
        reranker_used     = reranker_used,
        checker_score     = g_score,
    )


# ── Batch mode ────────────────────────────────────────────────────────────────
def run_batch(questions: list[str], reranker_mode: str = "auto", debug: bool = False):
    """
    Run all questions in a single process — BGE and Chroma load once,
    are reused for every question. Prints a compact summary table.
    """
    console.print(f"\n[bold]TPM/NPI Expert — Batch Run[/bold]")
    console.print(f"Questions : {len(questions)}")
    console.print(f"Reranker  : {reranker_mode}\n")

    # Warm up BGE and Chroma once
    col = get_collection()
    if reranker_mode in ("bge", "auto"):
        get_bge()  # loads model now so first question isn't slow

    results = []
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        console.print(f"[dim][{i}/{len(questions)}] {q[:80]}…[/dim]")
        try:
            r = answer(q, debug=debug, reranker_mode=reranker_mode, collection=col)
            elapsed = time.time() - t0
            results.append({
                "q":         q,
                "sources":   r.sources,
                "reranker":  r.reranker_used,
                "top_score": round(r.ranked_chunks[0].get("rerank_score", 0), 3) if r.ranked_chunks else 0,
                "n_sources": len(r.sources),
                "grnd":      round(r.checker_score, 2) if r.checker_score is not None else "—",
                "elapsed":   round(elapsed, 1),
                "top_chunks": [(c["doc_name"], round(c.get("rerank_score",0),3)) for c in r.ranked_chunks[:3]],
            })
        except Exception as e:
            console.print(f"[red]  Error: {e}[/red]")
            results.append({"q": q, "error": str(e)})

    # ── Summary table ──────────────────────────────────────────────────────────
    console.print("\n")
    table = Table(title="Batch Results", show_lines=True)
    table.add_column("#",          width=3,  style="dim")
    table.add_column("Question",   width=48, no_wrap=False)
    table.add_column("Top 3 chunks (doc · score)", width=52, no_wrap=False)
    table.add_column("Src", width=4,  justify="center")
    table.add_column("Grnd", width=5, justify="center")
    table.add_column("s",    width=5, justify="right", style="dim")

    for i, r in enumerate(results, 1):
        if "error" in r:
            table.add_row(str(i), r["q"][:80], f"[red]ERROR: {r['error'][:40]}[/red]", "—", "—", "—")
            continue
        chunks_str = "\n".join(
            f"{doc.replace('_',' ')[:28]} · {sc}"
            for doc, sc in r["top_chunks"]
        )
        grnd_color = "green" if isinstance(r["grnd"], float) and r["grnd"] >= 0.8 \
                     else "yellow" if isinstance(r["grnd"], float) and r["grnd"] >= 0.6 \
                     else "dim"
        table.add_row(
            str(i),
            r["q"][:80],
            chunks_str,
            str(r["n_sources"]),
            f"[{grnd_color}]{r['grnd']}[/{grnd_color}]",
            str(r["elapsed"]),
        )

    console.print(table)

    # ── Aggregate stats ────────────────────────────────────────────────────────
    scores = [r["grnd"] for r in results if isinstance(r.get("grnd"), float)]
    multi  = [r for r in results if r.get("n_sources", 0) > 1]
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Avg groundedness : {sum(scores)/len(scores):.2f}" if scores else "  Avg groundedness : n/a")
    console.print(f"  Cross-doc hits   : {len(multi)}/{len(results)} questions pulled from 2+ docs")
    console.print(f"  Total time       : {sum(r.get('elapsed',0) for r in results):.0f}s\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPM/NPI Expert RAG pipeline")
    parser.add_argument("question",     nargs="?",  help="Single question to answer")
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--reranker",   choices=["auto", "bge", "llm"], default="auto")
    parser.add_argument("--batch",      metavar="FILE",
                        help="Path to a text file with one question per line — runs all in one process")
    args = parser.parse_args()

    if args.batch:
        # ── Batch mode: load BGE once, run all questions ───────────────────────
        path = Path(args.batch)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
        questions = [l.strip() for l in path.read_text().splitlines()
                     if l.strip() and not l.strip().startswith("#")]
        run_batch(questions, reranker_mode=args.reranker, debug=args.debug)

    else:
        # ── Single question mode ───────────────────────────────────────────────
        q = args.question or input("Question: ").strip()
        result = answer(
            q,
            use_rewrite   = not args.no_rewrite,
            debug         = args.debug,
            reranker_mode = args.reranker,
        )
        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(result.answer)
        console.print(f"\n[dim]Sources: {', '.join(result.sources)}[/dim]")
        console.print(f"[dim]Reranker: {result.reranker_used} | Groundedness: {result.checker_score}[/dim]")
