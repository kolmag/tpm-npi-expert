"""
ingest.py — TPM/NPI Expert Knowledge Base Builder
Chunk → Enrich (Claude Haiku) → Embed (text-embedding-3-small) → Store (Chroma)
"""
import os, re, sys, json, argparse, hashlib
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import chromadb
from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
KB_DIR         = Path(__file__).parent.parent / "data" / "knowledge_base"
CHROMA_DIR     = Path(__file__).parent.parent / "chroma_db"
COLLECTION     = "tpm_npi_expert"
EMBED_MODEL    = "text-embedding-3-small"
ENRICH_MODEL   = "claude-haiku-4-5"
CHUNK_SIZE     = 500   # tokens
CHUNK_OVERLAP  = 200   # tokens
# ─────────────────────────────────────────────────────────────────────────────

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_cl = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc       = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(enc.encode(text))


def chunk_document(text: str, doc_name: str) -> list[dict]:
    """Paragraph-aware, word-boundary chunking."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks, current, current_tokens = [], [], 0

    for para in paragraphs:
        pt = token_len(para)
        if current_tokens + pt > CHUNK_SIZE and current:
            chunks.append("\n\n".join(current))
            # overlap: keep last paragraphs that fit within CHUNK_OVERLAP tokens
            overlap, ot = [], 0
            for p in reversed(current):
                pt2 = token_len(p)
                if ot + pt2 <= CHUNK_OVERLAP:
                    overlap.insert(0, p)
                    ot += pt2
                else:
                    break
            current, current_tokens = overlap, ot
        current.append(para)
        current_tokens += pt

    if current:
        chunks.append("\n\n".join(current))

    return [
        {
            "text": c,
            "doc_name": doc_name,
            "chunk_index": i,
            "token_count": token_len(c),
            "chunk_id": hashlib.md5(f"{doc_name}:{i}:{c[:80]}".encode()).hexdigest()[:12],
        }
        for i, c in enumerate(chunks)
    ]


def enrich_chunk(chunk: dict) -> dict:
    """Ask Claude Haiku to generate a headline + summary for the chunk."""
    prompt = f"""You are indexing a TPM/NPI expert knowledge base. For the chunk below, write:
1. HEADLINE: A precise 8–12 word title capturing the main topic (no generic titles like "Overview")
2. SUMMARY: 2–3 sentences capturing the key facts, rules, or decision criteria in this chunk.

Respond in JSON only:
{{"headline": "...", "summary": "..."}}

CHUNK (from {chunk['doc_name']}):
{chunk['text'][:1500]}"""

    try:
        resp = anthropic.messages.create(
            model=ENRICH_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        # Extract first {...} block — tolerates trailing text or extra output
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if not m:
            raise ValueError("No JSON object in response")
        data = json.loads(m.group())
        chunk["headline"] = data.get("headline", "")
        chunk["summary"]  = data.get("summary", "")
    except Exception as e:
        console.print(f"[yellow]Enrich failed for {chunk['chunk_id']}: {e}[/yellow]")
        chunk["headline"] = chunk["doc_name"]
        chunk["summary"]  = chunk["text"][:200]

    # embed_text = what gets embedded (headline + summary + original)
    chunk["embed_text"] = f"{chunk['headline']}\n{chunk['summary']}\n\n{chunk['text']}"
    return chunk


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed with text-embedding-3-small."""
    resp = openai_cl.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in resp.data]


def ingest(reset: bool = False):
    # ── Chroma setup ─────────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
            console.print(f"[red]Deleted existing collection '{COLLECTION}'[/red]")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Load docs ─────────────────────────────────────────────────────────────
    docs = sorted(KB_DIR.glob("*.md"))
    if not docs:
        console.print(f"[red]No .md files found in {KB_DIR}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]TPM/NPI Expert — Ingest Pipeline[/bold]")
    console.print(f"KB directory : {KB_DIR}")
    console.print(f"Documents    : {len(docs)}")
    console.print(f"Collection   : {COLLECTION}\n")

    all_chunks = []
    for doc in docs:
        text   = doc.read_text(encoding="utf-8")
        chunks = chunk_document(text, doc.stem)
        console.print(f"  {doc.name:50s} → {len(chunks)} chunks ({token_len(text):,} tokens)")
        all_chunks.extend(chunks)

    console.print(f"\nTotal chunks before enrichment: {len(all_chunks)}")

    # ── Enrich ────────────────────────────────────────────────────────────────
    console.print("\n[bold]Enriching chunks with Claude Haiku…[/bold]")
    enriched = []
    for chunk in track(all_chunks, description="Enriching"):
        enriched.append(enrich_chunk(chunk))

    # ── Embed ─────────────────────────────────────────────────────────────────
    console.print("\n[bold]Embedding with text-embedding-3-small…[/bold]")
    BATCH = 100
    embeddings = []
    for i in track(range(0, len(enriched), BATCH), description="Embedding"):
        batch = enriched[i : i + BATCH]
        embeddings.extend(embed_texts([c["embed_text"] for c in batch]))

    # ── Store ─────────────────────────────────────────────────────────────────
    console.print("\n[bold]Storing in Chroma…[/bold]")
    collection.add(
        ids        = [c["chunk_id"]   for c in enriched],
        embeddings = embeddings,
        documents  = [c["text"]       for c in enriched],
        metadatas  = [
            {
                "doc_name":    c["doc_name"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "headline":    c["headline"],
                "summary":     c["summary"],
            }
            for c in enriched
        ],
    )

    console.print(f"\n[green]Ingest complete.[/green]")
    console.print(f"  Chunks stored : {len(enriched)}")
    console.print(f"  Collection    : {COLLECTION}")
    console.print(f"  Chroma dir    : {CHROMA_DIR}\n")

    # ── Spot-check ────────────────────────────────────────────────────────────
    console.print("[bold]Spot-check — first 3 enriched chunks:[/bold]")
    for c in enriched[:3]:
        console.print(f"\n  [{c['doc_name']} chunk {c['chunk_index']}]")
        console.print(f"  Headline : {c['headline']}")
        console.print(f"  Summary  : {c['summary'][:120]}…")
        console.print(f"  Tokens   : {c['token_count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild collection")
    args = parser.parse_args()
    ingest(reset=args.reset)
