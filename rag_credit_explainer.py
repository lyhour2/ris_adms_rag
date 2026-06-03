"""
Paper 1 Implementation: RAG Credit Decision Explainer
======================================================
Based on: "An approach to integrate generative artificial intelligence into
automated credit decision-making processes"
Quirini (Monte dei Paschi di Siena), Guerra & Mariani (Experian)

How to run:
    1. pip install -r requirements.txt
    2. cp .env.example .env  and fill in your OPENAI_API_KEY
    3. python rag_credit_explainer.py

What this does:
    Step 1 - Loads policy documents and builds a ChromaDB vector store
    Step 2 - Simulates the ADMS decision (rule-based scoring)
    Step 3 - Retrieves relevant policy passages via similarity search
    Step 4 - Calls GPT to validate compliance and generate explanation
    Step 5 - Prints a structured decision table + rationale (like the paper's demo)
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# ── ChromaDB for vector store ──────────────────────────────────────────────
import chromadb
from chromadb.utils import embedding_functions

# ── OpenAI client ──────────────────────────────────────────────────────────
from openai import OpenAI

load_dotenv()
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 80        # overlap between chunks
TOP_K_DOCS = 4            # how many passages to retrieve
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD VECTOR STORE FROM POLICY DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a document into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # drop tiny trailing chunks


def build_vector_store() -> chromadb.Collection:
    """
    Load all .txt files from knowledge_base/, chunk them,
    embed with sentence-transformers, store in ChromaDB.
    """
    console.print("\n[bold cyan]Step 1 — Building knowledge base vector store[/bold cyan]")

    # Use a local embedding model (no API key needed for embeddings)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.Client()  # in-memory; use chromadb.PersistentClient("./chroma_db") to persist

    # Drop and recreate collection each run for clean state
    try:
        client.delete_collection("credit_policy")
    except Exception:
        pass

    collection = client.create_collection(
        name="credit_policy",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks, all_ids, all_metas = [], [], []
    chunk_idx = 0

    for doc_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        console.print(f"  Loaded [green]{doc_path.name}[/green] → {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_idx}")
            all_metas.append({"source": doc_path.stem, "filename": doc_path.name})
            chunk_idx += 1

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metas)
    console.print(f"  [bold]Total: {chunk_idx} chunks embedded and indexed[/bold]\n")
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ADMS DECISION ENGINE (RULE-BASED, FOLLOWS PAPER)
# ─────────────────────────────────────────────────────────────────────────────

def run_adms(case: dict) -> dict:
    """
    Simulate the ADMS tri-component decision:
      - Application score + credit rules  → risk axis
      - Affordability score               → affordability axis
      - rNPV                              → profitability axis
    Returns decision color and flags.
    """
    flags = []

    # Hard reject conditions
    if case["rnpv_mean"] < 0:
        flags.append("NEGATIVE_RNPV")
    if case["dsti"] > 0.40:
        flags.append("DSTI_EXCEEDS_CEILING")
    if case["debt_plafond"] > 0.75:
        flags.append("DEBT_PLAFOND_MANDATORY_REVIEW")
    if case["credit_rule_flag"]:
        flags.append("CREDIT_RULE_TRIGGERED")
    if case["app_score"] == "C" and case["afford_score"] == "C":
        flags.append("DUAL_C_SCORE")

    # Borderline conditions
    borderline_flags = []
    if 0.32 < case["dsti"] <= 0.40:
        borderline_flags.append("DSTI_BORDERLINE")
    if case["debt_income_months"] > 14:
        borderline_flags.append("DEBT_INCOME_ELEVATED")
    if 0 <= case["rnpv_mean"] < 100:
        borderline_flags.append("RNPV_MARGINAL")
    if case["app_score"] == "C" or case["afford_score"] == "C":
        borderline_flags.append("SINGLE_C_SCORE")
    if case["default_prob_annual"] > 0.04:
        borderline_flags.append("ELEVATED_DEFAULT_PROB")

    # Map to decision
    if len(flags) > 0:
        decision = "RED"
    elif len(borderline_flags) > 0:
        decision = "YELLOW"
    else:
        decision = "GREEN"

    return {
        "decision": decision,
        "hard_flags": flags,
        "borderline_flags": borderline_flags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RETRIEVAL (SIMILARITY SEARCH)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_policy_passages(case: dict, adms_result: dict, collection: chromadb.Collection) -> list[dict]:
    """
    Build a semantic query from the credit decision and retrieve
    the most relevant policy passages from the vector store.
    """
    query = (
        f"Credit decision for {case['product']} loan. "
        f"DSTI {case['dsti']*100:.0f}%, "
        f"Debt/Plafond {case['debt_plafond']*100:.0f}%, "
        f"Debt/Income {case['debt_income_months']} months, "
        f"default probability {case['default_prob_annual']*100:.1f}% per year, "
        f"rNPV mean €{case['rnpv_mean']:.0f}. "
        f"ADMS outcome: {adms_result['decision']}. "
        f"Flags: {', '.join(adms_result['hard_flags'] + adms_result['borderline_flags']) or 'none'}. "
        f"Assess policy compliance, GDPR, AI Act obligations."
    )

    results = collection.query(query_texts=[query], n_results=TOP_K_DOCS)

    passages = []
    for i in range(len(results["documents"][0])):
        passages.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "filename": results["metadatas"][0][i]["filename"],
            "distance": results["distances"][0][i],
        })

    return passages


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — LLM GENERATION (GROUNDED IN RETRIEVED DOCS)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a credit compliance assistant embedded in an automated credit
decision system. Your role is to evaluate credit decisions for compliance with bank policy,
responsible lending obligations, GDPR, and the EU AI Act.

STRICT RULES:
- You must ONLY use information from the policy documents provided. Do not use general knowledge.
- Every statement in your output must be traceable to a specific document and rule cited.
- If a topic is not covered in the provided documents, state "Not covered in retrieved documents."
- Do not speculate or extrapolate beyond what the documents say.
- Output ONLY valid JSON — no preamble, no markdown fences, no extra text.
"""

def build_prompt(case: dict, adms_result: dict, passages: list[dict]) -> str:
    """
    Construct the RAG prompt: retrieved docs + decision data + output schema.
    This mirrors the ExperianGPT approach from the paper.
    """
    docs_block = "\n\n".join([
        f"[Document: {p['source']}]\n{p['text']}"
        for p in passages
    ])

    output_schema = {
        "summary_table": [
            {
                "indicator": "string — indicator name",
                "value": "string — actual value",
                "threshold": "string — policy threshold",
                "risk_interpretation": "string — assessment",
                "final_impact": "Positive | Review Required | Concern",
                "policy_reference": "string — doc and rule cited"
            }
        ],
        "normative_flags": ["list of compliance issues found, or empty list"],
        "recommendation": "APPROVE | MANUAL REVIEW | REJECT",
        "rationale": "string — 2-3 sentence explanation grounded in retrieved documents",
        "gdpr_art22_explanation": "string — explanation suitable for providing to the applicant",
        "alternative_structure": "string — suggested alternative if borderline, or null",
        "prob_score": "number 0-10 — confidence that output is fully grounded in retrieved docs"
    }

    return f"""POLICY DOCUMENTS (retrieved via similarity search):
{docs_block}

CREDIT DECISION DATA:
{json.dumps(case, indent=2)}

ADMS OUTCOME: {adms_result['decision']}
HARD FLAGS: {adms_result['hard_flags']}
BORDERLINE FLAGS: {adms_result['borderline_flags']}

TASK: Using ONLY the policy documents above, produce a structured compliance evaluation.
Output ONLY a JSON object matching this schema exactly:
{json.dumps(output_schema, indent=2)}
"""


def call_llm(prompt: str) -> dict:
    """
    Call OpenAI GPT or a local Ollama model.
    Returns parsed JSON output from the LLM.
    """
    if USE_LOCAL_LLM:
        # Ollama local model — run: ollama run llama3
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model = LOCAL_MODEL
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = "gpt-4o-mini"  # affordable and sufficient; swap to gpt-4o for better quality

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,    # deterministic — important for compliance use
        max_tokens=1200,
        response_format={"type": "json_object"},  # force JSON output
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — RENDER OUTPUT (MATCHES PAPER'S DEMO FORMAT)
# ─────────────────────────────────────────────────────────────────────────────

def render_output(case: dict, adms_result: dict, passages: list[dict], llm_output: dict):
    """Print the structured explanation in a format matching the paper's demo."""

    console.print()

    # ── Header ──────────────────────────────────────────────────────────────
    decision = llm_output.get("recommendation", adms_result["decision"])
    color = {"APPROVE": "green", "MANUAL REVIEW": "yellow", "REJECT": "red"}.get(decision, "white")
    adms_color = {"GREEN": "green", "YELLOW": "yellow", "RED": "red"}.get(adms_result["decision"], "white")

    console.print(Panel(
        f"[bold]Case: {case['case_id']}[/bold]  |  "
        f"Product: {case['product']}  |  "
        f"Amount: €{case['amount']:,}  |  "
        f"Term: {case['term_months']} months\n"
        f"ADMS Decision: [{adms_color}]{adms_result['decision']}[/{adms_color}]  →  "
        f"RAG Recommendation: [{color}]{decision}[/{color}]",
        title="[bold cyan]Credit Decision Analysis[/bold cyan]",
        border_style="cyan",
    ))

    # ── Retrieved docs ───────────────────────────────────────────────────────
    console.print("\n[bold]Retrieved Policy Passages[/bold]")
    for i, p in enumerate(passages, 1):
        console.print(f"  [{i}] [blue]{p['source']}[/blue] (similarity: {1-p['distance']:.2f})")
        console.print(f"      [dim]{p['text'][:120]}...[/dim]")

    # ── Summary table ────────────────────────────────────────────────────────
    console.print("\n[bold]Summary Table — Key Indicators[/bold]")
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Key Indicator", style="bold", min_width=18)
    table.add_column("Value", min_width=10)
    table.add_column("Threshold", min_width=12)
    table.add_column("Risk Interpretation", min_width=30)
    table.add_column("Impact", min_width=14)
    table.add_column("Policy Reference", min_width=20)

    impact_colors = {
        "Positive": "green",
        "Review Required": "yellow",
        "Concern": "red",
    }

    for row in llm_output.get("summary_table", []):
        impact = row.get("final_impact", "")
        impact_colored = f"[{impact_colors.get(impact, 'white')}]{impact}[/]"
        table.add_row(
            row.get("indicator", ""),
            str(row.get("value", "")),
            str(row.get("threshold", "")),
            row.get("risk_interpretation", ""),
            impact_colored,
            row.get("policy_reference", ""),
        )
    console.print(table)

    # ── Normative flags ──────────────────────────────────────────────────────
    flags = llm_output.get("normative_flags", [])
    if flags:
        console.print("\n[bold yellow]Normative Flags[/bold yellow]")
        for f in flags:
            console.print(f"  ⚠  {f}")

    # ── Recommendation + Rationale ──────────────────────────────────────────
    console.print(Panel(
        f"[bold]Recommendation: [{color}]{decision}[/{color}][/bold]\n\n"
        f"{llm_output.get('rationale', '')}\n\n"
        f"[bold]Alternative structure:[/bold] {llm_output.get('alternative_structure') or 'N/A'}\n\n"
        f"[bold]GDPR Art.22 explanation for applicant:[/bold]\n"
        f"[italic]{llm_output.get('gdpr_art22_explanation', '')}[/italic]\n\n"
        f"[dim]PROB score (groundedness): {llm_output.get('prob_score', 'N/A')}/10[/dim]",
        title=f"[bold {color}]Final Decision[/bold {color}]",
        border_style=color,
    ))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — RUN PIPELINE ON ALL TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print(Panel(
        "[bold]Paper 1 Implementation: RAG Credit Decision Explainer[/bold]\n"
        "Based on Quirini, Guerra, Mariani (2025)\n"
        "ADMS → Retrieve policy docs → LLM validates compliance → Structured explanation",
        border_style="cyan",
    ))

    # ── Step 1: Build vector store ───────────────────────────────────────────
    collection = build_vector_store()

    # ── Load test cases ──────────────────────────────────────────────────────
    df = pd.read_csv("data/test_cases.csv")
    cases = df.to_dict(orient="records")

    console.print(f"[bold]Loaded {len(cases)} test cases[/bold]\n")
    console.print("Which case(s) to run?")
    for i, c in enumerate(cases):
        console.print(f"  [{i+1}] {c['case_id']} — {c['product']} €{c['amount']:,} "
                      f"(expected: {c['expected_adms_decision']})")

    console.print("\n  [0] Run ALL cases")
    choice = input("\nEnter number (default=2 for borderline case): ").strip() or "2"

    if choice == "0":
        selected = cases
    else:
        idx = int(choice) - 1
        selected = [cases[idx]]

    # ── Run pipeline for each selected case ──────────────────────────────────
    for case in selected:
        console.print(f"\n{'='*70}")
        console.print(f"[bold cyan]Processing: {case['case_id']}[/bold cyan]")

        # Step 2: ADMS decision
        console.print("[dim]Step 2: Running ADMS...[/dim]")
        adms_result = run_adms(case)
        console.print(f"  ADMS → [{adms_result['decision']}] "
                      f"hard_flags={adms_result['hard_flags']} "
                      f"borderline={adms_result['borderline_flags']}")

        # Step 3: Retrieve relevant passages
        console.print("[dim]Step 3: Retrieving policy passages...[/dim]")
        passages = retrieve_policy_passages(case, adms_result, collection)
        console.print(f"  Retrieved {len(passages)} passages from: "
                      f"{list({p['source'] for p in passages})}")

        # Step 4: Call LLM
        console.print("[dim]Step 4: Calling LLM for compliance validation...[/dim]")
        prompt = build_prompt(case, adms_result, passages)
        try:
            llm_output = call_llm(prompt)
        except Exception as e:
            console.print(f"[red]LLM call failed: {e}[/red]")
            console.print("[yellow]Check your API key in .env or try USE_LOCAL_LLM=true[/yellow]")
            continue

        # Step 5: Render output
        render_output(case, adms_result, passages, llm_output)

        # Save output to JSON
        out_path = Path(f"data/output_{case['case_id']}.json")
        out_path.write_text(json.dumps({
            "case": case,
            "adms_result": adms_result,
            "retrieved_passages": passages,
            "llm_output": llm_output,
        }, indent=2))
        console.print(f"[dim]Output saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
