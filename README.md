# Paper 1 — RAG Credit Decision Explainer

Implementation of:
> "An approach to integrate generative artificial intelligence into automated credit decision-making processes"
> Quirini (Monte dei Paschi di Siena), Guerra & Mariani (Experian), 2025

---

## What this implements

```
Credit application
      ↓
ADMS (rule-based scoring: DSTI, Debt/Plafond, rNPV, credit rules)
      ↓
System-to-System: ADMS output → RAG layer
      ↓
Vector DB similarity search (ChromaDB + sentence-transformers)
      ↓
LLM validates compliance (GPT-4o-mini or local Ollama)
      ↓
Structured explanation table + GDPR Art.22 narrative
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run
python rag_credit_explainer.py
```

---

## Using a local model instead of OpenAI (free, no API key)

```bash
# 1. Install Ollama: https://ollama.ai
# 2. Pull a model
ollama pull llama3

# 3. In .env set:
USE_LOCAL_LLM=true
LOCAL_LLM_MODEL=llama3

# 4. Run as normal — Ollama must be running in background
python rag_credit_explainer.py
```

---

## Project structure

```
paper1_rag/
├── rag_credit_explainer.py     ← main script (run this)
├── requirements.txt
├── .env.example                ← copy to .env and add API key
├── knowledge_base/             ← policy documents (add your own .txt files here)
│   ├── internal_credit_policy.txt
│   ├── responsible_lending_guidelines.txt
│   └── gdpr_ai_act_compliance.txt
└── data/
    ├── test_cases.csv          ← 8 test cases from the paper
    └── output_CASE_XX.json     ← generated outputs (created when you run)
```

---

## Adding your own policy documents

Just drop any `.txt` file into the `knowledge_base/` folder.
The script will automatically chunk, embed, and index it on next run.
No code changes needed.

---

## Test cases included

| Case | Product | Amount | Expected |
|------|---------|--------|----------|
| CASE_01 | Personal Loan | €17,000 | GREEN |
| CASE_02 | Personal Loan | €17,000 | YELLOW (borderline — from paper's running example) |
| CASE_03 | Personal Loan | €17,000 | YELLOW (marginal rNPV) |
| CASE_04 | Personal Loan | €17,000 | RED (high DSTI, negative rNPV) |
| CASE_05 | Mortgage | €120,000 | YELLOW |
| CASE_06 | Overdraft | €5,000 | GREEN |
| CASE_07 | Personal Loan | €25,000 | YELLOW |
| CASE_08 | Personal Loan | €10,000 | RED |

---

## Output format

For each case, the system prints:
1. Retrieved policy passages (with similarity scores)
2. Summary table: each indicator vs threshold, risk interpretation, policy reference
3. Recommendation: APPROVE / MANUAL REVIEW / REJECT
4. Rationale grounded in retrieved documents
5. GDPR Art.22 explanation (suitable for applicant)
6. PROB score (groundedness confidence)

Full output also saved as JSON in `data/output_CASE_XX.json`.

---

## Key implementation notes

- **Embeddings**: Uses `all-MiniLM-L6-v2` (local, no API cost) via sentence-transformers
- **Vector store**: ChromaDB in-memory (change to `PersistentClient` to save across runs)
- **LLM temperature**: Set to 0.0 for deterministic compliance outputs
- **Hallucination control**: System prompt strictly forbids output outside retrieved documents
- **JSON output**: Forces `response_format={"type": "json_object"}` for structured parsing
# ADMS_RAG_credit
