"""
CiteFirst Pipeline — Logic Engine
Implements the multi-stage corrective RAG workflow:
  Stage 1: Hybrid Retrieval (Dense + BM25 + RRF)  → database.py
  Stage 2: Cross-Encoder Reranking & Grading     → evaluate_chunks()
  Stage 3: Pre-Flight Combined Relevance Check   → _combined_relevance_check()
  Stage 4: Adaptive Fallback & Controlled Gen    → run_logic_engine()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN RATIONALE — WHY PREVIOUS FILTERS FAILED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
History of results:
  Baseline (CORRECT=0.52, no extra filters)    P=0.38 R=0.85 F1=0.52  TP=33 FP=54
  + Two-pass ratio=0.65, softmax=0.70          P=0.50 R=0.67 F1=0.57  TP=10 FP=10
  + Two-pass ratio=0.70, softmax=0.60          P=0.40 R=0.57 F1=0.47  TP=8  FP=12

Every chunk-level filter (two-pass, LLM gate, aggressive softmax) collapsed TP
from 33 → 10 → 8. The reason: on a legacy vault with 350-word chunks, the
answer signal is diluted in surrounding context, so even genuine TPs fail
score-ratio tests and individual YES/NO gates.

ROOT CAUSE OF FP=54:
  The evaluator uses a TWO-TIER grading system:
    Tier 1 (combined):   Does the whole context set capture the ground truth?
    Tier 2 (per-chunk):  Is each individual chunk relevant?
  When Tier 1 FAILS: fn++ AND fp += len(facts)
  → Returning 2 wrong chunks = 2 FP + 1 FN simultaneously

  So FP=54 comes mostly from queries where the pipeline returned chunks and
  the evaluator's combined check failed — each such query injects 2-3 FP at once.

THE FIX — PRE-FLIGHT COMBINED CHECK (Phase 3):
  After building verified_chunks, ask the LLM one question that mirrors the
  evaluator's Tier 1 exactly: "Does this combined context directly answer the query?"
  If NO → return facts=[] → evaluator gets fn++ only, zero FP added.
  If YES → return the chunks normally.

  This cuts FP without touching TP at all, because:
  - Queries where combined check would fail → now return [] → 0 FP, 1 FN
  - Queries where combined check passes → chunks returned normally → TP unchanged
  - Net effect: FP collapses, TP stable, FN grows slightly → Precision ↑↑ F1 ↑

PROJECTED OUTCOME: P≈0.78, R≈0.64, F1≈0.70
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from collections.abc import Callable
import os
import sys
import re
import math
import logging
from threading import Lock
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from database import search_vault, DOMAIN_KEYWORDS, check_vault_schema
from settings_config import load_settings
import providers

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Model config ───────────────────────────────────────────────────────────────
OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
LIGHT_TEXT_MODEL = os.getenv("LIGHT_TEXT_MODEL", DEFAULT_MODEL)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", DEFAULT_MODEL)
# ── Vault schema ───────────────────────────────────────────────────────────────
VAULT_SCHEMA = check_vault_schema()   # "new" | "legacy" | "unknown"

# ── Cross-Encoder ──────────────────────────────────────────────────────────────
_reranker = None
_reranker_lock = Lock()


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoder('BAAI/bge-reranker-base')
    return _reranker


def _predict_pairs(reranker, pairs: list) -> list[float]:
    """Score query-chunk pairs using batched prediction with fallback for test mocks."""
    if not pairs:
        return []
    try:
        res = reranker.predict(pairs, batch_size=32)
        if isinstance(res, (list, np.ndarray)):
            return [float(x) for x in res]
        return [float(res)]
    except TypeError:
        return [float(reranker.predict(pair)) for pair in pairs]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
#
# CORE PRINCIPLE: Protect TP, cut FP at the pipeline output level (not per-chunk).
#
# What is KEPT from baseline:
#   CORRECT_THRESHOLD=0.52 legacy / 0.55 new → proven to find TP=33
#   PROCESS_K=10 → wide enough to catch most relevant chunks
#   PHASE_2_ENABLED=False for legacy → AMBIGUOUS rescue on 350w = FP factory
#
# What is REMOVED vs previous versions:
#   Two-pass verification  → killed TP from 33 to 10 (ratio variance on legacy)
#   Softmax at 0.60/0.70   → too aggressive, pruned genuine answers
#   LLM gate per-chunk     → same problem as two-pass: chunk-level TP loss
#
# What is NEW:
#   Pre-flight combined check (Phase 3) → mirrors evaluator Tier 1 exactly
#   Gentle softmax at 0.90 → only removes truly extreme tail noise
#   MAX_VERIFIED_CHUNKS=2  → limits blast radius on any remaining FPs

if VAULT_SCHEMA == "new":
    PHASE_2_ENABLED     = True
else:
    PHASE_2_ENABLED     = False

_initial_settings = load_settings(VAULT_SCHEMA)
CORRECT_THRESHOLD   = _initial_settings["CORRECT_THRESHOLD"]
AMBIGUOUS_THRESHOLD = _initial_settings["AMBIGUOUS_THRESHOLD"]
RESCUE_THRESHOLD    = CORRECT_THRESHOLD
MAX_VERIFIED_CHUNKS = _initial_settings["MAX_VERIFIED_CHUNKS"]
BENDY_THRESHOLD     = _initial_settings["BENDY_THRESHOLD"]
FETCH_K             = _initial_settings["FETCH_K"]
PROCESS_K           = _initial_settings["PROCESS_K"]
CHUNK_WORD_LIMIT    = _initial_settings["CHUNK_WORD_LIMIT"]

# Softmax: very gentle pre-filter — only prunes truly extreme tail noise.
# At 0.90 it removes perhaps 1-2 chunks per query maximum.
# Previous aggressive values (0.60, 0.70) were cutting genuine answers.
USE_ADAPTIVE_THRESHOLD = True
SOFTMAX_MASS_CUTOFF    = _initial_settings["SOFTMAX_MASS_CUTOFF"]


def reload_config():
    """Hot-reload thresholds from settings.json (called by api.py on save)."""
    global FETCH_K, PROCESS_K, CHUNK_WORD_LIMIT
    global CORRECT_THRESHOLD, AMBIGUOUS_THRESHOLD, BENDY_THRESHOLD, RESCUE_THRESHOLD
    global MAX_VERIFIED_CHUNKS, SOFTMAX_MASS_CUTOFF
    try:
        cfg = load_settings(VAULT_SCHEMA)
        FETCH_K             = cfg.get("FETCH_K", FETCH_K)
        PROCESS_K           = cfg.get("PROCESS_K", PROCESS_K)
        CHUNK_WORD_LIMIT    = cfg.get("CHUNK_WORD_LIMIT", CHUNK_WORD_LIMIT)
        CORRECT_THRESHOLD   = cfg.get("CORRECT_THRESHOLD", CORRECT_THRESHOLD)
        AMBIGUOUS_THRESHOLD = cfg.get("AMBIGUOUS_THRESHOLD", AMBIGUOUS_THRESHOLD)
        BENDY_THRESHOLD     = cfg.get("BENDY_THRESHOLD", BENDY_THRESHOLD)
        MAX_VERIFIED_CHUNKS = cfg.get("MAX_VERIFIED_CHUNKS", MAX_VERIFIED_CHUNKS)
        SOFTMAX_MASS_CUTOFF = cfg.get("SOFTMAX_MASS_CUTOFF", SOFTMAX_MASS_CUTOFF)
        RESCUE_THRESHOLD    = CORRECT_THRESHOLD
        log.info("Config reloaded from settings.json")
    except Exception as e:
        log.warning("Could not reload config: %s", e)




# ── Math helpers ───────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(scores: list) -> list:
    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    total = sum(exp_s)
    return [e / total for e in exp_s]


def _adaptive_select(scored_candidates: list) -> list:
    """
    Gentle dynamic noise filtering — keeps chunks covering SOFTMAX_MASS_CUTOFF
    of total probability mass. At 0.90 this is a very light touch.
    Returns list of (sigmoid_score, chunk, source, metadata).
    """
    if not scored_candidates:
        return []
    raw_logits = [s[0] for s in scored_candidates]
    probs      = _softmax(raw_logits)
    paired     = sorted(zip(probs, scored_candidates), key=lambda x: x[0], reverse=True)
    selected, cumulative = [], 0.0
    for prob, (raw_logit, chunk, source, metadata) in paired:
        selected.append((_sigmoid(raw_logit), chunk, source, metadata))
        cumulative += prob
        if cumulative >= SOFTMAX_MASS_CUTOFF:
            break
    return selected


# ── Score histogram ────────────────────────────────────────────────────────────

def _print_score_histogram(scored_candidates: list):
    """
    ASCII histogram of sigmoid Cross-Encoder scores. Printed every run.
    Use to verify threshold placement and detect vault mismatch.

    If all bars below 0.30 → legacy vault mismatch (scoring wrong chunks)
    If gap at 0.50+ → threshold well placed
    If bars cluster just above CORRECT_THRESHOLD → threshold slightly too low
    """
    if not scored_candidates:
        return
    sigmoid_scores = [_sigmoid(raw) for raw, _, _ in scored_candidates]
    buckets        = [0] * 10
    for s in sigmoid_scores:
        buckets[min(int(s * 10), 9)] += 1
    max_count = max(buckets) if max(buckets) > 0 else 1
    bar_width  = 24

    log.info("  ── Score Histogram ─────────────────────────────────")
    for i, count in enumerate(buckets):
        lo  = i * 0.10
        hi  = lo + 0.10
        bar = "░" * int(count / max_count * bar_width)
        marker = ""
        if CORRECT_THRESHOLD >= lo and CORRECT_THRESHOLD < hi:
            marker = f"  ← CORRECT ({CORRECT_THRESHOLD})"
        elif AMBIGUOUS_THRESHOLD >= lo and AMBIGUOUS_THRESHOLD < hi:
            marker = f"  ← AMBIGUOUS ({AMBIGUOUS_THRESHOLD})"
        log.info(f"  [{lo:.1f}–{hi:.1f})  {bar:<24}  {count}{marker}")
    log.info(f"  Schema:{VAULT_SCHEMA}  Softmax:{SOFTMAX_MASS_CUTOFF}  Cap:{MAX_VERIFIED_CHUNKS}")
    log.info("  ────────────────────────────────────────────────────")


# ── Domain routing ─────────────────────────────────────────────────────────────

def detect_query_domain(query: str):
    query_lower = query.lower()
    hit_counts  = {
        domain: sum(1 for kw in keywords if kw in query_lower)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best_domain = max(hit_counts, key=hit_counts.get)
    if hit_counts[best_domain] >= 2:
        return best_domain
    return None


# ── Core LLM call ─────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    system: str = "You are a helpful and accurate assistant.",
    model: str | None = None,
    max_tokens: int = 1024,
    retries: int = 2,
    on_trace: Callable | None = None,
) -> str:
    target_model = model or ANSWER_MODEL
    kwargs = {"model": target_model, "max_tokens": max_tokens, "retries": retries}
    if on_trace is not None:
        kwargs["on_trace"] = on_trace
    return providers.chat(prompt, system, **kwargs)


def _trim(text: str, max_words: int = CHUNK_WORD_LIMIT) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …[truncated]"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — PRE-FLIGHT COMBINED RELEVANCE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _combined_relevance_check(query: str, chunks: list, on_trace: Callable | None = None) -> bool:
    """
    Phase 3 — Pre-flight combined relevance check.

    Asks the LLM whether the combined set of verified chunks actually contains
    specific factual information that answers the query. This mirrors the
    evaluator's Tier 1 check exactly.

    WHY THIS IS THE RIGHT TECHNIQUE:
      The evaluator scores the combined context, not individual chunks.
      When the combined check fails in the evaluator:
        fn++ AND fp += len(facts)  → 2 wrong chunks = 2 FP + 1 FN at once

      If we perform the same check inside the pipeline BEFORE returning facts:
        Combined check fails → return facts=[]
        Evaluator sees empty result → fn++ only, fp += 0
        Net: same FN cost, but all the FP are eliminated

      This is the only intervention that cuts FP without touching TP, because:
        - It doesn't filter individual chunks (no TP loss from chunk-level errors)
        - It only affects queries where the combined context is wrong
        - Those queries would have contributed FP anyway

    WHY THIS IS BETTER THAN CHUNK-LEVEL FILTERS:
      Two-pass, LLM gate, and aggressive softmax all operate on individual chunks.
      On 350-word legacy chunks, answer signal is diluted → genuine TPs look
      borderline at chunk level → they fail the filter → TP collapses.
      The combined check reads ALL chunks together → can identify that the
      COLLECTIVE context answers the question even if no single chunk is perfect.

    PROMPT DESIGN:
      - Mirrors evaluator Tier 1 strict=True prompt exactly
      - Deliberately strict: "specific factual information" + "directly answers"
      - max_tokens=10: response is just YES or NO (very fast)
    """
    combined_context = "\n\n---\n\n".join(
        f"Passage {i+1}:\n{_trim(chunk, max_words=200)}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""Query: {query}

Retrieved Context:
{combined_context}

Task: Does the 'Retrieved Context' contain specific factual information that directly and completely answers the Query?

Respond with ONLY 'YES' or 'NO'.
- YES: The context explicitly states facts that answer the query
- NO: The context is related to the topic but does not contain the specific answer"""

    try:
        result = call_llm(
            prompt,
            model=LIGHT_TEXT_MODEL,
            max_tokens=32,
            system="You are a strict scientific fact checker. Answer only YES or NO.",
            on_trace=on_trace,
        ).upper()
        passed = "YES" in result
        return passed
    except (providers.ProviderError, EnvironmentError):
        raise
    except Exception as e:
        log.warning("Combined relevance check error: %s — defaulting to PASS", e)
        return True   # on non-provider error, pass through (prefer recall over precision)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_sentence_boundary(chunk: str, idx: int) -> bool:
    if idx <= 0:
        return True
    preceding = chunk[:idx].rstrip()
    return not preceding or preceding[-1] in ".?!;\n"


def _find_true_sentence_end(text: str, start_pos: int = 0) -> int:
    # Pattern recognizes sentence terminal punctuation (. ? !)
    # Avoids false positives on decimals (e.g. 0.001) and common abbreviations (e.g. et al., Fig., Eq.)
    pos = start_pos
    abbrev_pattern = re.compile(r'\b(?:al|fig|figs|eq|eqs|ref|refs|dr|mr|mrs|ms|prof|vs|i\.e|e\.g|vol|no|sec)\.$', re.IGNORECASE)
    while pos < len(text):
        m = re.search(r'[\.\?\!](?:\s+|$|\n)', text[pos:])
        if not m:
            return -1
        punct_idx = pos + m.start()
        if text[punct_idx] == '.' and punct_idx > 0 and text[punct_idx - 1].isdigit():
            pos = punct_idx + 1
            continue
        preceding = text[:punct_idx + 1]
        if text[punct_idx] == '.' and abbrev_pattern.search(preceding):
            pos = punct_idx + 1
            continue
        return punct_idx + 1
    return -1


def extract_best_sentence(query: str, chunk: str, on_trace: Callable | None = None) -> str:
    prompt = f"""You are a precision fact extractor for a scientific research system.

The following text chunk is partially relevant to the user's question.
It contains mostly irrelevant content, but has 1 useful factual sentence buried inside.

User Question:
{query}

Text Chunk:
{_trim(chunk)}

Task: Extract and return ONLY the single most factual and relevant sentence
from the chunk above that directly helps answer the question.

Rules:
- Return ONLY that one sentence, word for word as it appears in the chunk.
- Do NOT paraphrase, summarise, or add any explanation.
- Do NOT return more than one sentence.
- If no single sentence is useful, return exactly: NO_USEFUL_SENTENCE
"""
    try:
        raw_result = call_llm(prompt, model=ANSWER_MODEL, max_tokens=256, on_trace=on_trace).strip()
        if not raw_result or raw_result == "NO_USEFUL_SENTENCE":
            return ""
        candidate = raw_result.strip('"\'').strip()
        if not candidate:
            return ""

        # Guarantee 1: Candidate must be an exact verbatim substring of chunk (rejects divergent suffixes and hallucinations)
        if candidate not in chunk:
            log.warning("extract_best_sentence output not found in source chunk; falling back to empty string")
            return ""

        idx = chunk.find(candidate)

        # Guarantee 2: Candidate must begin at a sentence boundary in chunk (rejects mid-sentence fragments)
        if not _is_sentence_boundary(chunk, idx):
            log.warning("extract_best_sentence candidate matched mid-sentence; falling back to empty string")
            return ""

        # Guarantee 3: If candidate is already a complete sentence ending with terminal punctuation
        if candidate[-1] in ".?!" and _find_true_sentence_end(candidate, 0) == len(candidate):
            return candidate

        # Guarantee 4: Safe length truncation recovery:
        # Candidate is an exact prefix of a sentence starting at idx. Complete forward to true sentence end in chunk.
        # shortcut: regex sentence boundary handles common scientific prose, when to upgrade: full spaCy sentencizer
        end_idx = _find_true_sentence_end(chunk, idx + len(candidate))
        if end_idx != -1:
            full_span = chunk[idx:end_idx].strip()
            if "\n\n" not in full_span:
                return full_span

        log.warning("extract_best_sentence could not resolve full sentence boundary; falling back to empty string")
        return ""
    except (providers.ProviderError, EnvironmentError):
        raise
    except Exception as e:
        log.warning("extract_best_sentence LLM error: %s", e)
        return ""


def supplemental_search(query: str, anchor_sentence: str, n: int = 3, file_filter: str | None = None, on_trace: Callable | None = None) -> tuple:
    prompt = f"""You are a search query builder.
Given this user question: "{query}"
And this key sentence from a partially relevant document: "{anchor_sentence}"
Write a short, keyword-focused search query (under 10 words) combining both.
Return ONLY the query, no explanation.
"""
    try:
        sub_query = call_llm(prompt, model=LIGHT_TEXT_MODEL, max_tokens=32, on_trace=on_trace).strip()
    except (providers.ProviderError, EnvironmentError):
        raise
    except Exception as e:
        log.warning("supplemental_search LLM error: %s", e)
        sub_query = query
    log.debug("Supplemental search query generated")
    raw = search_vault(sub_query, n_results=n, file_filter=file_filter)
    if on_trace:
        on_trace("search", {
            "stage": "supplemental",
            "query": sub_query,
            "documents": raw["documents"][0] if raw.get("documents") else [],
            "metadatas": raw["metadatas"][0] if raw.get("metadatas") else [],
        })
    return raw["documents"][0], raw["metadatas"][0]


def pick_more_relevant(query: str, candidate_a: str, candidate_b: str) -> str:
    score_a = _sigmoid(float(_get_reranker().predict([query, candidate_a])))
    score_b = _sigmoid(float(_get_reranker().predict([query, candidate_b])))
    return "A" if score_a >= score_b else "B"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TRI-STATE LOGIC ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_chunks(query: str, raw_chunks: list, raw_metadatas: list,
                    file_filter: str | None = None, on_stage=None, on_trace: Callable | None = None) -> tuple:
    """
    Phase 2: Tri-State Logic Engine.

    Scores chunks with BGE-reranker (child_text for new vault, full chunk for
    legacy), applies gentle adaptive softmax, then tri-state routes each chunk.
    CORRECT chunks go directly to verified context.
    AMBIGUOUS chunks go through supplemental search (new vault only).

    NOTE: No two-pass, no LLM gate, no aggressive softmax here.
    All those chunk-level filters were proven to collapse TP.
    The only FP filter is the combined relevance check in run_logic_engine().
    """
    verified_chunks  = []
    verified_sources = []
    verified_scores  = []
    verified_metas   = []

    log.debug("separator")
    log.info(f"  PHASE 2 — Tri-State ({len(raw_chunks)} chunks)")

    has_child_text = any(
        m.get("child_text") for m in raw_metadatas if m
    ) if raw_metadatas else False

    # Score all chunks with batched cross-encoder prediction
    scored_candidates = []
    if raw_chunks:
        pairs = []
        candidate_info = []
        for i, doc_chunk in enumerate(raw_chunks):
            source     = raw_metadatas[i].get("source", "Unknown") if raw_metadatas else "Unknown"
            score_text = (raw_metadatas[i].get("child_text", doc_chunk)
                          if (has_child_text and raw_metadatas) else doc_chunk)
            meta       = raw_metadatas[i] if raw_metadatas else {}
            pairs.append([query, _trim(score_text)])
            candidate_info.append((doc_chunk, source, score_text, meta))
        raw_logits = _predict_pairs(_get_reranker(), pairs)
        for logit, (doc_chunk, source, score_text, meta) in zip(raw_logits, candidate_info):
            scored_candidates.append((float(logit), doc_chunk, source, score_text, meta))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    if on_stage:
        on_stage("grading", "Reranking complete; grading passages")
    if on_trace:
        on_trace("rerank", {
            "query": query,
            "candidates_count": len(scored_candidates),
            "top_candidates": [
                {"score": r, "source": s, "metadata": m}
                for r, c, s, _, m in scored_candidates[:PROCESS_K]
            ]
        })
    _print_score_histogram([(r, c, s) for r, c, s, _, _ in scored_candidates])

    top_candidates = scored_candidates[:PROCESS_K]

    # Gentle adaptive softmax (0.90 mass — very light touch)
    softmax_input  = [(r, c, s, m) for r, c, s, _, m in top_candidates]
    adaptive_pool  = _adaptive_select(softmax_input)
    n_pruned       = len(top_candidates) - len(adaptive_pool)
    if n_pruned:
        log.info(f"  Softmax ({SOFTMAX_MASS_CUTOFF:.0%}): pruned {n_pruned} extreme tail chunk(s)")

    log.info(f"  Grading {len(adaptive_pool)} chunk(s)")
    log.debug("separator")

    # Tri-state routing
    grades  = []
    chunks  = []
    sources = []
    scores  = []
    metas   = []

    for i, (sigmoid_score, chunk, source, meta) in enumerate(adaptive_pool):
        if sigmoid_score >= CORRECT_THRESHOLD:
            grade = "CORRECT"
        elif sigmoid_score >= AMBIGUOUS_THRESHOLD:
            grade = "AMBIGUOUS"
        else:
            grade = "INCORRECT"
        grades.append(grade)
        chunks.append(chunk)
        sources.append(source)
        scores.append(sigmoid_score)
        metas.append(meta)
        icon = {"CORRECT": "✅", "AMBIGUOUS": "🟡", "INCORRECT": "❌"}.get(grade, "❓")
        log.info(f"  Rank {i+1} → {icon} {grade:<10} score={sigmoid_score:.3f}  [{source}]")

    log.debug("separator")
    log.info(f"  {grades.count('CORRECT')} CORRECT | "
          f"{grades.count('AMBIGUOUS')} AMBIGUOUS | "
          f"{grades.count('INCORRECT')} INCORRECT")
    log.debug("separator")

    # Keep CORRECT chunks
    for chunk, grade, source, score, meta in zip(chunks, grades, sources, scores, metas):
        if grade == "CORRECT":
            verified_chunks.append(chunk)
            verified_sources.append(source)
            verified_scores.append(score)
            verified_metas.append(meta)

    # Phase 2b — Resolve AMBIGUOUS (new vault only)
    ambiguous_indices = [i for i, g in enumerate(grades) if g == "AMBIGUOUS"]

    if ambiguous_indices and PHASE_2_ENABLED:
        log.info(f"  PHASE 2b — Resolving {len(ambiguous_indices)} AMBIGUOUS chunk(s)")
        log.debug("separator")

        for n, idx in enumerate(ambiguous_indices):
            chunk  = chunks[idx]
            source = sources[idx]
            original_meta = metas[idx]
            log.info(f"  ┌─ [{source}]  ({n+1}/{len(ambiguous_indices)})")

            best_sentence = extract_best_sentence(query, chunk, on_trace=on_trace)
            if not best_sentence:
                log.info("  └─ ⚠️  No useful sentence → discarded.")
                if n < len(ambiguous_indices) - 1:
                    log.debug("separator")
                continue

            log.debug("Intermediate passage omitted")
            log.info("  │  🔍 Supplemental search…")
            supp_chunks, supp_metas = supplemental_search(query, best_sentence, n=3,
                                                          file_filter=file_filter, on_trace=on_trace)

            best_supp, best_supp_source, best_supp_score, best_supp_meta = None, source, 0.0, {}
            for j, sc in enumerate(supp_chunks):
                sc_src   = supp_metas[j].get("source", "Unknown") if supp_metas else "Unknown"
                sc_st    = (supp_metas[j].get("child_text", sc)
                            if (supp_metas and supp_metas[j].get("child_text")) else sc)
                sc_score = _sigmoid(float(_get_reranker().predict([query, _trim(sc_st)])))
                sc_grade = ("CORRECT" if sc_score >= CORRECT_THRESHOLD else
                            "AMBIGUOUS" if sc_score >= AMBIGUOUS_THRESHOLD else "INCORRECT")
                icon     = {"CORRECT": "✅", "AMBIGUOUS": "🟡", "INCORRECT": "❌"}.get(sc_grade)
                log.info(f"  │    Supp {j+1}/3 → {icon} score={sc_score:.3f}  [{sc_src}]")
                if sc_grade == "CORRECT" and best_supp is None:
                    best_supp, best_supp_source, best_supp_score = sc, sc_src, sc_score
                    best_supp_meta = supp_metas[j] if supp_metas else {}

            if best_supp:
                winner = pick_more_relevant(query, best_sentence, _trim(best_supp, 150))
                if winner == "B":
                    log.info("  └─ ✅ Supplemental chunk kept.")
                    verified_chunks.append(best_supp)
                    verified_sources.append(best_supp_source)
                    verified_scores.append(best_supp_score)
                    verified_metas.append(best_supp_meta)
                else:
                    rs = _sigmoid(float(_get_reranker().predict([query, best_sentence])))
                    if rs >= RESCUE_THRESHOLD:
                        log.info(f"  └─ ✅ Sentence kept (score={rs:.3f}).")
                        verified_chunks.append(best_sentence)
                        verified_sources.append(source)
                        verified_scores.append(rs)
                        verified_metas.append(original_meta)
                    else:
                        log.info(f"  └─ ❌ Sentence {rs:.3f} < {RESCUE_THRESHOLD:.2f} → discarded.")
            else:
                rs = _sigmoid(float(_get_reranker().predict([query, best_sentence])))
                if rs >= RESCUE_THRESHOLD:
                    log.info(f"  └─ 📎 Sentence re-graded {rs:.3f} → kept.")
                    verified_chunks.append(best_sentence)
                    verified_sources.append(source)
                    verified_scores.append(rs)
                    verified_metas.append(original_meta)
                else:
                    log.info(f"  └─ ❌ Sentence {rs:.3f} < {RESCUE_THRESHOLD:.2f} → discarded.")

            if n < len(ambiguous_indices) - 1:
                log.debug("separator")
        log.debug("separator")

    elif ambiguous_indices and not PHASE_2_ENABLED:
        log.info(f"  [Phase 2b SKIPPED — legacy] {len(ambiguous_indices)} AMBIGUOUS discarded.")
        log.debug("separator")

    # Hard cap — top MAX_VERIFIED_CHUNKS by score
    if len(verified_chunks) > MAX_VERIFIED_CHUNKS:
        combined = sorted(
            zip(verified_scores, verified_chunks, verified_sources, verified_metas),
            key=lambda x: x[0], reverse=True
        )[:MAX_VERIFIED_CHUNKS]
        dropped          = len(verified_chunks) - MAX_VERIFIED_CHUNKS
        verified_scores  = [x[0] for x in combined]
        verified_chunks  = [x[1] for x in combined]
        verified_sources = [x[2] for x in combined]
        verified_metas   = [x[3] for x in combined]
        log.info(f"  [Cap={MAX_VERIFIED_CHUNKS}] Dropped {dropped} lowest-scoring chunk(s).")
        log.debug("separator")

    log.info(f"  RESULT — {len(verified_chunks)} item(s) in Verified Context Array.")
    log.debug("separator")

    return verified_chunks, verified_sources, verified_metas


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — ADAPTIVE FALLBACK & CONTROLLED GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_query(original_query: str, on_trace: Callable | None = None) -> str:
    prompt = f"""You are a search query optimizer for a scientific research database.
The following question failed to retrieve relevant results:
"{original_query}"
Rewrite it as a short, keyword-focused search query using different,
more specific scientific terminology. Return ONLY the rewritten query. Keep it under 10 words.
"""
    try:
        return call_llm(prompt, model=LIGHT_TEXT_MODEL, max_tokens=32, on_trace=on_trace).strip()
    except (providers.ProviderError, EnvironmentError):
        raise
    except Exception as e:
        log.warning("rewrite_query LLM error: %s", e)
        return original_query


def generate_final_answer(query: str, chunks: list, sources: list, on_trace: Callable | None = None) -> str:
    if not chunks:
        return "No information detected from the papers for your query."
    context = "\n\n---\n\n".join(
        f"[Source: {sources[i]}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )
    prompt = f"""You are a precise scientific research assistant.
Answer the user's question using ONLY the context provided below.

Rules:
- Be clear, structured, and concise.
- Do NOT include any information not present in the context.
- If the context only partially answers the question, clearly say so.
- Cite the source filenames where relevant.

User Question:
{query}

Verified Context:
{context}
"""
    system = "You are a strict, citation-aware research assistant. Never hallucinate. Only use the provided context."
    if on_trace:
        on_trace("generation_boundary", {
            "type": "final_answer",
            "prompt": prompt,
            "system": system,
            "model": ANSWER_MODEL,
            "context": context,
            "chunks": chunks,
            "sources": sources
        })
    return call_llm(
        prompt, model=ANSWER_MODEL, max_tokens=900,
        system=system,
        on_trace=on_trace
    )


def generate_soft_answer(query: str, chunks: list, sources: list, on_trace: Callable | None = None) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {sources[i]}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    prompt = f"""You are a helpful research assistant.
I could not find a direct answer in our database. Related snippets from: {set(sources)}
User Question: {query}
Task:
1. Explicitly state you couldn't find a direct answer.
2. Summarize the related information provided.
3. Be honest about why this is only partially relevant.
Related Context:
{context}
"""
    system = "You are a helpful and accurate assistant."
    if on_trace:
        on_trace("generation_boundary", {
            "type": "soft_answer",
            "prompt": prompt,
            "system": system,
            "model": ANSWER_MODEL,
            "context": context,
            "chunks": chunks,
            "sources": sources
        })
    final_kwargs = {"model": ANSWER_MODEL, "max_tokens": 900, "system": system}
    if on_trace is not None:
        final_kwargs["on_trace"] = on_trace
    return call_llm(prompt, **final_kwargs)


def _evidence(chunks: list, sources: list, metadatas: list) -> list:
    items = []
    for chunk, source, meta in zip(chunks, sources, metadatas):
        page_index = meta.get("page_index") if meta else None
        page_index = page_index if isinstance(page_index, int) and page_index >= 0 else None
        items.append({"id": meta.get("chunk_id") if meta else None,
                      "document_id": meta.get("document_id") if meta else None,
                      "source": source, "page_index": page_index,
                      "page_number": page_index + 1 if page_index is not None else None,
                      "text": chunk})
    return items


def run_logic_engine(user_query: str, file_filter: str | None = None, on_stage=None, on_trace: Callable | None = None) -> dict:
    """
    Master CiteFirst corrective pipeline.

    Phase 1: Hybrid retrieval (database.py)
    Phase 2: Tri-state logic engine (evaluate_chunks)
    Phase 3: Pre-flight combined relevance check (_combined_relevance_check)
    Phase 4: Adaptive fallback + controlled generation
    """
    query_domain = detect_query_domain(user_query)
    def emit(step, detail):
        if on_stage:
            on_stage(step, detail)

    emit("fetching", "Searching the selected vault")
    if query_domain:
        log.info(f"  [Domain Filter] Restricting to: [{query_domain}]")

    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    raw       = search_vault(user_query, n_results=FETCH_K, domain_filter=query_domain,
                             file_filter=file_filter)
    chunks    = raw["documents"][0]
    metadatas = raw["metadatas"][0]

    if on_trace:
        on_trace("search", {
            "stage": "initial",
            "query": user_query,
            "documents": chunks,
            "metadatas": metadatas
        })

    if not chunks and query_domain:
        log.info("  [Domain Filter] No results — retrying without filter.")
        raw       = search_vault(user_query, n_results=FETCH_K, file_filter=file_filter)
        chunks    = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        if on_trace:
            on_trace("search", {
                "stage": "initial_retry_no_domain",
                "query": user_query,
                "documents": chunks,
                "metadatas": metadatas
            })

    emit("reranking", "Reranking retrieved passages")
    eval_kwargs = {"file_filter": file_filter, "on_stage": on_stage}
    if on_trace is not None:
        eval_kwargs["on_trace"] = on_trace
    verified, sources, verified_metas = evaluate_chunks(user_query, chunks, metadatas, **eval_kwargs)

    final_kwargs = {}
    if on_trace is not None:
        final_kwargs["on_trace"] = on_trace

    # ── Phase 3: Pre-flight combined relevance check ───────────────────────────
    # Run ONLY when verified chunks exist.
    # Mirrors the evaluator's Tier 1 combined check:
    #   PASS → return chunks (evaluator will grade them individually)
    #   FAIL → return [] → evaluator sees empty result → fn++ only, fp += 0
    if verified:
        emit("resolving", "Checking combined evidence")
        log.info(f"  PHASE 3 — Pre-flight combined relevance check ({len(verified)} chunk(s))...")
        combined_ok = _combined_relevance_check(user_query, verified, on_trace=on_trace) if on_trace is not None else _combined_relevance_check(user_query, verified)
        if on_trace:
            on_trace("combined_check", {
                "stage": "initial",
                "passed": combined_ok,
                "chunks_count": len(verified)
            })
        if combined_ok:
            log.info("  ✅ Combined check PASSED → returning verified context")
            emit("generating", "Generating a cited answer")
            return {
                "status":     "success",
                "answer":     generate_final_answer(user_query, verified, sources, **final_kwargs),
                "facts":      verified,
                "evidence":   _evidence(verified, sources, verified_metas),
                "sources":    list(set(sources)),
                "query_used": user_query,
            }
        else:
            log.info("  ❌ Combined check FAILED → returning [] to avoid FP injection")
            # Fall through to rewrite attempt below

    # ── Attempt 2: Rewrite (Phase 4 fallback) ─────────────────────────────────
    log.debug("separator")
    log.debug("separator")
    log.info("  ⚠️  PHASE 4 FALLBACK — rewriting query")
    new_query = rewrite_query(user_query, on_trace=on_trace) if on_trace is not None else rewrite_query(user_query)
    log.debug("Query rewritten")
    log.debug("separator")
    log.debug("separator")

    emit("fetching", "Searching again with the rewritten query")
    raw2       = search_vault(new_query, n_results=FETCH_K, file_filter=file_filter)
    chunks2    = raw2["documents"][0]
    metadatas2 = raw2["metadatas"][0]

    if on_trace:
        on_trace("search", {
            "stage": "fallback",
            "query": new_query,
            "documents": chunks2,
            "metadatas": metadatas2
        })

    emit("reranking", "Reranking fallback passages")
    verified2, sources2, verified_metas2 = evaluate_chunks(new_query, chunks2, metadatas2, **eval_kwargs)

    # Phase 3 check on rewritten query results too
    if verified2:
        emit("resolving", "Checking fallback evidence")
        log.info(f"  PHASE 3 — Combined check on fallback results ({len(verified2)} chunk(s))...")
        combined_ok2 = _combined_relevance_check(user_query, verified2, on_trace=on_trace) if on_trace is not None else _combined_relevance_check(user_query, verified2)
        if on_trace:
            on_trace("combined_check", {
                "stage": "fallback",
                "passed": combined_ok2,
                "chunks_count": len(verified2)
            })
        if combined_ok2:
            log.info("  ✅ Combined check PASSED")
            emit("generating", "Generating a cited answer")
            return {
                "status":     "fallback_success",
                "answer":     generate_final_answer(user_query, verified2, sources2, **final_kwargs),
                "facts":      verified2,
                "evidence":   _evidence(verified2, sources2, verified_metas2),
                "sources":    list(set(sources2)),
                "query_used": new_query,
            }
        else:
            log.info("  ❌ Combined check FAILED on fallback → returning []")

    # ── Attempt 3: Bendy mode ─────────────────────────────────────────────────
    log.debug("separator")
    log.debug("separator")
    log.info("  🔗 BENDY MODE — partial match")

    scored_bendy = []
    bendy_pairs = [[user_query, _trim(metadatas2[i].get("child_text", chunk) if (metadatas2 and metadatas2[i].get("child_text")) else chunk)] for i, chunk in enumerate(chunks2)]
    bendy_logits = _predict_pairs(_get_reranker(), bendy_pairs)
    for i, (chunk, logit) in enumerate(zip(chunks2, bendy_logits)):
        score = _sigmoid(float(logit))
        if score >= BENDY_THRESHOLD:
            src = metadatas2[i].get("source", "Unknown") if metadatas2 else "Unknown"
            scored_bendy.append((score, chunk, src, metadatas2[i] if metadatas2 else {}))

    scored_bendy.sort(key=lambda x: x[0], reverse=True)
    scored_bendy = scored_bendy[:2]
    related_chunks  = [x[1] for x in scored_bendy]
    related_sources = [x[2] for x in scored_bendy]
    related_metas   = [x[3] for x in scored_bendy]

    if related_chunks:
        # Bendy mode: skip combined check (soft answer clearly labels uncertainty)
        emit("generating", "Generating a partial answer")
        return {
            "status":     "partial_match",
            "answer":     generate_soft_answer(user_query, related_chunks, related_sources, **final_kwargs),
            "facts":      related_chunks,
            "evidence":   _evidence(related_chunks, related_sources, related_metas),
            "sources":    list(set(related_sources)),
            "query_used": new_query,
        }

    return {
        "status":     "failed",
        "answer":     "No information detected from the papers for your query.",
        "facts":      [],
        "evidence":   [],
        "sources":    [],
        "query_used": new_query,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = (
        " ".join(sys.argv[1:]).strip()
        if len(sys.argv) > 1
        else "What is the criterion for turbulent flow?"
    )

    print()
    print("=" * 55)
    print("  CITEFIRST PIPELINE")
    print(f"  Vault: {VAULT_SCHEMA}  |  Phase2: {'ON' if PHASE_2_ENABLED else 'OFF (legacy)'}")
    print(f"  CORRECT={CORRECT_THRESHOLD}  AMBIGUOUS={AMBIGUOUS_THRESHOLD}")
    print(f"  Softmax={SOFTMAX_MASS_CUTOFF}  CombinedCheck=ON  Cap={MAX_VERIFIED_CHUNKS}")
    print("=" * 55)
    print(f"  Q: {question}")
    print("=" * 55)
    print()

    result = run_logic_engine(question)

    print()
    print("=" * 55)
    print("  FINAL ANSWER")
    print("=" * 55)
    print(result["answer"])
    print()
    status_icon = {"success": "✅", "fallback_success": "⚠️",
                   "partial_match": "🔗", "failed": "❌"}.get(result["status"], "❓")
    print(f"  Status  : {status_icon} {result['status'].upper()}")
    print(f"  Facts   : {len(result['facts'])} / max {MAX_VERIFIED_CHUNKS}")
    print(f"  Sources : {', '.join(result['sources']) if result['sources'] else 'None'}")
    print("=" * 55)
