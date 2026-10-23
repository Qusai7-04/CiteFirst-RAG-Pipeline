# CiteFirst Pipeline Evaluation & Benchmark Report

- **Date:** 2026-10-21
- **Provider:** OpenRouter (`meta-llama/llama-3.3-70b-instruct`)
- **Pipeline Architecture:** Hybrid dense (BGE-small-en-v1.5) + BM25 retrieval, BGE-reranker-base cross-encoder, adaptive softmax context selection, CRAG corrective routing.
- **Evaluation Baseline:** Consolidated Phase 4 benchmark on main pipeline

---

## 1. Overview & Evaluation Scope

CiteFirst is a citation-grounded RAG application for answering questions across research papers. This report provides a consolidated, factual evaluation of both retrieval and answer generation performance across 26 benchmark questions spanning 23 research papers.

### Key Headline Results

- **Known-Source Retrieval Hit@1:** 94.4% pre-rerank (17/18), 100.0% post-rerank (18/18).
- **Emitted Paper-Level Citation Rate:** 100.0% (26/26 questions cited the target paper).
- **Approved Physical Page Presence in Generation Context:** 73.1% (19/26 overall; 15/22 Development, 4/4 Disclosed Pilot).
- **Human-Verified Answer Acceptance:**
  - **Development Partition (Q001–Q022):** **9.1% ACCEPT (2/22)**, 40.9% PARTIAL (9/22), 50.0% REJECT (11/22).
  - **Disclosed Pilot Partition (Q023–Q026):** **100.0% ACCEPT (4/4)**, 0% PARTIAL (0/4), 0% REJECT (0/4).
  - **Combined Descriptive Total:** **23.1% ACCEPT (6/26)**, 34.6% PARTIAL (9/26), 42.3% REJECT (11/26).
- **Claim-Level Coverage (N=83 required claims):** 23 Covered (27.7%), 21 Partial (25.3%), 39 Omitted (47.0%).
- **Faithfulness:** 23 Faithful (88.5%), 3 Partially Faithful (11.5%), 0 Unfaithful.

> [!IMPORTANT]
> **Production Status & Citation Verification:**
> CiteFirst is an experimental research system. The 2/22 (9.1%) development answer acceptance rate demonstrates that the pipeline is not production-certified. Emitted citations and answers require human verification.

---

## 2. Evaluation Partitions & Benchmark Datasets

The evaluation operates on two distinct, phase-neutral datasets serving specific purposes:

### 2.1 Partition Definitions

1. **Development Set (Q001–Q022, 19 papers):**
   - **Q001–Q018 (15 papers):** Historical baseline questions originally defined in `reviewed_dataset.json`.
   - **Q019–Q022 (4 papers):** Human-verified questions across unrepresented papers (GANs `1406.2661v1.pdf`, Adam `1412.6980v9.pdf`, ResNet `1512.03385v1.pdf`, CRAG `2401.15884v3.pdf`).
   - *Status:* Used for iterative development and diagnostic tuning. Not a blind holdout.
2. **Disclosed Previously Exposed Pilot Set (Q023–Q026, 4 papers):**
   - Questions across GFS (`945445.945450.pdf`), SELF-RAG (`2310.11511v1.pdf`), AMA (`AMA_AdaptiveMemory...`), and CGM (`2603.25600v1.pdf`).
   - *Disclosure:* Retrieval candidates for these questions were inspected during diagnostic exploration prior to generation execution. They are disclosed strictly as an exposed pilot, **not** an untouched holdout.

### 2.2 Active Datasets in Repository

| Dataset | Path | Records | Role |
|---|---|:---:|---|
| **Canonical Benchmark** | [`docs/evaluation/phase4_questions.json`](phase4_questions.json) | 26 | Canonical 26-question benchmark set. Q001–Q018 contain verified question-level evidence (`relevant_evidence`); Q019–Q026 contain itemized claim supports (`claim_supports`). All 26 questions have independent signed answer verdicts. |
| **Historical Baseline** | [`reviewed_dataset.json`](../../reviewed_dataset.json) | 18 | Preserved for baseline retrieval comparison against historical Phase 3 measurements. |

---

## 3. Retrieval Performance & Cross-Encoder Reranking

Evaluated across the 18 historical development questions (`reviewed_dataset.json`) on the 66-paper corpus (7,816 child chunks, 2,418 parent chunks).

### 3.1 Retrieval Variant Comparison (N=18)

| Metric | Dense-Only (`bge-small-en-v1.5`) | BM25-Only | Hybrid (Dense + BM25) | Hybrid + BGE Reranker |
|---|:---:|:---:|:---:|:---:|
| **Known-Source Hit@1** | 0.9444 (17/18) | 0.9444 (17/18) | 0.9444 (17/18) | **1.0000 (18/18)** |
| **Known-Source Hit@3** | 0.8333 (5/6 avail) | 1.0000 (16/16 avail) | 1.0000 (9/9 avail) | **1.0000 (9/9 avail)** |
| **Known-Source MRR** | 0.9583 | 0.9722 | 0.9722 | **1.0000** |
| **Survival in Adaptive Context** | — | — | 0.9444 (17/18) | **0.9444 (17/18)** |
| **Mean Timing (CPU)** | 0.19s | 2.60s | 1.39s | 36.13s |

### 3.2 Q015 RRF Inversion & Reranker Promotion

For Q015 (*RAG architectures and training*), target paper `2005.11401v4.pdf` placed rank 2 in the raw hybrid candidate pool behind `2002.08909v1.pdf` (REALM).
- **Cause:** Reciprocal Rank Fusion (RRF) rewarded REALM for balanced top-5 placement across both dense (rank 3) and BM25 (rank 5) channels ($RRF = 0.031258$), whereas the top RAG chunk had dense rank 12 and BM25 rank 3 ($RRF = 0.029762$).
- **Resolution:** The `BAAI/bge-reranker-base` cross-encoder scored the approved RAG chunk highest of all 30 candidates, successfully promoting `2005.11401v4.pdf` from rank 2 to **rank 1**.

---

## 4. End-to-End Generation & Human Verdict Audit

Live generation was executed against the single active provider OpenRouter using `meta-llama/llama-3.3-70b-instruct`.
- **Frozen Generation Worksheet (`phase4_generation_worksheet.json`):** Captures the exact prompt settings, model ID, raw outputs, latency, tokens, and context at execution time. Fields marked `PENDING_HUMAN_REVIEW` reflect generation-time audit status before human scoring, not unfinished work.
- **Completed Human Verdicts (`phase4_human_answer_verdicts.json`):** Records completed human evaluation across all 26 outputs independently evaluated and signed by reviewer Qusai on 2026-10-21 against the authoritative signed verdict form.

### 4.1 Grounding Levels Disaggregation

1. **Target Filename Citation (100.0%, 26/26):** Every generated answer emitted the correct target PDF name. This is purely lexical citation emission.
2. **Approved Physical Page Presence (73.1%, 19/26):** The target physical page containing supporting evidence was present in the final context passed to the LLM for 19 of 26 questions (15/22 Development, 4/4 Pilot).
3. **Human Answer Acceptance (23.1%, 6/26):** An answer was marked ACCEPT only when all required claims were factually correct, grounded in context, and directly answered the prompt.

### 4.2 Partitioned Human Verdicts

| Partition | QIDs | ACCEPT | PARTIAL | REJECT | Factual Correct (C / PC / I) | Faithfulness (F / PF / U) | Citation Support (S / PS / U) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Development** | Q001–Q022 | **2 (9.1%)** | **9 (40.9%)** | **11 (50.0%)** | 2 / 12 / 8 | 19 / 3 / 0 | 16 / 6 / 0 |
| **Disclosed Pilot** | Q023–Q026 | **4 (100.0%)** | **0 (0.0%)** | **0 (0.0%)** | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| **Total** | Q001–Q026 | **6 (23.1%)** | **9 (34.6%)** | **11 (42.3%)** | **6 / 12 / 8** | **23 / 3 / 0** | **20 / 6 / 0** |

### 4.3 Claim-Level Analysis (N=83 Claims)

- **Covered (Directly answered with evidence):** 23 claims (27.7%) [Dev: 14/74 (18.9%), Pilot: 9/9 (100.0%)]
- **Partial (Partially answered or minor omissions):** 21 claims (25.3%) [Dev: 21/74 (28.4%), Pilot: 0/9 (0.0%)]
- **Omitted (Claim not addressed in generated answer):** 39 claims (47.0%) [Dev: 39/74 (52.7%), Pilot: 0/9 (0.0%)]

---

## 5. Primary Failure Modes

Detailed analysis of the 20 non-accepted answers revealed four distinct failure modes:

1. **Retrieval Misses (Approved Page Absent from Context, 7/26 cases):**
   - For Q005, Q006, Q008, Q009, Q011, Q020, and Q022, the approved evidence page was not retrieved into the top-K context passed to generation.
   - *Example:* For Q008, the context contained pages 2 and 3 rather than approved pages 4, 5, 8. For Q009, approved page 1 placed at rank 10 and was pruned by the 0.90 softmax threshold.
2. **Soft Refusals with Relevant Context Present (6/26 cases):**
   - For Q001, Q002, Q003, Q013, Q014, and Q017, the model produced template-like soft refusals (*"Direct Answer Not Found"* or hedging) even though relevant paper context was present.
   - *Example:* For Q001, page 17 was present in chunk 2, but the model opened with a refusal and omitted all required claims.
3. **Algorithm Parameter Confusion (Q020):**
   - For Q020 (Adam hyperparameters), the retrieved context included Section 7 (AdaMax on page 9) rather than Algorithm 1 on pages 2–3. The model faithfully quoted AdaMax parameters, resulting in an `Incorrect` factual correctness verdict (`PARTIAL` overall).
4. **Pipeline Route vs. Human Acceptance Divergence:**
   - Seven questions executed on the pipeline `success` route (Q004, Q007, Q010, Q018, Q019, Q020, Q022) but received human `PARTIAL` or `REJECT` verdicts due to incomplete claim coverage. A technical `success` route indicates retrieval confidence, not answer completeness.

---

## 6. Cryptographic Artifact Provenance & Private Archive

Authoritative signed evaluation artifacts and frozen historical records are durably preserved outside Git history in `CiteFirst_Private_Archive/` (an offline, durable private archive directory).

### 6.1 Checksums & Verified Hashes

| Artifact Description | Location / Mode | Git Blob LF SHA-256 | Working Copy (CRLF) SHA-256 | Bytes (LF / CRLF) |
|---|---|---|---|---|
| **Frozen Generation Worksheet** | [`docs/evaluation/phase4_generation_worksheet.json`](phase4_generation_worksheet.json) | `7a21710b9172bc791af1d922d54589b79f23a60aed36b2fb1faf8efa6b084914` | `0027f9d3697587b3e036bf4d280bba9e58c4e228f1bae5fe53f8943a304167bb` | 342,895 / 345,467 |
| **Structured Human Answer Verdicts** | [`docs/evaluation/phase4_human_answer_verdicts.json`](phase4_human_answer_verdicts.json) | `775af36885de9d90dd5ecc6863697f2eb6271a12d827d08961741d17e5bb7850` | `2eba4bee92fefe8dfd73fe89a2424bc4928c073a83d3afee79279d6e035d82df` | 82,934 / 84,357 |
| **Canonical 26-Question Benchmark** | [`docs/evaluation/phase4_questions.json`](phase4_questions.json) | `e5abc5ffb5c10f7c6ffc8e926c294ae035a6143e2cd1eeb279b0b47afab97ce1` | `a9effa8fab40b3b2c3cf8a255f94395de84f24a9610b7fe46965b0392d0de25a` | 118,293 / 119,728 |
| **Historical 18-Question Baseline** | [`reviewed_dataset.json`](../../reviewed_dataset.json) | `11b467995f423495e53207539c20764cf8d531b71a3162de7b5c8a3d57b6417b` | `395b11567576562e5528c52801f0c8da79f86cbd66fc8e7ec0e1dad8ea847487` | 128,112 / 128,903 |
| **66 Indexed Papers (Manifest Digest)** | `papers/` manifest | `4035c42aeb626527ef714fd3105fdab3157a095bae64f782d84add65558e2583` | `4035c42aeb626527ef714fd3105fdab3157a095bae64f782d84add65558e2583` | — |

### 6.2 Authoritative Signed Forms Preserved in Private Archive

| Form Description | Archive Filename | Authoritative SHA-256 Hash | Bytes |
|---|---|---|---|
| **Authoritative Signed Answer Verdict Form** | `phase4_human_answer_verdict_form.md` | `172bb3a46973534430aa2c2e909cac76a76d0801c2618dbd22ca80a152ceeb08` | 143,143 |
| **Authoritative Signed Evidence Sign-Off Form** | `phase4_human_evidence_signoff.md` | `4408fd4d2b0be0c9a030c02a34ef99984e3396bd5a468eaa0383b13b25c0b84d` | 34,354 |
| **Normalized Evidence Sign-Off Form** | `docs/evaluation/phase4_human_evidence_signoff.md` | `f23f20587167017c3f74642f8f00afee88bf91f89f0055170740847fbbcf413b` (LF) / `b3354f08946b918c26b35c0b627dfcdec2379626bb02aba1f5c16239ab107d5f` (CRLF) | 36,503 |

> [!NOTE]
> **Reconciliation of Evaluated Checksums:**
> Earlier intermediate drafts of this report listed placeholder hashes `a8ee8db6...` and `abdf710a...` for the worksheet and verdict JSON files. Those values matched neither checked-in Git LF bytes nor local Windows CRLF bytes. Direct byte inspection reconciles that the checked-in Git blobs possess LF SHA-256 hashes `7a21710b...` and `775af368...`, and that local CRLF working copies (`0027f9d3...` and `2eba4bee...`) are 100% semantically equal when parsed as JSON (`git == wt == archive` is `True`).

---

## 7. Limitations & Reproducibility Guidelines

1. **Non-Exhaustive Relevance Annotations:** The 16 signed secondary relevance judgments cover evaluated candidate pairs only. They do not constitute an exhaustive corpus-wide binary labeling.
2. **Partition Integrity & Pilot Disclosure:** Development questions (Q001–Q022) were used iteratively. While `docs/evaluation/phase4_split_manifest.json` originally designated Q023–Q026 as a reserved held-out split, candidate pools were inspected during question formulation; Q023–Q026 are therefore disclosed as an exposed pilot, not a blind holdout.
3. **Corpus & Index Exclusions:** The reported benchmark was evaluated against a 66-paper academic corpus (7,816 chunks) stored in a local ChromaDB index and BM25 cache. In keeping with copyright and repository size hygiene, PDF source files and binary index caches are excluded from this source tree. A fresh git clone will not reproduce historical retrieval numbers without ingesting the corpus into a local vault.
4. **Routine Run Output Protection:** The routine evaluation runner [`scripts/run_real_pipeline_evaluation.py`](../../scripts/run_real_pipeline_evaluation.py) writes new evaluation runs to `evaluation_results/live_generation_worksheet.json` (git-ignored) to protect the frozen benchmark worksheet [`docs/evaluation/phase4_generation_worksheet.json`](phase4_generation_worksheet.json) from accidental overwriting.
