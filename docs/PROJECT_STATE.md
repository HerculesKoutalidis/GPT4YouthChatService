# GPT4YOUTH — RAG Overhaul: State & Roadmap (handoff)

_Last updated: end of Step 3 (hybrid + query rewriting). Upload this (plus CLAUDE.md) to resume in a fresh session._

## What this project is
RAG chatbot helping EU youth workers design workshops / navigate Erasmus+.
Stack: Qwen3-14B-AWQ (vLLM) + Qdrant + Streamlit + two TEI services
(bge-m3 embeddings, bge-reranker-v2-m3) + BM25 sparse (FastEmbed, CPU), all on one shared A100.
The chatbot retrieves **raw material / inspiration** from ~14 (soon 200+) youth-work
PDFs and the LLM *generates* content (workshops, icebreakers, debriefs) from it.
"Correct" retrieval = surfaces usable, specific material, NOT a literal answer.
Primary use is CREATION of material/ideas, not "search-engine" lookup.

## Git
- Branch: `feature/rag-v2` (off `dev`). Tags: `v0.2` baseline, `v0.3` Step 2. Tag Step 3 when merged.
- Commit small & often; push to remote. `qdrant_storage/` is (correctly) gitignored.
- Judged eval results are tracked (`!eval/results/*_judged.jsonl`) — they cost ~€1/run.

## Services (docker-compose.yml)
- `vllm-engine`     Qwen/Qwen3-14B-AWQ, host:8005. GPU (~17GB at GPU_MEM_UTIL=0.30). Thinking mode OFF.
- `qdrant`          vector DB, host:6343 (HTTP) / 6344 (gRPC). No GPU.
- `tei-embeddings`  BAAI/bge-m3, host:8090 -> :80. GPU (~2-3GB). Serves /embed (dense).
- `tei-reranker`    BAAI/bge-reranker-v2-m3, host:8091 -> :80. GPU (~2GB). Serves /rerank.
- `streamlit-ui`    host:8505. No GPU. `RUNNING_IN_DOCKER=true` -> reaches others by name.
                    Runs FastEmbed BM25 (CPU) in-process for sparse query encoding.
Named volume `tei_data` caches the TEI model weights.

### Shared-GPU etiquette
```
docker compose stop vllm-engine tei-embeddings tei-reranker   # free GPU, keep everything
docker compose start vllm-engine tei-embeddings tei-reranker  # resume
```
Prefer `stop/start` over `down` (down tears down + forces a slow rebuild/re-pull on next up).
NOTE: on a shared box the NVIDIA driver may get upgraded under you -> `nvidia-smi` shows
"Driver/library version mismatch" and NO gpu container starts. Fix = reload nvidia kernel
modules or reboot (needs sudo/admin). Not a project problem; data is safe.

## Qdrant collections (current — Step 3, hybrid)
- `gpt4youth_docs`   — 6836 CHILD chunks (~180 tok). NAMED VECTORS:
                       "dense" = bge-m3 1024-dim Cosine; "sparse" = BM25 with IDF modifier.
- `gpt4youth_parents`— 1210 PARENT sections (~1000 tok), single unnamed dense vector.
                       Retrieved by id, fed to LLM. Never searched.
- `eu_job_market`    — OLD flat collection (384-dim). Delete when convenient.
Child payload: document_id, file_name, title, text, parent_id, section_path,
page_start/end, chunk_index, token_count, tags[], summary, country, language,
doc_type, who_it_helps, how_it_helps, source_link, year.

## config.yaml (current)
```yaml
llm:
  model_name: "Qwen/Qwen3-14B-AWQ"
  temperature: 0.7
  max_tokens: 4096
  max_history: 7
rag:
  collection_name: "gpt4youth_docs"
  parent_collection_name: "gpt4youth_parents"
  retrieve_top_k: 50           # dense prefetch candidates
  sparse_top_k: 50             # sparse (BM25) prefetch candidates
  hybrid_top_k: 50             # fused list size fed to the reranker
  use_sparse: true             # hybrid RRF fusion (false = dense-only, for A/B)
  use_reranker: true           # bge-reranker-v2-m3 re-orders candidates
  use_query_rewrite: true      # LLM rewrites last message into a standalone query
  rewrite_max_tokens: 80
  max_parents: 4
  max_parents_per_doc: 2
system_instructions: |         # includes the "Attribution Honesty" rule (Step 2.5)
  ...
```
- `.env`: HF_TOKEN, VLLM_API_KEY, GPU_MEM_UTIL=0.30, MAX_MODEL_LEN=16384.

## Retrieval flow (rag_engine.py, ChatEngine)
1. `_rewrite_query()` — if multi-turn & use_query_rewrite: LLM (Qwen3, temp 0, no-stream,
   thinking off) rewrites the last message into a standalone query. Falls back to the old
   concatenation if single-turn / empty / error. (Single-turn => no-op, so INVISIBLE to eval.)
2. dense embed (bge-m3/TEI) + sparse embed (BM25/FastEmbed, CPU).
3. Qdrant Query API on CHILD: prefetch dense(retrieve_top_k) + sparse(sparse_top_k),
   server-side FusionQuery(RRF) -> hybrid_top_k fused candidates. (use_sparse:false = dense-only.)
4. rerank (bge-reranker-v2-m3); falls back to fused order if reranker errors.
5. dedupe reranked children -> unique PARENTS (per-doc cap) -> max_parents=4.
6. fetch parents, attach deterministic Sources block -> vLLM (streaming, thinking off).

## How to run things
- Ingest (host .venv; needs qdrant + tei-embeddings up; fastembed installed):
  `python3 src/ingestion/build_catalog.py` then `python3 src/ingestion/ingest_pdfs.py --reset`
  (--reset MANDATORY when the collection schema / embedding changes.)
- Verify: `curl -s localhost:6343/collections/gpt4youth_docs | python3 -m json.tool | grep -A3 -E '"dense"|"sparse"'`
- Eval: `docker compose exec -w /app streamlit-ui python3 -m eval.run_pipeline --run-name <name> --sleep 0.3`
  then (host, ANTHROPIC_API_KEY) `python3 -m eval.judge --run <name>` and `python3 -m eval.report <base> <name>`.
- After editing rag_engine.py / tei_client.py / config: `docker compose restart streamlit-ui`.

## Roadmap & status
- [x] Step 0 — eval harness + baseline (`baseline_v0.2`)
- [x] Step 1 — parent/child ingestion + small-to-big + deterministic citations (`after_step1`)
- [x] Step 2 — bge-m3 embeddings + bge-reranker-v2-m3 via TEI (`step2_bge_rerank`, tag `v0.3`)
- [x] Step 2.5 — (a) prompt "Attribution Honesty" fix: groundedness +0.16 with source_attribution
      held (`step2_bge_rerank_prompt_tweak`); (b) LLM upgrade Llama-3.2-3B -> Qwen3-14B-AWQ:
      HUGE, broad win (`step2_qwen3`): specificity +0.57, actionability +0.53, all metrics up.
      This confirmed the 3-eval hypothesis that the GENERATOR was the ceiling, not retrieval.
- [x] Step 3 — hybrid dense+sparse (BM25/RRF, server-side) + LLM query rewriting.
      Hybrid: `step3_hybrid` vs `step2_qwen3` -> specificity +0.13, source_attribution +0.13
      (EXPERIENCED +0.23), net-positive. Small now because 14 PDFs rarely need rare-term recall;
      the real hybrid payoff arrives WITH scale-up. Rewriting validated qualitatively (multi-turn
      "Yes" -> correct topical sources); invisible to the single-turn judge, so not eval-measured.
- [ ] Step 4 — document-level routing (filter/prioritise WHICH PDFs via tags/summary/doc_type
      before chunk-search) + freshness field (prefer newer editions, e.g. ESC 2019).
      IMPORTANT: only worth doing AFTER scale-up — with 14 PDFs there's nothing to route.
- [ ] Step 5 — UI upgrade: left sidebar (st.sidebar) with SAVE/LOAD of past conversations.
      Needs on-disk persistence (JSON files or SQLite) because st.session_state is lost on refresh.
      Also polish: center the header image (done — nested columns), general modern-chat styling.
- [ ] Step 6 — Auth: Keycloak (lab already runs it) in front of the app, standard pattern is an
      oauth2-proxy container -> Keycloak. Keycloak can federate Google as an Identity Provider,
      so "login with Google" works via Keycloak (Google -> Keycloak -> chatbot).
- [ ] Step 7 — Scale up the knowledge base to 200+ PDFs. Keep filling Partners_data_assets.xlsx
      by hand (tags/summary etc.); Cazalla gives links (Cazala_sources_metadata.xlsx, 306 rows) but
      not the PDFs — sourcing those is a separate effort. DO THIS BEFORE Step 4 pays off.
- [ ] Later — DBSF fusion experiment (vs RRF); bge-m3 learned-sparse (vs BM25); LoRA for
      behavior/safety (NOT knowledge). All separate, individually-measured experiments.

## Eval cost / noise
- One judged run of 70 questions costs ~€1. Judge noise ~±0.1; a delta >= ~0.15 is real signal.
- Judged runs on disk: baseline_v0.2, after_step1, step2_bge_rerank, step2_bge_rerank_prompt_tweak,
  step2_qwen3, step3_hybrid. Current validated best = `step3_hybrid` (on Qwen3-14B).
- The eval harness is SINGLE-TURN, so multi-turn features (query rewriting) can't be judged by it;
  validate those qualitatively in the UI.

## Known issues / notes
- vLLM port 8005 exposed to the public internet (bots scanning). Firewall / SSH-tunnel when there's time.
- Disk on the VM is tight (was 100%). `huggingface-cli delete-cache`, `docker image prune`,
  `~/.cache/vllm/torch_compile_cache/*` are safe reclaims. Do NOT `docker system prune -a` (shared box).
- Qwen3 needs thinking mode OFF for content gen: extra_body={"chat_template_kwargs":{"enable_thinking":False}}
  is set in rag_engine's create() calls (both generation and rewrite).
- BM25 tokenization is language-dependent (best for EN); fine for ES/UK but less optimal. Invisible to EN-only eval.
- source_link only present for 3/14 docs; fill more for better citations.
- Ingestion upserts in batches of 256 (Qdrant 32MB/request cap).
- TEI --max-client-batch-size defaults to 32; tei_client batches embed & rerank at <=32.
- Judge must NOT be the served model; uses Claude via Anthropic API.
