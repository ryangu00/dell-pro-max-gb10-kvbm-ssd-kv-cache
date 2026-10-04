![banner](docs/assets/banner.png)

# KV cache on NVMe with NVIDIA Dynamo KVBM — one Dell Pro Max with GB10

> A minimal, measured test of three-tier KV-cache offload (GPU → host → SSD) on a single Dell Pro Max with GB10 (128 GB unified memory, sm_121, aarch64), using vLLM 0.27.1 and the standalone `kvbm` 1.4.2 wheel. The point of this cookbook is the **evidence**, not a recommendation: what works, what silently does not, and how to tell the difference in five minutes. Every number below comes from a JSON file in `results/`, and every log line from `results/logs/` (the dated Update (2026-10) section at the end is the exception; it says where each of its numbers comes from).

## TL;DR

| Tier path | On this machine | Evidence |
|---|---|---|
| GPU → host (G2) offload, host → GPU onboard | **Works, and the answer is correct.** | Same 30K-token document after eviction: TTFT 9.03 s → 0.25 s, answer identical and correct, `kvbm_onboard_blocks_h2d` +971 in that step (`results/probe-kvbm-hostonly-8b-e1.json`) |
| GPU → host → SSD (G3) offload | Writes are counted (`kvbm_offload_blocks_h2d` +7536 per run) | — |
| SSD → GPU onboard | **Wrong KV, no error to the client.** | KVBM counts a full hit (`kvbm_onboard_blocks_d2d` +1884 = 30,156 tokens / 16), TTFT drops to 0.22 s, and the model answers with an earlier request's content (the last evicting document in the original runs; the first evicting document in the 2026-09-19 re-run). Three variants, same result. The failing step is identified below; why cuFile registration fails is NOT established (see Update (2026-10)). |
| Hybrid GDN/Mamba models (Qwen3.5/3.8 family) | **Engine does not start.** | `ValueError: Failed to promote local KV cache specs to one unified type` — `DynamoConnector` does not implement `SupportsHMA` (`results/logs/27b-attempt1.log`) |

Two conclusions worth the whole exercise:

1. **On a unified-memory box the host tier buys no capacity.** The "host" 8 GB comes out of the same 128 GB the GPU KV pool would have used; it is a different eviction policy, not more memory. The only tier that could add capacity is the SSD — and that is the tier that does not work here.
2. **A fast TTFT after a cache hit proves nothing.** Assert the answer. Without the answer check, this probe would have reported a 40× speed-up as a success.

Scope: one machine, one image, one KVBM version, one dense 8B model. Nothing here says how KVBM behaves on hardware that has GPUDirect Storage. A later re-run with the `nvidia_fs` kernel module loaded on this same machine did not change the result (see Update (2026-10)), so loading the module is not sufficient to make the disk tier work on this platform.

## Hardware and software

| | |
|---|---|
| Machine | Dell Pro Max with GB10 (GB10, 128 GB unified LPDDR5X, sm_121, aarch64); disk cache on the local NVMe (ext4) |
| Driver / CUDA | 580.142 / CUDA 13.0 on kernel 6.17.0-1014-nvidia for the original runs; no `nvidia-fs` kernel module was loaded in those runs. This is a condition of the runs, not the established cause of the failure (see Update (2026-10)). |
| Container | `vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404` (vLLM 0.27.1, torch 2.13 cu130, NIXL 1.3.1) |
| KVBM | `kvbm==1.4.2` from PyPI (`manylinux_2_28_aarch64` wheel), installed with `--no-deps` |
| Models | `Qwen/Qwen3-8B` (dense, standard attention — the clean case); `Qwen3.8-27B-NVFP4` (GDN hybrid — the failure case) |

## Setup

Model weights live in `$HOME/models/<name>` on the host and are mounted at `/models` in the container. `scripts/kvbm-launch.sh` writes the serve command to a script and mounts it (the `--kv-transfer-config` JSON does not survive `bash -c "..."` quoting), then starts the container with the KVBM environment:

```bash
# inside the container the launcher runs:
pip install --no-deps kvbm==1.4.2         # --no-deps: kvbm pins nixl==1.0.1, which shadows the image's NIXL 1.3.1 and breaks vllm's nixl_ep import
vllm serve /models/Qwen3-8B --served-model-name qwen3-8b --port 8000 \
  --max-model-len 40960 --gpu-memory-utilization 0.20 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.vllm_integration.connector"}'
```

KVBM environment set by the launcher (worker container):

```
DYN_KVBM_CPU_CACHE_GB=<KVBM_CPU_GB, default 8>       # must be >= the engine's GPU KV pool, or KVBM churns (its own docs)
DYN_KVBM_DISK_CACHE_GB=<KVBM_DISK_GB, default 64>    # KVBM_DISK_GB=0 -> no disk tier
DYN_KVBM_DISK_CACHE_DIR=/kvbm-disk
DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true            # test only; by default a block must be hit twice before it is written to disk (SSD wear)
DYN_KVBM_METRICS=true  DYN_KVBM_METRICS_PORT=6880
```

The exact launcher invocation for each run in the results table:

```bash
# no KVBM (control)
./scripts/kvbm-launch.sh vllm-kvbm-8b /models/Qwen3-8B qwen3-8b 0.20 40960 0
# host tier only
KVBM_CPU_GB=8 KVBM_DISK_GB=0  ./scripts/kvbm-launch.sh vllm-kvbm-8b /models/Qwen3-8B qwen3-8b 0.20 40960 1
# host + disk (attempt 1)
KVBM_CPU_GB=8 KVBM_DISK_GB=64 ./scripts/kvbm-launch.sh vllm-kvbm-8b /models/Qwen3-8B qwen3-8b 0.20 40960 1
# host + disk, POSIX backend toggles (attempt 2)
KVBM_CPU_GB=8 KVBM_DISK_GB=64 KVBM_POSIX=true KVBM_GDS=false KVBM_GDS_MT=false ./scripts/kvbm-launch.sh vllm-kvbm-8b /models/Qwen3-8B qwen3-8b 0.20 40960 1
# host + disk, cuFile compat mode (attempt 3); scripts/cufile.compat.json copied to $HOME/kvbm-logs/cufile.json (mounted at /kvlogs)
KVBM_CPU_GB=8 KVBM_DISK_GB=64 KVBM_EXTRA_DOCKER_ARGS="-e CUFILE_ENV_PATH_JSON=/kvlogs/cufile.json -e CUFILE_FORCE_COMPAT_MODE=true" ./scripts/kvbm-launch.sh vllm-kvbm-8b /models/Qwen3-8B qwen3-8b 0.20 40960 1
```

Wait for `Application startup complete` in `docker logs -f vllm-kvbm-8b`. With `--gpu-memory-utilization 0.20` the engine reports `GPU KV cache size: 44,832 tokens` (45,088 without the connector) — one 30K-token document fits, four evict it.

## The probe

`scripts/kvbm_probe.py` sends the same ~30K-token document (A) twice, with N distinct 30K-token documents (E1..EN) in between to evict A from the GPU pool (N ≥ 2 also evicts it from the 8 GB host tier), and records per request:

- TTFT (monotonic time to the first streamed content token; temperature 0; thinking disabled)
- the answer to "what is the first code word in this document?" — the probe generated the document, so it knows the word; the answer's first alphabetic token must equal it
- the `kvbm_*` counters after the request and the delta **for that request alone**, so a hit is attributed to a tier rather than assumed
- stream completeness and `finish_reason`; exit code 0 = A2 correct, 2 = A2 wrong, 3 = a request failed

```bash
python3 scripts/kvbm_probe.py http://127.0.0.1:8000/v1 qwen3-8b --tokens 30000 --evict 4 --metrics http://127.0.0.1:6880/metrics --out results/my-run.json
```

The probe calibrates tokens-per-word with one request first (this vocabulary tokenizes at ~5.6 tokens/word). Keep `--tokens` under the engine's `--max-model-len`.

## Results

Qwen3-8B, 30,156-token document, `--evict 4` unless noted. All files in `results/`.

| Run | Tiers | Evict | A1 TTFT | A2 TTFT | A2 answer | KVBM counters over the run | File |
|---|---|---|---|---|---|---|---|
| no KVBM (control) | GPU only | 4 | 8.98 s | 9.13 s | `yankee` (correct) | — | `probe-kvbm-off-8b.json` ¹ |
| KVBM host only | GPU+host | 1 | 9.03 s | **0.25 s** | `yankee` (**correct**, identical) | onboard h2d +971 | `probe-kvbm-hostonly-8b-e1.json` |
| KVBM host only | GPU+host | 4 | 0.17 s ² | 9.19 s | `yankee` (correct) | host evicted → recompute | `probe-kvbm-hostonly-8b-e4.json` |
| KVBM host+disk, attempt 1 | GPU+host+SSD | 4 | 9.01 s | **0.22 s** | **`delta` — wrong** (E4's answer) | onboard d2d +1884 | `probe-kvbm-on-8b-nothink.json` |
| attempt 2: `DYN_KVBM_NIXL_BACKEND_POSIX=true`, GDS/GDS_MT off | same | 4 | 9.04 s | 0.23 s | `delta` — wrong | onboard d2d +1884 | `probe-kvbm-on-8b-posix.json` |
| attempt 3: cuFile compat mode | same | 4 | 9.08 s | 0.23 s | `delta` — wrong | onboard d2d +1884 | `probe-kvbm-on-8b-cufile-compat.json` |

¹ The control file was written by an earlier revision of the probe whose `expected` field still carried the numeric suffix (`yankee7028`); its `correct_*` flags are therefore `false` even though both answers are `yankee`. Judge it by `text_A1` / `text_A2`. The other files were written by the revision that stores the word stem. The current `scripts/kvbm_probe.py` additionally records per-request counter deltas, `finish_reason` and stream completeness; the committed files predate those fields.
² A was still resident in the host tier from the preceding run (same server), so A1 itself was a host hit; the interesting number in that row is A2 after four evictions.

## Why the SSD tier returns the wrong KV

**Verified from the logs** (`results/logs/8b-kvbm-on-attempt1.log`, identical in attempts 2 and 3):

```
gds_mt_backend.cpp:237] GDS_MT: failed to create file handle: GDS_MT: file register error: error=5027, fd=149    <- at startup, right after "DiskStorage created"
nixl_agent.cpp:921] createXferReq: no specified or potential backend had the required registrations to be able to do the transfer   <- at the moment of the A2 "hit"
dynamo_llm::block_manager::distributed::transfer: Failed to write to blocks: Other(Failed to create XferRequest)
dynamo_runtime::utils::tasks::critical: Critical task 'ZmqActiveMessageWorker: Handler for function: transfer_blocks' failed: Failed to create XferRequest
KVBM [worker-…] Cache Hit Rates - Host: 0.0% (0/11998), Disk: 15.7% (1884/11998)                                     <- KVBM still books the hit
```

The client got a 200 with the wrong answer; a later request then sat at `Running: 0 reqs, Waiting: 1 reqs, Deferred: 1 reqs` for minutes while `/health` kept returning 200.

**Inferred from the kvbm 1.4.2 source** (not verified by patching — treat as the most likely mechanism):

1. cuFile cannot register the cache file (`GDS_MT: file register error: error=5027`, verified in the logs). We first attributed this to the missing `nvidia-fs` module, but a re-run on 2026-09-19 with `nvidia_fs` 2.29.4 loaded and the `/dev/nvidia-fs*` devices passed into the container still logged the same error=5027 and another incorrect answer. The reason registration fails on this platform is unresolved.
2. `build_agent(worker_id, need_disk)` in `lib/llm/src/block_manager/distributed/worker.rs` (tag v1.4.2) creates the GDS_MT backend whenever a disk tier is configured — the parameter is named `use_gds` but the caller passes `need_disk` — and the transfer planner then takes the direct disk→device path. That is consistent with attempts 2 and 3 changing nothing: the `DYN_KVBM_NIXL_BACKEND_*` toggles and cuFile compat mode do not enter this path.
3. The failed transfer is logged but not surfaced to the connector, so the blocks are handed to vLLM as loaded and the physical GPU blocks still hold some earlier request's KV — in the original runs the last evicting document, in the 2026-09-19 re-run the first evicting document — which is what the model answered from. An upstream maintainer's triage of our issue (2026-09-22) reached the same mechanism independently, according to our notes: any failed transfer is marked complete and counted as loaded, and this is not specific to GDS. (Verify against the live thread before quoting further.)

The same `createXferReq` error appears in ai-dynamo/dynamo issues #5012 (closed as stale) and #5857.

## Status of KVBM upstream (historical record as of 2026-09-18)

- On 2026-09-10 a maintainer closed issue #12750 with "KVBM is no longer supported", and issue #13867 disputes that deprecation wording. These events were recorded as of 2026-09-18; the live thread should be checked for any later change.
- On 2026-09-22 a maintainer closed our upstream issue (ai-dynamo/dynamo #15079, filed with the reproduction in this repository) stating that KVBM has been sunsetted in favor of KVCR and will not be fixed. (Paraphrased from our notes; not re-fetched for this update.)
- Hybrid-model support in KVBM is moot after the sunset; whether KVCR supports hybrid (GDN/Mamba) models is not verified (its official examples we read disable the hybrid KV cache manager).

## Pitfalls, in the order we hit them

1. `pip install kvbm` pulls `nixl==1.0.1`, which shadows the image's NIXL 1.3.1 and breaks `vllm.model_executor…nixl_ep` (`module 'nixl_ep' has no attribute 'Buffer'`). Use `--no-deps`.
2. The `--kv-transfer-config` JSON is eaten by `bash -c "..."` quoting; write the serve command to a file and mount it.
3. Qwen3.8-27B is **not** a plain dense model (`Qwen3_5ForConditionalGeneration`, GDN hybrid). It cannot be the control for a connector test.
4. Prompt sizing: this vocabulary tokenizes at ~5.6 tokens/word. Calibrate with one request instead of guessing, and keep prompts under `--max-model-len`.
5. A cache "hit" must be checked three ways: TTFT, per-request tier counters, **and the answer**.
6. The disk cache file is created with `fallocate` and immediately unlinked (deleted when the fd closes) — `ls` will not show it; look for `DiskStorage created` in the log.
7. `/health` returning 200 does not mean the engine is serving; probe with a real request.

## Files

- `scripts/kvbm-launch.sh` — container launcher (tier sizes, backend toggles, extra docker args)
- `scripts/kvbm_probe.py` — the probe (TTFT + answer check + per-request counter deltas; exit code is the verdict)
- `scripts/cufile.compat.json` — cuFile compat-mode config used in attempt 3 (no effect here)
- `results/*.json` — raw probe output per run; `results/logs/*.log` — engine + KVBM logs per run (hostnames and LAN addresses replaced)

## License

Apache-2.0. Author: ryangu00 (RyanAI Lab).

## Update (2026-10): upstream sunset, a re-test that contradicts our cause, and why we did not try the successor

### A. Upstream status

- On 2026-09-22 a maintainer closed our upstream issue (ai-dynamo/dynamo #15079) stating that KVBM has been sunsetted in favor of KVCR and would not be fixed. (Paraphrased from our notes; not re-fetched for this update.)
- In the same thread a second maintainer's triage said: (a) the 5027 / transfer-creation failure is an older, unfixed issue (ai-dynamo/dynamo #6032); (b) silently returning wrong KV is a correctness bug, worse than a hang; (c) the corrupting path is not limited to GDS — any failed transfer is marked complete and counted as loaded. (Paraphrased from our notes.)
- KVCR: repository `github.com/ai-dynamo/kvcr`; version 0.1.0 on PyPI (published 2026-09-19 according to our notes); it plugs into vLLM as a secondary tier of vLLM's tiered offloading spec (vLLM PR #53624, merged to main on 2026-09-14). Read in source and docs by us on 2026-09-22; nothing was executed.
- What the source read shows (not run): the file-tier backend is configurable (default `GDS_MT`); if the backend plugin is missing or registration fails, startup raises an error (fail loud); a failed transfer is reported back to vLLM as unsuccessful, so vLLM recomputes; the memory side of file transfers is host DRAM, so data moves in two hops instead of a direct GPU path. These are exactly the two behaviours the original test found missing.

### B. Zero-GPU check that the successor resolves (observed once, 2026-09-22)

Container `vllm/vllm-openai:v0.30.0-aarch64` (vLLM 0.30.0; the tag appeared on 2026-09-22), inspected without a GPU attached.

- The secondary-tier registry contains `kvcr`; `TransferJob.chunk_ids` is present; `OffloadPolicy.CHUNK_LEVEL` is present.
- NIXL 1.4.1 is installed (CUDA 12 and 13 wheels) with plugins POSIX, GDS, GDS_MT, UCX and OBJ.
- After `pip install --no-deps kvcr==0.1.0` the `kvcr` tier class resolves.

Caveats: KVCR pins `nixl==1.3.2` while the image carries 1.4.1, so the combination is untested. An older vLLM main-branch build from 2026-09-13 lacked `chunk_ids`, so KVCR could not be overlaid on it. "Resolves" means imports and registry entries, not that a transfer works.

KVCR was not deployed. No KVCR request was ever served on our hardware; there are no KVCR latency, hit-rate or correctness numbers. We wrote launch and probe scripts for a test, did not run them, and do not publish them.

### C. Re-test of KVBM with the kernel module loaded (measured once, 2026-09-19)

Conditions: same launcher, same probe and same model as the original runs (Qwen3-8B, host tier 8 GB, disk tier 64 GB, 30,156-token document, 4 evicting documents, thinking off). The machine had since been updated to kernel 7.0.0-1019 and driver 580.178.04; `nvidia_fs` 2.29.4 loaded; `/dev/nvidia-fs*` device nodes passed into the container. The image tag and kvbm version were not separately recorded in the result files; the same launcher was used. The first attempt at GPU memory utilisation 0.20 did not start (available KV cache 4.26 GiB, below the 5.62 GiB needed); the run reported here used 0.28 (engine up after about 240 s).

Result (one run): startup log still contains `GDS_MT: file register error: error=5027`. A1 TTFT 9.376 s, answer correct (`yankee`). A2 TTFT 0.22 s, answer wrong (`oscar`), 1,884 blocks counted as onboarded from disk, three `Failed to create XferRequest` errors in the log. The wrong answer equals the answer of the first evicting document, not the last (in the original runs it equalled the last, `delta`).

Interpretation: loading `nvidia_fs` and passing the devices through did not change the failure. We do not know why registration fails; the earlier explanation (missing module) is withdrawn. n = 1 run.

### D. Why we did not try the successor: production KV metrics and the arithmetic (judgement, with projections marked)

Source of the metrics: the serving engine's Prometheus counters on the production endpoint. Model and topology: Qwen3.8-Flash-Next served by vLLM tensor-parallel across two Dell Pro Max with GB10 machines (RoCE link), fp8 KV cache, prefix caching on, 1,000,000-token maximum context.

Window: 47 hours starting 2026-09-20 at 13:38 local time (engine start); 856 requests. Observed (summary figures recorded at the time; the raw counter dump was not archived):

| Metric | Value |
|---|---|
| KV pool | 25.19 GiB per node = 3,435,666 tokens (about 7.69 KiB per token per node) |
| Unique prompt tokens actually computed (prompt tokens by source, local compute) | 1,956,722 (below the pool size) |
| Prefix-cache hits | 5.50M of 7.46M queried prompt tokens (73.8 %) |
| Prompt length | mean 8.7K tokens, p93 at most 50K, maximum at most 200K |
| Total prefill time over the window | 889 s (mean 1.04 s per request) |

Judgement (derived, not directly measured): because the unique computed prompt tokens were smaller than the pool, we expect no prefix-cache eviction in that window, so a disk tier would have been read zero times. We did not read an eviction counter, and the arithmetic ignores output tokens, which also occupy KV.

Offloading KV to disk does not save GPU memory: KV of running requests must stay in GPU memory; offload only lengthens the lifetime of cached prefixes. Memory is saved by shrinking the pool.

Projections (arithmetic, never executed): shrinking the per-node KV pool to hold 550K tokens would release about 21.2 GiB per node; holding 1M tokens would release about 17.9 GiB. With such a pool, the benefit ceiling of a disk tier would be re-prefilling the roughly 5.5M cache-hit tokens of the window, about 30–60 minutes of prefill per 47 hours (cross-check: 1.96M tokens in 889 s is about 2.2K tokens/s, giving about 42 minutes). We did not run a shrunk-pool experiment.

Reusable check before evaluating any KV offload tier: compare unique computed prompt tokens (by source) against the KV pool size, and look at prefix-cache hits/queries. If the former is below the pool, a lower tier has nothing to catch — this is only an inference for that window; it does not confirm zero evictions or zero disk reads.

### What is still unresolved or unverified

- Why cuFile registration fails (error=5027) on this platform with the module loaded: unknown. We did not decode the error code from a primary record.
- KVCR on this hardware: untested. Nothing is known about its correctness, hit path or speed here; the zero-GPU check only shows that the component resolves in one image. The nixl 1.4.1 versus pinned 1.3.2 combination is untested. Hybrid (GDN/Mamba) model support in KVCR is unverified.
- The production-metrics judgement is for one 47-hour window and one deployment. A later cumulative read of the same engine's counters (2026-09-24, about 3.5 days including evaluation and other test traffic) showed 14,757 requests and about 10.9M computed prompt tokens (35.9M queried minus 25.0M hits), which is larger than the 3.4M-token pool. We did not analyse whether evictions occurred in that traffic. So "zero evictions" must be read as a statement about the first 47 hours, not about our load in general, and the review condition that was set (evictions appear, or hit rate falls after shrinking the pool) has not been checked against the later data.
- Upstream links were re-checked only against our own dated records, without browsing. #15079 closure and the maintainers' statements are paraphrased from our notes; #12750, #13867, #5012, #5857 and #8051 have no record newer than 2026-09-18 in our files, so their wording is kept dated, or should be verified live before publishing.
- Re-test is a single run on one machine.

