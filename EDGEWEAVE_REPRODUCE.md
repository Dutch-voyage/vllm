# EdgeWeave: vLLM experiment integration

This branch contains the exact serving implementation used by **EdgeWeave: Redistributing Communication Precision for MoE Inference**, plus reproduction documentation. Start with the [PACE reproduction guide](https://github.com/Dutch-voyage/PACE/blob/main/reproduction/README.md), which includes 105 recorded commands, source-pair selection, frozen request/routing inputs, expected results, and offline table generation.

| Result | vLLM measured commit/tag | PACE measured commit |
| --- | --- | --- |
| Current serving tables, Qwen EP4 overlap, GPT/Qwen EP4/8 efficiency | `6b6a7d920acac843ebd1e5048a720e318cffc766` / `edgeweave-serving-20260906` | `27e37921d5a393abe217d61800c6a662d2f7efa0` |
| New Qwen benchmark KL and frozen quality; S1/S2/S5 probes | `e81b41c314281622b768901fac4180c7e693ae2b` / `edgeweave-quality-20260906` | `c50d758a63409668f26837a56fe7a9ceffe7ca2f` (S5: `665e865a680443a4b6d3713b63b85c73db2ba727`) |
| Historical S6 lane gate/S7 DBO | serving commit above | `973afd91a65c558644eb312273cf14eb9c5cabae` |

The serving revision adds deferred exact NCCL lane exchanges and preserves the sentinel row needed by the codec. Quality and serving share the BF16, multi-rung and CE integration lineage. Preserve source pairs rather than mixing revisions. The default `main` branch tracks a different upstream state and is not the paper runtime. The old `ce-a2a-codec` branch is historical and insufficient for these results.

## Build and qualification

Use Linux x86-64, CUDA-capable PyTorch, the CUDA toolkit and four/eight peer-accessible GPUs. Recorded runtime: Python 3.12, Torch 2.11.0+cu130, driver 595.71.05, RTX 4090, two NUMA nodes. Create the environment outside source worktrees so they remain clean:

```bash
uv venv --python 3.12 /absolute/path/to/runtime/.venv
uv pip install --python /absolute/path/to/runtime/.venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /absolute/path/to/runtime/.venv/bin/python -e /absolute/path/to/pinned-vllm --torch-backend=cu130
```

This invokes the source installation path. Build requirements and CUDA extension selection are defined by this revision's `pyproject.toml`, `setup.py`, and `requirements/`. Build-generated Python modules, third-party interfaces and shared libraries must be present when importing each worktree. Put the selected vLLM and matching PACE source on `PYTHONPATH`, as the PACE runner does. Do not overwrite the pinned adapter with PACE's older reference copy or apply legacy patch series to this branch.

**Recorded binary provenance:** measured runs reused compatible precompiled artifacts from an existing validated installation (`vllm-0.1.dev1+g97a98006b.precompiled`), not a newly rebuilt wheel. PACE's `reproduction/runtime/REPAIR.md`, `binaries.sha256`, and `generated-python.sha256` retain exact provenance. Fresh source-build byte identity has not been established; qualify the resulting runtime before treating measurements as comparable. Shared libraries are not included in Git.

Use `uv pip` to install test dependencies from the revision's requirement files. With the selected PACE source on `PYTHONPATH`, run:

```bash
/absolute/path/to/runtime/.venv/bin/python -m pytest tests/distributed/test_ce_a2a.py tests/v1/worker/test_gpu_ubatch_wrapper.py -q
```

Then run PACE's direct-wire, eight-layer delta-wire and four-rank exact-NCCL lane proof (the S6 commands in the reproduction index). The lane proof checks separate slots, exact received bytes, deferred dependencies and safe drain/close. Complete model startup and workload warmups before timings. Tests and runtime paths must correspond to the selected source pair.

## Protocol boundaries

- Main Dynamic 4–8 means direct; delta stays separate.
- BF16 compute is mandatory; the full-width reference uses FP16 wire.
- Qwen uses codec group128/period48. GPT serving uses group64/period24; frozen GPT quality delta used period48 and is not a matched serving/quality delta comparison.
- Current off versus DBO changes chunk/lane admission; only serialized versus overlapped DBO isolates overlap.
- Quality wrappers align external-DP request groups and preserve constrained GPT decoded MCQ versus unmasked full-vocabulary KL.

This publication adds guidance only to the measured vLLM source; it does not change inference behavior or claim another GPU rerun. Follow the PACE guide for result-by-result commands and acceptance checks.
