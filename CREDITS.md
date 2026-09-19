# Credits and third-party information

This repository integrates software and model artifacts from several projects. The
MIT license in [`LICENSE`](LICENSE) applies to the original project material only.
Files identified below as derived works, and artifacts fetched at install time,
retain their own terms. Verify upstream terms before redistribution or commercial use.

| Component | Terms and use in this repository |
| --- | --- |
| [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | FP8 weights are referenced and fetched, not redistributed. The model card declared MIT when checked on 2026-09-04. |
| [inco.ai GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | Drafter is referenced and fetched, not redistributed. Its card declared CC BY-NC-ND 4.0 when checked on 2026-09-04: the shipped recipe is non-commercial and does not modify the drafter. |
| [vLLM](https://github.com/vllm-project/vllm) | Serving engine. `scripts/node/sparse_attn_indexer_kpool_sm121.py`, `scripts/node/patches/adaptive_k_scheduler.py` and the engine overrides under `scripts/node/overrides/vllm/` retain Apache-2.0 SPDX headers. The overrides include cache allocation, worker utilities and allocator diagnostics, the GLM model and pooled indexer, and hybrid KDA projections using native vLLM quantization and Marlin packing. The model and worker sources retain the vLLM contributors' copyright notice. The isolated scheduler in `scripts/node/experiments/e03/draft-budget/` retains the same provenance; exact R10 async-scheduler and output fixtures under `scripts/tests/fixtures/r10-draft-budget/` retain their Apache-2.0 and vLLM contributor notices. |
| [JSpark3](https://github.com/jakejharris/jspark3/tree/a68bd902768ec9a1e681368f3f0a0807cb9278c0) | The Apache-2.0 `recipe/overlays/trunk_w8a16.py` post-load integration pattern informed `e20_kda_w8a16.py`; the local implementation selects 34 TP4 KDA input projections and uses the installed vLLM APIs. The companion `e20_hybrid_scratch.py` reconstructs those same quantized weights for BF16 prefill. The upstream [LICENSE](third_party/jspark3/LICENSE), [NOTICE](third_party/jspark3/NOTICE), and [required attribution](third_party/jspark3/REQUIRED_ATTRIBUTION.md), including its ShapleyMcg notice, are preserved verbatim from revision `a68bd902768ec9a1e681368f3f0a0807cb9278c0`. |
| SparkRing / SparkCache (`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache`) | Provider of the SM121 vLLM container image in the September 18 and September 19 configurations with the SparkCache prefix-cache connector. The image is pulled by registry digest, not redistributed, and keeps its upstream component terms. The connector, page codec, and SIRCL transport bundle and runtime are operator-supplied payload; no license file or notice accompanied the archived sources. This repository tracks their SHA-256 manifests and expects the operator to place the files on the nodes, including the selected cache-memory corrections and replay views. Ask the authors before redistributing them. |
| [NVIDIA NCCL](https://github.com/NVIDIA/nccl) | The vendored patch modifies NCCL v2.30.7-1 sources; resulting binaries retain NCCL's BSD-3-Clause terms. No NCCL source tree or binary is redistributed here. |
| [SparkRing mHC source package](https://github.com/FujitsuPolycom/sparkring/tree/61f277bd0c97fbff892668e12ea04a330a45fa01/runtime/glm53-spark-mtp3-mesh/performance/mhc-prefill) | Apache-2.0 package ported narrowly onto R10 for the accepted FP8 E03 recipe. The port adapts TP4/DCP4 admission to TP4/DCP1 and preserves local FP8/E20 and KDA changes. Source hashes and the delta are under `scripts/node/experiments/e03/`; [LICENSE](third_party/sparkring-mhc/LICENSE) and [NOTICE](third_party/sparkring-mhc/NOTICE) are preserved. The CPU tests derive from the same package. This license applies to the mHC package, not to the separate operator-supplied SparkCache/SIRCL payloads. |
| [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) | Source of the vendored switchless overlay. No upstream license file or repository license metadata was found when checked on 2026-09-04. Attribution does not resolve that uncertainty; ask the author before redistributing a derivative. |
| [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | Provider of the SM121 vLLM container image and DFlash2 patch chain in the September 11 configuration, kept as the historical rollback. The image is pulled, not redistributed, and keeps its upstream component terms. |

The four-node FP8 lane descends from
[Wpnx330/GLM-5.3-Flash-FP8-4x-DGX-Spark](https://github.com/Wpnx330/GLM-5.3-Flash-FP8-4x-DGX-Spark).
The switchless architecture and presentation were also informed by
[Alex Ellis's four-node recipe](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless).
[jspark3](https://github.com/jakejharris/jspark3) informed the retained functional
acceptance instrumentation.

CUDA, cuDNN, NVIDIA drivers and tools, PyTorch, Triton, Docker, and other host or
container dependencies are invoked rather than redistributed and retain their own
licenses. NVIDIA, GB10, DGX Spark, ConnectX and Nsight are NVIDIA trademarks; ASUS
and Ascent GX10 are ASUSTeK trademarks. This personal project is unsupported and is
not affiliated with or endorsed by those companies, Z.ai, inco.ai, or vLLM.
