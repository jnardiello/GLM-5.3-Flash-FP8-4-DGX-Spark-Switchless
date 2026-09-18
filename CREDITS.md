# Credits and third-party information

This repository integrates software and model artifacts from several projects. The
MIT license in [`LICENSE`](LICENSE) applies to the original project material only.
Files identified below as derived works, and artifacts fetched at install time,
retain their own terms. Verify upstream terms before redistribution or commercial use.

| Component | Terms and use in this repository |
| --- | --- |
| [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | FP8 weights are referenced and fetched, not redistributed. The model card declared MIT when checked on 2026-09-04. |
| [inco.ai GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | Drafter is referenced and fetched, not redistributed. Its card declared CC BY-NC-ND 4.0 when checked on 2026-09-04: the shipped recipe is non-commercial and does not modify the drafter. |
| [vLLM](https://github.com/vllm-project/vllm) | Serving engine. `scripts/node/sparse_attn_indexer_kpool_sm121.py`, `scripts/node/patches/adaptive_k_scheduler.py` and the six override modules under `scripts/node/overrides/` (KV-cache interface and utilities, worker utilities, GLM-5.3 pooled indexer and kpool ops, bind-mounted over the R10 image) are Apache-2.0-derived files with SPDX and provenance headers. |
| SparkRing / SparkCache (`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache`) | Provider of the F1 SM121 vLLM container image with the SparkCache prefix-cache connector. The image is pulled by registry digest, not redistributed, and keeps its upstream component terms. The connector module and the SIRCL transport bundle and runtime used with it carried no license file or notice when checked on 2026-09-18; this repository tracks only their SHA-256 manifests and expects the operator to place the files on the nodes. Ask the authors before redistributing them. |
| [NVIDIA NCCL](https://github.com/NVIDIA/nccl) | The vendored patch modifies NCCL v2.30.7-1 sources; resulting binaries retain NCCL's BSD-3-Clause terms. No NCCL source tree or binary is redistributed here. |
| [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) | Source of the vendored switchless overlay. No upstream license file or repository license metadata was found when checked on 2026-09-04. Attribution does not resolve that uncertainty; ask the author before redistributing a derivative. |
| [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | Provider of the F0-lane SM121 vLLM container image and DFlash2 patch chain, kept as the documented rollback. The image is pulled, not redistributed, and keeps its upstream component terms. |

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
