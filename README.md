# support-bot inference

Batch inference for our support assistant (Qwen2.5-0.5B-Instruct) on a RunPod GPU pod.

Every push ships the commit to the pod and runs the throughput gate (`bench.py`): 16 sampled
support prompts, 32 new tokens each. CI fails below 200 tok/s or if inference isn't on the GPU.
