"""Throughput gate: sample prompts, generate, report tok/s, fail under the bar.

CI runs this on the GPU pod. Thresholds come from the workflow env, not here.
"""

import json
import os
import random
import sys
import time

import torch

from serve import Engine

MIN_TOKS = float(os.environ.get("MIN_TOKENS_PER_SEC", "200"))
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") == "1"
BATCH = int(os.environ.get("BENCH_BATCH", "16"))
NEW_TOKENS = int(os.environ.get("BENCH_NEW_TOKENS", "32"))


def main() -> int:
    prompts = [json.loads(line)["prompt"] for line in open("prompts.jsonl", encoding="utf-8")]
    random.Random(int(os.environ.get("SEED", "7"))).shuffle(prompts)
    batch = prompts[:BATCH]

    engine = Engine()
    gpu = torch.cuda.get_device_name(0) if engine.device == "cuda" else "none"
    print(f"device={engine.device} gpu={gpu} torch={torch.__version__} "
          f"cuda_build={torch.version.cuda} dtype={engine.model.dtype}", flush=True)

    engine.generate(batch[:1], max_new_tokens=4)  # warmup
    if engine.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = engine.generate(batch, max_new_tokens=NEW_TOKENS)
    if engine.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    pad = engine.tok.pad_token_id
    tokens = int(sum(int((o != pad).sum()) for o in outputs))
    toks = tokens / elapsed
    cfg = engine.model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * engine.model.dtype.itemsize
    result = {
        "device": engine.device,
        "gpu": gpu,
        "torch": torch.__version__,
        "dtype": str(engine.model.dtype),
        "prompts": len(batch),
        "generated_tokens": tokens,
        "seconds": round(elapsed, 2),
        "tokens_per_sec": round(toks, 1),
        "avg_latency_per_prompt_s": round(elapsed / len(batch), 2),
        "kv_cache_kib_per_token": round(kv_bytes_per_token / 1024, 1),
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20) if engine.device == "cuda" else 0,
    }
    print(json.dumps(result, indent=2))
    with open("bench.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n--- sampled completions ---")
    for prompt, out in list(zip(batch, outputs))[:3]:
        print(f"> {prompt}\n  {engine.decode(out)[:200]}\n")

    failures = []
    if REQUIRE_GPU and engine.device != "cuda":
        failures.append(f"inference ran on {engine.device}, GPU required")
    if toks < MIN_TOKS:
        failures.append(f"throughput {toks:.1f} tok/s < required {MIN_TOKS:.0f} tok/s")
    if failures:
        print("THROUGHPUT GATE: FAIL - " + "; ".join(failures))
        return 1
    print(f"THROUGHPUT GATE: PASS - {toks:.1f} tok/s on {gpu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
