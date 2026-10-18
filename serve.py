"""support-bot inference engine: loads the model once, answers a batch of prompts."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


class Engine:
    def __init__(self, model_id: str = MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.to(self.device).eval()

    def _encode(self, prompt: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int = 64) -> list[torch.Tensor]:
        """Returns the generated token ids for each prompt (prompt stripped)."""
        outputs = []
        for prompt in prompts:
            ids = self.tok(self._encode(prompt), return_tensors="pt").to(self.device)
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
            )
            outputs.append(out[0, ids["input_ids"].shape[1]:])
        return outputs

    def decode(self, ids: torch.Tensor) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()
