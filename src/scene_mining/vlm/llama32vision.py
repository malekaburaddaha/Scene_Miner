"""LLaMA-3.2-11B-Vision VLM backend.

Same interface as Qwen25VL — injects Q-Former visual tokens via
inputs_embeds prepend.
"""

from __future__ import annotations

import torch

from .base_vlm import BaseVLM

_LLAMA_HF_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"


class LLaMA32Vision(BaseVLM):
    """LLaMA-3.2-11B-Vision-Instruct VLM backend.

    Args:
        model_name: HuggingFace model ID.
        load_in_4bit: Enable 4-bit NF4 quantisation.
        lora_r: LoRA rank (0 = no LoRA).
    """

    def __init__(
        self,
        model_name: str = _LLAMA_HF_ID,
        load_in_4bit: bool = True,
        lora_r: int = 0,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_cfg = None
        if load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._d_model = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad_(False)

        if lora_r > 0:
            self._apply_lora(
                lora_r, lora_alpha,
                lora_target_modules or ["q_proj", "v_proj"],
            )

    def _apply_lora(self, r: int, alpha: int, target_modules: list[str]) -> None:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=r, lora_alpha=alpha, target_modules=target_modules,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, cfg)

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # LLaMA embed layer
        text_embeds = self.model.model.embed_tokens(input_ids)
        visual_tokens = visual_tokens.to(text_embeds.dtype)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        if attention_mask is None:
            B, L = inputs_embeds.shape[:2]
            attention_mask = torch.ones(B, L, device=inputs_embeds.device)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]

    def generate(
        self,
        visual_tokens: torch.Tensor,
        prompts: list[str],
        max_new_tokens: int = 128,
    ) -> list[str]:
        device = visual_tokens.device
        encoding = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        text_embeds = self.model.model.embed_tokens(encoding["input_ids"])
        inputs_embeds = torch.cat([visual_tokens.to(text_embeds.dtype), text_embeds], dim=1)
        B = inputs_embeds.shape[0]
        vis_mask = torch.ones(B, visual_tokens.shape[1], device=device, dtype=torch.long)
        full_mask = torch.cat([vis_mask, encoding["attention_mask"]], dim=1)

        with torch.no_grad():
            out_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in out_ids
        ]
