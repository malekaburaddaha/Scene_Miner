"""Qwen2.5-VL-7B VLM backend.

Loads Qwen2.5-VL with 4-bit NF4 quantisation and injects visual tokens
from the Q-Former via ``inputs_embeds`` prepend.

LoRA (PEFT) is applied to q_proj, v_proj in attention layers during
stage-3 training.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base_vlm import BaseVLM

_QWEN_HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class Qwen25VL(BaseVLM):
    """Qwen2.5-VL-7B-Instruct with optional 4-bit NF4 quantisation.

    Args:
        model_name: HuggingFace model ID.
        load_in_4bit: Enable BitsAndBytes NF4 quantisation.
        lora_r: LoRA rank (0 = no LoRA).
        lora_alpha: LoRA alpha scaling.
        lora_target_modules: Attention modules to apply LoRA.
    """

    def __init__(
        self,
        model_name: str = _QWEN_HF_ID,
        load_in_4bit: bool = True,
        lora_r: int = 0,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig

        bnb_cfg = None
        if load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="cpu" if not load_in_4bit else "auto",
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Qwen2.5-VL stores hidden_size under text_config, not at top level
        cfg = self.model.config
        self._d_model = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size

        # Freeze all base model params initially
        for p in self.model.parameters():
            p.requires_grad_(False)

        if lora_r > 0:
            self._apply_lora(
                lora_r, lora_alpha,
                lora_target_modules or ["q_proj", "v_proj"],
            )

    def _apply_lora(
        self,
        r: int,
        alpha: int,
        target_modules: list[str],
    ) -> None:
        """Wrap model with PEFT LoRA adapters."""
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepend visual tokens to text embeddings and run forward.

        Args:
            visual_tokens: (B, N_vis, d_vlm) from Q-Former.
            input_ids: (B, L_text) tokenised text input.
            attention_mask: Optional (B, N_vis + L_text) mask.

        Returns:
            hidden_states: (B, N_vis + L_text, d_vlm)
        """
        # get_input_embeddings() navigates through PEFT/LoRA wrappers and
        # Qwen2.5-VL's nested structure (ForConditionalGeneration → VLModel).
        text_embeds = self.model.get_input_embeddings()(input_ids)  # (B, L, d_vlm)
        visual_tokens = visual_tokens.to(text_embeds.dtype)

        # Prepend visual tokens
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # Build attention mask if not provided
        if attention_mask is None:
            B, L_total = inputs_embeds.shape[:2]
            attention_mask = torch.ones(B, L_total, device=inputs_embeds.device)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]  # (B, N_vis + L, d_vlm)

    def generate(
        self,
        visual_tokens: torch.Tensor,
        prompts: list[str],
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Generate captions conditioned on visual tokens.

        Args:
            visual_tokens: (B, N_vis, d_vlm).
            prompts: List of B text prompts.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            List of B generated caption strings.
        """
        device = visual_tokens.device
        encoding = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        text_embeds = self.model.get_input_embeddings()(encoding["input_ids"])
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
