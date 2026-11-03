"""답변 생성 모듈 (논문 2.2.3, 4.3): LLaMA 3.1 8B 4-bit + 3개 instruction."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .parse import referenced_tables

INSTRUCTIONS = """1. Deeply understand and analyze the provided rule information to accurately and insightfully answer the question.
2. If relevant equations or tables exist in the rule, they should be actively used in the answer.
3. Base your answer strictly on the related rule."""


def build_prompt(query, hits, table_lookup):
    blocks = []
    for i, h in enumerate(hits, 1):
        block = f"[{i}] {h['title']} (p.{h['page']})\n{h['content']}"
        for t in referenced_tables(h, table_lookup):
            block += f"\n{t['html']}"
        blocks.append(block)
    return "Related rule:\n" + "\n\n".join(blocks) + f"\n\nQuery: {query}"


class Generator:
    def __init__(self, model_name, load_in_4bit=True, max_new_tokens=768):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant, torch_dtype=torch.float16, device_map="auto")
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def answer(self, query, hits, table_lookup):
        messages = [{"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": build_prompt(query, hits, table_lookup)}]
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
