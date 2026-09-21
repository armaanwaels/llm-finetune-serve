"""Fixed prompt set with a controlled input length.

Prompts come from the fine-tuning dataset, so they look like real traffic for
this model. Some may overlap training rows; that does not affect latency. Each prompt starts at a different row, which keeps
prefixes distinct and stops a prefix cache from flattering the numbers. Every
prompt is cut to exactly `input_tokens` tokens with the served model's tokenizer.
"""
from lfs.data import PROMPT


def make_prompts(tokenizer_path: str, dataset_id: str, n: int, input_tokens: int, offset: int = 50_000) -> list[str]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    rows = load_dataset(dataset_id, split="train").select(range(offset, offset + n + 64))
    texts = [PROMPT.format(context=r["context"], question=r["question"]) for r in rows]
    prompts = []
    for i in range(n):
        ids: list[int] = []
        j = i
        while len(ids) < input_tokens:  # pad with following rows until long enough
            ids += tok(texts[j % len(texts)] + "\n\n", add_special_tokens=False)["input_ids"]
            j += 1
        prompts.append(tok.decode(ids[:input_tokens]))
    return prompts
