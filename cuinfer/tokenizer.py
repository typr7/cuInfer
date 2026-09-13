import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cuinfer.protocol import FinishReason


@dataclass(frozen=True)
class ModelInfo:
    path: str
    name: str
    max_model_len: int
    eos_token_ids: list[int]
    vocab_size: int


def load_model(model: str) -> tuple[ModelInfo, PreTrainedTokenizerBase]:
    path = Path(model).expanduser()
    if path.is_dir():
        path = path.resolve()
        name = path.name
    else:
        name = model
        path = Path(snapshot_download(
            model,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
                "chat_template.jinja",
                "chat_templates/*.jinja",
            ],
        ))

    config = json.loads((path / "config.json").read_text())
    eos = config.get("eos_token_id")
    generation_path = path / "generation_config.json"
    if generation_path.exists():
        generation_eos = json.loads(generation_path.read_text()).get("eos_token_id")
        if generation_eos is not None:
            eos = generation_eos

    info = ModelInfo(
        path=str(path),
        name=name,
        max_model_len=config["max_position_embeddings"],
        eos_token_ids=[] if eos is None else ([eos] if isinstance(eos, int) else eos),
        vocab_size=config["vocab_size"],
    )
    tokenizer = AutoTokenizer.from_pretrained(info.path, local_files_only=True)
    return info, tokenizer


class IncrementalDetokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, prompt_token_ids: list[int]
    ) -> None:
        self.tokenizer = tokenizer
        # A short prompt suffix preserves the decoder's surrounding-space context.
        self.token_ids = prompt_token_ids[-5:]
        self.prefix_offset = 0
        self.read_offset = len(self.token_ids)

    def push(self, token_ids: list[int], finish_reason: int) -> str:
        if finish_reason == FinishReason.STOP:
            token_ids = token_ids[:-1]
        self.token_ids.extend(token_ids)
        prefix = self.tokenizer.decode(
            self.token_ids[self.prefix_offset:self.read_offset],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = self.tokenizer.decode(
            self.token_ids[self.prefix_offset:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # Keep incomplete byte sequences until a later token completes them.
        if len(text) <= len(prefix) or text.endswith("\ufffd"):
            return ""
        delta = text[len(prefix):]
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.token_ids)
        return delta
