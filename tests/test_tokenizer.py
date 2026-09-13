import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from cuinfer.protocol import FinishReason
from cuinfer.tokenizer import IncrementalDetokenizer, load_model


class ByteTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return bytes(token for token in token_ids if token != 256).decode(
            "utf-8", errors="replace"
        )


def test_split_unicode_and_batched_tokens():
    decoder = IncrementalDetokenizer(ByteTokenizer(), list(b"Prompt: "))
    assert decoder.push([0xE4], FinishReason.RUNNING) == ""
    assert decoder.push([0xBD], FinishReason.RUNNING) == ""
    assert decoder.push([0xA0], FinishReason.RUNNING) == "\u4f60"
    assert decoder.push([0xF0, 0x9F], FinishReason.RUNNING) == ""
    assert decoder.push([0x98, 0x80, ord("!")], FinishReason.LENGTH) == "\U0001f600!"


def test_stop_strips_last_token_and_skips_other_special_tokens():
    decoder = IncrementalDetokenizer(ByteTokenizer(), list(b"Prompt"))
    assert decoder.push([ord("A"), 256], FinishReason.RUNNING) == "A"
    assert decoder.push([ord("B"), ord("Z")], FinishReason.STOP) == "B"


@pytest.mark.parametrize("reason", [FinishReason.STOP, FinishReason.LENGTH])
def test_incomplete_terminal_bytes_remain_withheld(reason):
    decoder = IncrementalDetokenizer(ByteTokenizer(), [])
    assert decoder.push([0xE4], FinishReason.RUNNING) == ""
    terminal = [256] if reason == FinishReason.STOP else [0xBD]
    assert decoder.push(terminal, reason) == ""


def test_context_dependent_leading_spaces():
    class SpaceTokenizer:
        def decode(self, token_ids, **kwargs):
            return " ".join({1: "prompt", 2: "hello", 3: "world"}[i] for i in token_ids)

    decoder = IncrementalDetokenizer(SpaceTokenizer(), [1])
    assert decoder.push([2], FinishReason.RUNNING) == " hello"
    assert decoder.push([3], FinishReason.LENGTH) == " world"


@pytest.fixture
def model_directory(tmp_path):
    path = tmp_path / "tiny-model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "max_position_embeddings": 128,
        "vocab_size": 257,
        "eos_token_id": 256,
    }))
    return path


@pytest.fixture
def tokenizer_loader(monkeypatch):
    loader = Mock(return_value=ByteTokenizer())
    monkeypatch.setattr("cuinfer.tokenizer.AutoTokenizer.from_pretrained", loader)
    return loader


@pytest.mark.parametrize("generation_eos", [None, 255, [254, 255]])
def test_load_local_model_metadata(model_directory, tokenizer_loader, generation_eos):
    (model_directory / "generation_config.json").write_text(
        json.dumps({"eos_token_id": generation_eos})
    )
    info, tokenizer = load_model(str(model_directory))
    expected_eos = [256] if generation_eos is None else (
        [generation_eos] if isinstance(generation_eos, int) else generation_eos
    )
    assert info.name == "tiny-model"
    assert info.path == str(model_directory)
    assert info.max_model_len == 128
    assert info.vocab_size == 257
    assert info.eos_token_ids == expected_eos
    assert tokenizer is tokenizer_loader.return_value
    tokenizer_loader.assert_called_once_with(str(model_directory), local_files_only=True)


def test_download_model_uses_only_runtime_files(
    model_directory, tokenizer_loader, monkeypatch
):
    download = Mock(return_value=str(model_directory))
    monkeypatch.setattr("cuinfer.tokenizer.snapshot_download", download)
    info, _ = load_model("organization/tiny-model")
    assert info.name == "organization/tiny-model"
    assert info.eos_token_ids == [256]
    assert download.call_args.args == ("organization/tiny-model",)
    patterns = download.call_args.kwargs["allow_patterns"]
    assert "model.safetensors" in patterns
    assert "config.json" in patterns
    assert "tokenizer.json" in patterns
    assert "tokenizer_config.json" in patterns
    assert "chat_template.jinja" in patterns
    assert all(pattern != "*" and "pytorch" not in pattern for pattern in patterns)


def test_current_directory_model_name(model_directory, tokenizer_loader, monkeypatch):
    monkeypatch.chdir(model_directory)
    info, _ = load_model(".")
    assert info.name == "tiny-model"
    assert info.path == str(model_directory)


@pytest.fixture(scope="module")
def local_tokenizer():
    path = Path(__file__).resolve().parents[1] / "models" / "qwen3-0.6b"
    if not path.is_dir():
        pytest.skip("local Qwen3 tokenizer is unavailable")
    return load_model(str(path))


@pytest.mark.parametrize("text", [" Hello, world!", "\u4f60\u597d\U0001f600 caf\u00e9", "line one\nline two"])
def test_real_tokenizer_stream_matches_decode(local_tokenizer, text):
    _, tokenizer = local_tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi"}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoder = IncrementalDetokenizer(tokenizer, prompt)
    chunks = [decoder.push([token], FinishReason.RUNNING) for token in tokens]
    chunks.append(decoder.push([tokenizer.eos_token_id], FinishReason.STOP))
    assert "".join(chunks) == text
    assert all("\ufffd" not in chunk for chunk in chunks)


def test_real_model_config_eos_precedence(local_tokenizer):
    info, _ = local_tokenizer
    config = json.loads((Path(info.path) / "config.json").read_text())
    generation = json.loads((Path(info.path) / "generation_config.json").read_text())
    assert info.max_model_len == config["max_position_embeddings"]
    assert info.eos_token_ids == generation["eos_token_id"]
