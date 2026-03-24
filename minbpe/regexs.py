"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from typing import Literal
from .base import Tokenizer, get_stats, merge
from tqdm import tqdm

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# TAMIL_SPLIT_PATTERN_V2 = r"""'\s*(?:[\u0B80-\u0BFF]+|\d+|[^\s\w\u0B80-\u0BFF])"""
# TAMIL_SPLIT_PATTERN_V3 = r"""'\s*(?:[\u0B82-\u0BFA]+|\d+|[^\s\w\u0B82-\u0BFA])"""
TAMIL_SPLIT_PATTERN = r"[\u0B80-\u0BFF]+"


def count_tokens(ids: list[list[int]]) -> int:
    tot_length = 0
    for id in ids:
        tot_length += len(id)
    return tot_length


class RegexTokenizer(Tokenizer):
    __slots__ = ("pattern", "spl_tokens", "inv_spl_tokens", "compiled_pattern")

    def __init__(self, pattern=None):
        super().__init__()

        self.pattern = TAMIL_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.spl_tokens: dict[str, int] = {}
        self.inv_spl_tokens: dict[int, str] = {}

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= 256
        num_merges: int = vocab_size - 256

        # split the text up into text chunks
        text_chunks: list[str] = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids: list[list[int]] = [
            list(t_chunk.encode("utf-8")) for t_chunk in text_chunks
        ]

        # iteratively merge the most common pairs to create new token
        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}  # keep it persistance state

            for chunk_ids in ids:
                get_stats(chunk_ids, stats)  # passing everytime state so it'll update

            pair = max(stats, key=stats.get)  # find the hightes count
            idx = 256 + i  # new idx for new created token

            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose and i%10==0:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )
                print(
                    f"compression ratio: {len(text.encode('utf-8')) / count_tokens(ids):.2f}X"
                )

            self.merges = merges
            self.vocab = vocab

    def register_special_tokens(self, special_tokens: dict[str, int]):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.spl_tokens = special_tokens
        self.inv_spl_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids: list[int]) -> str:
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inv_spl_tokens:
                part_bytes.append(self.inv_spl_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")

    def encode_ordinary(self, text: str) -> list[int]:
        txt_chunks: list[str] = re.findall(self.compiled_pattern, text)

        ids: list[int] = []
        for chunk in txt_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def _encode_chunk(self, text_bytes: list[bytes]) -> list[int]:
        # first conver all bytes to int in range 0..255
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # if pair not found break it
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(
        self,
        text: str,
        allowed_special: Literal["none_raise", "all", "none"] = "none_raise",
    ):
        special = None
        match allowed_special:
            case "all":
                special = self.spl_tokens
            case "none":
                special = {}
            case "none_raise":
                special = {}
                assert all(token not in text for token in self.spl_tokens)
            case isinstance(allowed_special, set):
                special = {
                    k: v for k, v in self.spl_tokens.items() if k in allowed_special
                }
            case _:
                raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # no spl token just use ordinary encoding
            return self.encode_ordinary(text=text)

        # otherwise, we have to be careful with potential special tokens in text, we handle special tokens by splitting the text
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(
            special_pattern, text
        )  # # now all the special characters are separated from the rest of the text
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
