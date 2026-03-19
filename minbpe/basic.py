"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge, get_compression_ratio


class BasicTokenizer(Tokenizer):
    __slots__ = ()

    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose=False):
        """
        1. convert text->UTF-8
        2. Map each bytes (0-255) to a unique UNICODE character
        3. run bpe algo
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes: bytes = text.encode("utf-8")

        ids = list(text_bytes)  # list of integers in range 0..255
        initial_length_ids = len(ids)

        # iteratively merge the most common pairs to create new toens
        merges: dict[tuple[int, int], int] = {}

        # vocab = [(0, b'\x00'), (1, b'\x01'),.., (32, b' '), (33, b'!'), (34, b'"'),..(67, b'C'), (68, b'D'), (69, b'E'),...,(254, b'\xfe'), (255, b'\xff')]   ]
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # counts up the number of times every consecutive paurs appers
            stats: dict[tuple[int, int], int] = get_stats(ids)

            # find the pair with the highest frequency/count
            pair: tuple[int, int] = max(stats, key=stats.get)

            # assign new token
            idx = 256 + i

            # replace all occurance of pair in ids with idx; replace ids
            ids = merge(ids, pair=pair, idx=idx)

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )
                print(f"compression ratio: {initial_length_ids / len(ids):.2f}X")

            # save class variables
            self.merges = merges
            self.vocab = vocab

    def decode(self, ids: list[int]) -> str:
        # given ids (list of integers), return str
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)  # list of integers in range 0...255

        while len(ids) >= 2:
            # find the pair with lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # if there are no more merge available stop
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair=pair, idx=idx)

        return ids
