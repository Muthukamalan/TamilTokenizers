import unicodedata

__all__ = [
    "Tokenizer",
    "get_stats",
    "merge",
    "get_compression_ratio",
    "replace_control_characters",
    "render_tokens",
]


def get_stats(
    ids: list[int], counts: dict[tuple[int, int], int] = None
) -> dict[tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update existing dictionary of counts

    returns: { (pair): frequency }
    """
    counts = {} if counts is None else counts
    # iterate consecutive elements
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: dict[tuple, int], idx: int) -> list[int]:
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids: list = []
    i = 0
    while i < len(ids):
        # i<len(ids) covers edge case
        if (ids[i] == pair[0]) and (i < len(ids) - 1) and (ids[i + 1] == pair[1]):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def get_compression_ratio(text: str, tokenizer: "Tokenizer"):
    tokens = tokenizer.encode(text)
    return len(tokens) / len(text)


def replace_control_characters(s: str) -> str:
    """
    we don't want to print control characters
    which distort the output (e.g. \n or much worse)
    https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    http://www.unicode.org/reports/tr44/#GC_Values_Table
    """
    chars: list = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_tokens(b: bytes) -> str:
    s = b.decode("utf-8", errors="replace")
    return replace_control_characters(s)


class Tokenizer:
    """
    Base class for Tokenizer
    """

    __slots__ = ["merges", "pattern", "spl_tokens", "vocab"]

    def __init__(self):
        #  default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges: dict[tuple[int, int], int] = {}  # (int,int) -> int
        self.pattern: str = ""  # str
        self.spl_tokens: dict[str, int] = {}  # str->int e.g) {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int ->bytes

    def train(self, text: str, vocab_size: int, verbose=False):
        raise NotImplementedError

    def encode(self, text: str):
        raise NotImplementedError

    def decode(self, ids: list[int]):
        raise NotImplementedError

    def _build_vocab(self):
        """
        Vocab is simply and deterministically derived from merges
        """
        #  Before any merging occurs, the BPE tokenizer initializes its vocabulary with these 256 base bytes (ASCII, characters, symbols, etc.). the tokenizer can handle any UTF-8 character, which is crucial for multilingual support.
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for spl, idx in self.spl_tokens.items():
            vocab[idx] = spl.encode("utf-8")

        return vocab

    def save(self, file_prefix: str) -> None:
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        """

        model_file: str = file_prefix + ".model"  # intended for load()
        with open(model_file, "w") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")

            # write number of spl tokens
            f.write(f"{len(self.spl_tokens)}\n")
            # write spl tokens
            for special, idx in self.spl_tokens.items():
                f.write(f"{special} {idx}\n")

            # the merge dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"  # intended for humans to compare
        inverted_merges: dict[int, tuple[int, int]] = {
            idx: pair for pair, idx in self.merges.items()
        }

        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # if token may be partial utf-8 sequences can't decode into valid strings.
                # so error='replace' with char � so we can't use `.vocab` in load()
                s = render_tokens(token)

                # find the child of this tokens, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as as merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_tokens(self.vocab[idx0])
                    s1 = render_tokens(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    f.write(f"[{s}] [{idx}]\n")

    def load(self, model_file: str) -> "Tokenizer":
        """Inverse the save() but only for model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        spl_tokens = {}
        idx = 256

        with open(model_file, "r", encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"

            # read the pattern
            self.pattern = f.readline().strip()

            # read the number of spl tokens
            num_special = int(f.readline().strip())

            for _ in range(num_special):
                spl, spl_idx = f.readline().strip().split()
                spl_tokens[spl] = int(spl_idx)

            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.spl_tokens = spl_tokens
        self.vocab = self._build_vocab()
