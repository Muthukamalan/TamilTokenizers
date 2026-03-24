import os
import time
from pathlib import Path
from minbpe import BasicTokenizer, RegexTokenizer

fpath = "/home/muthu/DATA/austen-emma.txt"
fpath = "./data/processed/tamil_corpus.txt"
text = open(f"{fpath}", "r", encoding="utf-8").read()


os.makedirs("models/", exist_ok=True)


# for TokenizerClass, prefix_name in zip(
#     [RegexTokenizer, BasicTokenizer], ["regex", "basic"]
# ):

#     tokenizer = TokenizerClass()
#     tokenizer.train(text, vocab_size=257, verbose=True)
#     tokenizer.save(os.path.join("models", prefix_name))


tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size=1000, verbose=True)
tokenizer.save(os.path.join("models", "regex"))
