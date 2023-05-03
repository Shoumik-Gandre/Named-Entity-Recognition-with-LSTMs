from pathlib import Path

# File Paths
BASE_DIR = Path(r"D:\USC\College\Courses\CSCI-544 Applied Natural Language Processing\homework\4")

task = "3"

# Vocab
WORD_VOCAB_PATH = BASE_DIR / "models" / task / "vocab" / "word_vocab.json"
TAG_VOCAB_PATH = BASE_DIR / "models" / task  / "vocab" / "tag_vocab.json"
CHAR_VOCAB_PATH = BASE_DIR / "models" / task / "vocab" / "char_vocab.json"


# Globals
WORD_PAD_INDEX = 0
TAG_PAD_INDEX = -1
