from pathlib import Path

# File Paths
BASE_DIR = Path(r"D:\USC\College\Courses\CSCI-544 Applied Natural Language Processing\homework\4")

# Inputs
TRAIN_FILE_PATH = BASE_DIR / "data" / "train"
DEV_FILE_PATH = BASE_DIR / "data" / "dev"
GLOVE_EMBEDDINGS_PATH = BASE_DIR / "embeddings" / "glove.6B.100d.txt"
PERL_SCRIPT = BASE_DIR / "conll03eval.txt"


# Globals
WORD_PAD_INDEX = 0
TAG_PAD_INDEX = -1