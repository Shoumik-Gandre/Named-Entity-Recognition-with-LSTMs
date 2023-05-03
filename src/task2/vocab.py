from os import PathLike
from ..vocabulary import Vocabulary
from ..iodata import read_glove_words, read_wordsequences_tagsequences
from ..settings import GLOVE_EMBEDDINGS_PATH
from .settings import WORD_VOCAB_PATH, TAG_VOCAB_PATH


def generate_vocabulary(input_path: PathLike):
    word_to_index = read_glove_words(GLOVE_EMBEDDINGS_PATH)
    _, tag_sequences = read_wordsequences_tagsequences(input_path, ' ')

    # Preprocess Train data
    # => Unknown words
    word_vocab = Vocabulary.from_dict(word_to_index)
    tag_vocab = Vocabulary.from_sequences(tag_sequences, start_index=0)

    word_vocab.to_file(WORD_VOCAB_PATH)
    tag_vocab.to_file(TAG_VOCAB_PATH)
