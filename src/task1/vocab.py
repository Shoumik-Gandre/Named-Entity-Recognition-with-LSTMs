from os import PathLike

from ..preprocessing import count_items
from ..vocabulary import Vocabulary
from ..iodata import read_wordsequences_tagsequences
from .unk_handle import PseudoWordUnknownHandler
from .settings import WORD_VOCAB_PATH, TAG_VOCAB_PATH


def generate_vocabulary(input_path: PathLike):
    
    word_sequences, tag_sequences = read_wordsequences_tagsequences(input_path, ' ')
    tag_vocab = Vocabulary.from_sequences(tag_sequences, start_index=0)
    tag_vocab.to_file(TAG_VOCAB_PATH)
    
    frequency = count_items(word_sequences, transform=lambda word: word.upper())
    unk_handler = PseudoWordUnknownHandler()
    frequency = unk_handler.transform_freq(frequency, threshold=2)
    word_vocab = Vocabulary({value: index for index, value in enumerate(frequency.keys(), start=1)})
    word_vocab.to_file(WORD_VOCAB_PATH)