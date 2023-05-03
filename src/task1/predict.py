from functools import partial
from os import PathLike
import torch
from torch.utils.data import DataLoader

from .unk_handle import PseudoWordUnknownHandler

from .model import BLSTM
from ..dataset import Conll03Dataset, case_feature_extractor, task1_collate
from ..vocabulary import Vocabulary
from ..preprocessing import LabelEncoder
from ..iodata import read_wordsequences, write_wordsequences_tagsequences
from .settings import (
    WORD_VOCAB_PATH,
    TAG_VOCAB_PATH,
)


def unk_handler_preprocesser(
        sequences: list[list[str]],
        word_vocab_path: PathLike) -> list[torch.Tensor]:

    # Label Encode the Word Sequences
    unk_handler = PseudoWordUnknownHandler()
    word_vocab = Vocabulary.from_file(word_vocab_path)
    words = set(word for word, _ in word_vocab.items())
    sequences = [
        [word.upper() for word in sequence] 
        for sequence in sequences
    ]
    sequences = unk_handler.replace(sequences, words)

    word_encoder = LabelEncoder(word_vocab)
    word_sequences_le = word_encoder.transform(sequences, key=lambda x: x.upper())
    
    # Create List of Label Encoded Word Sequence Tensors
    return [torch.tensor(sequence) for sequence in word_sequences_le]


def generate_prediction_file(input_path: PathLike, output_path: PathLike, model_path: PathLike):

    # raise NotImplementedError()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    # --- Load Test data ---
    dev_dataset = Conll03Dataset(input_path,
                                 TAG_VOCAB_PATH,
                                 feature_extractors={
                                     'word_encodings': partial(unk_handler_preprocesser, word_vocab_path=WORD_VOCAB_PATH),
                                    'capital_mask': case_feature_extractor},
                                 use_targets=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64,
                                collate_fn=partial(task1_collate, use_targets=False))
    print("[Loaded] data...")
    # --- End ---

    # --- For writing Prediction file ---
    X = read_wordsequences(input_path, ' ')
    tag_encoder = LabelEncoder(Vocabulary.from_file(TAG_VOCAB_PATH))

    # Load Embeddings for model
    vocab_size = len(Vocabulary.from_file(WORD_VOCAB_PATH))

    # Define Model
    model = BLSTM(vocab_size=vocab_size+1).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    print("[Loaded] Model")
    print("[Definition] Model")
    model = model.to(device)
    print(model)

    print("[Process] predicting...")
    y_pred = model.predict(dev_dataloader, device)
    y_pred = tag_encoder.inverse_transform(y_pred)

    write_wordsequences_tagsequences(output_path,
                                     wordsequences=X,
                                     tagsequences=y_pred,
                                     delimiter=' ')

    print("[---DONE---]")
