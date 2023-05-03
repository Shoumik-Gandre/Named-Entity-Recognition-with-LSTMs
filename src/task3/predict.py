from os import PathLike
import torch
from torch.utils.data import DataLoader
from functools import partial

from .model import BlstmCnn
from ..dataset import Conll03Dataset, task3_collate, character_feature_extractor, case_feature_extractor, word_preprocessor
from ..vocabulary import Vocabulary
from ..preprocessing import LabelEncoder
from ..iodata import read_wordsequences, read_glove_embeddings, write_wordsequences_tagsequences
from .settings import (
    WORD_VOCAB_PATH,
    TAG_VOCAB_PATH,
    CHAR_VOCAB_PATH,
)
from ..settings import GLOVE_EMBEDDINGS_PATH


def generate_prediction_file(input_path: PathLike, output_path: PathLike, model_path: PathLike):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    # --- Load Test data ---
    dataset = Conll03Dataset(input_path,
                             TAG_VOCAB_PATH,
                             feature_extractors={
                                 'word_encodings': partial(word_preprocessor, word_vocab_path=WORD_VOCAB_PATH),
                                 'capital_mask': case_feature_extractor,
                                 'char_encodings': partial(
                                     character_feature_extractor,
                                     char_vocab_path=CHAR_VOCAB_PATH,
                                     word_vocab_path=WORD_VOCAB_PATH)},
                             use_targets=False)
    dataloader = DataLoader(dataset, batch_size=64,
                            collate_fn=partial(task3_collate, use_targets=False))
    print("[Loaded] data...")
    # --- End ---

    # --- For writing Prediction file ---
    X_test = read_wordsequences(input_path, ' ')
    tag_encoder = LabelEncoder(Vocabulary.from_file(TAG_VOCAB_PATH))

    # Load Embeddings for model
    _, _, embeddings = read_glove_embeddings(GLOVE_EMBEDDINGS_PATH)
    print("[Loaded] Glove Embeddings...")

    # Model
    model = BlstmCnn(embeddings, num_char_embeddings=85).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model = model.to(device)
    print("[Loaded] Model")
    print("[Definition] Model")
    print(model)

    print("[Process] predicting...")
    y_pred = model.predict(dataloader, device)
    y_pred = tag_encoder.inverse_transform(y_pred)

    write_wordsequences_tagsequences(output_path,
                                     wordsequences=X_test,
                                     tagsequences=y_pred,
                                     delimiter=' ')

    print("[---DONE---]")
