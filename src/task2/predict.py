from functools import partial
from os import PathLike
import torch
from torch.utils.data import DataLoader

from .model import Blstm
from ..dataset import Conll03Dataset, task2_collate, case_feature_extractor, word_preprocessor
from ..vocabulary import Vocabulary
from ..preprocessing import LabelEncoder
from ..iodata import read_wordsequences, read_glove_embeddings, write_wordsequences_tagsequences
from .settings import (
    WORD_VOCAB_PATH,
    TAG_VOCAB_PATH,
)
from ..settings import GLOVE_EMBEDDINGS_PATH


def generate_prediction_file(input_path: PathLike, output_path: PathLike, model_path: PathLike):
    print("[PRED] blstm2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    # --- Load Test data ---
    dataset = Conll03Dataset(input_path,
                                 TAG_VOCAB_PATH,
                                 feature_extractors={
                                     'word_encodings': partial(word_preprocessor, word_vocab_path=WORD_VOCAB_PATH),
                                     'capital_mask': case_feature_extractor},
                                 use_targets=False)
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                                collate_fn=partial(task2_collate, use_targets=False))
    print("[Loaded] data...")
    # --- End ---

    # --- For writing Prediction file ---
    X = read_wordsequences(input_path, ' ')
    tag_encoder = LabelEncoder(Vocabulary.from_file(TAG_VOCAB_PATH))

    # Load Embeddings for model
    _, _, embeddings = read_glove_embeddings(GLOVE_EMBEDDINGS_PATH)
    print("[Loaded] Glove Embeddings...")

    # Define Model
    model = Blstm(embeddings).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    print("[Loaded] Model")
    print("[Definition] Model")
    model = model.to(device)
    print(model)

    print("[Process] predicting...")
    y_pred = model.predict(dataloader, device)
    y_pred = tag_encoder.inverse_transform(y_pred)

    write_wordsequences_tagsequences(output_path,
                                     wordsequences=X,
                                     tagsequences=y_pred,
                                     delimiter=' ')

    print("[---DONE---]")
