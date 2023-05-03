from functools import partial
from os import PathLike
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .model import BLSTM, train_step, eval_step
from ..dataset import Conll03Dataset, task1_collate, case_feature_extractor
from .predict import unk_handler_preprocesser
from ..iodata import read_wordsequences_tagsequences, write_prediction_file
from ..preprocessing import LabelEncoder, calc_tag_weights
from ..vocabulary import Vocabulary
from .settings import (
    WORD_VOCAB_PATH,
    TAG_VOCAB_PATH,
    TAG_PAD_INDEX,
)
from ..settings import PERL_SCRIPT

import subprocess


# Hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 0.7
TRAIN_EMBEDDINGS = True
SHUFFLE = True

MOMENTUM = 0.9


def train_model(train_path: PathLike, dev_path: PathLike, temp_pred_path: PathLike, checkpoint_path: PathLike):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load Train data
    train_dataset = Conll03Dataset(train_path,
                                   TAG_VOCAB_PATH,
                                   feature_extractors={
                                       'word_encodings': partial(unk_handler_preprocesser, word_vocab_path=WORD_VOCAB_PATH),
                                       'capital_mask': case_feature_extractor},
                                   use_targets=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=SHUFFLE, collate_fn=partial(task1_collate, use_targets=True))
    print("[Loaded] Train data...")
    # --- End ---

    # --- Load Dev data ---
    dev_dataset = Conll03Dataset(dev_path,
                                 TAG_VOCAB_PATH,
                                   feature_extractors={
                                       'word_encodings': partial(unk_handler_preprocesser, word_vocab_path=WORD_VOCAB_PATH),
                                       'capital_mask': case_feature_extractor},
                                 use_targets=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                                collate_fn=partial(task1_collate, use_targets=True))
    print("[Loaded] Dev data...")
    # --- End ---

    # --- For writing Prediction file ---
    X_dev, y_dev = read_wordsequences_tagsequences(dev_path, ' ')
    tag_encoder = LabelEncoder(Vocabulary.from_file(TAG_VOCAB_PATH))
    print("[Loaded] Dev Write data...")
    # --- End ---
    vocab_size = len(Vocabulary.from_file(WORD_VOCAB_PATH))

    # Define Model
    model = BLSTM(vocab_size+1).to(device)

    # Get Tag Weights
    tag_encodings = pad_sequence(train_dataset.tag_encodings, batch_first=True)
    tag_weights = calc_tag_weights(tag_encodings).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=tag_weights, ignore_index=TAG_PAD_INDEX)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    start_epoch = 0

    # Check if we need to continue training from a checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    model.embedding.weight.requires_grad_(TRAIN_EMBEDDINGS)
    
    print(model)

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"[Epoch] {epoch + 1}")
        print("=> [Training]")
        train_loss = train_step(
            model, criterion, optimizer, train_dataloader, device)
        print(f"Training loss: {train_loss}")
        print("=> [Validation]")
        dev_loss = eval_step(model, criterion, dev_dataloader, device)
        print(f"Validation loss: {dev_loss}")

        # save model checkpoint
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, checkpoint_path)

            print("=> [Prediction]")
            y_pred = model.predict(dev_dataloader, device)
            y_pred = tag_encoder.inverse_transform(y_pred)

            write_prediction_file(temp_pred_path, wordsequences=X_dev, tagsequences=y_dev,
                                  predsequences=y_pred, delimiter=' ')

            p1 = subprocess.run(f"perl \"{PERL_SCRIPT}\" < \"{temp_pred_path}\"",
                                capture_output=True, shell=True, text=True)

            f1_score = float(p1.stdout.split('\n')[1][-6:])
            print(f"{f1_score = }")
