import torch
from functools import partial
from os import PathLike
from ..dataset import Conll03Dataset, task1_collate
from ..task1.model import BLSTM
from ..task1.predict import unk_handler_preprocesser
from ..vocabulary import Vocabulary
from torch.utils.tensorboard.writer import SummaryWriter

def visualize_model(model_path: PathLike):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define Model
    model = torch.load(model_path).to(device)
    words = torch.ones(64, 113).type(dtype=torch.int).to(device)
    capital = torch.ones(64, 113, 4).type(dtype=torch.int).to(device)
    lengths = torch.ones(64, ).type(dtype=torch.int)
    y = model(words, capital, lengths)
    print("[Loaded] Model")
    print("[Definition] Model")
    model = model.to(device)
    print(model)
    writer = SummaryWriter("runs/blstm1")
    writer.add_graph(model, (words, capital, lengths))
    writer.close()

    print("[---DONE---]")