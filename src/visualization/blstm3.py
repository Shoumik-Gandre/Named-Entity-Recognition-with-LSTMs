import torch
from os import PathLike
from torch.utils.tensorboard.writer import SummaryWriter

def visualize_model(model_path: PathLike):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define Model
    model = torch.load(model_path).to(device)
    words = torch.ones(64, 113).type(dtype=torch.int).to(device)
    capital = torch.ones(64, 113, 4).type(dtype=torch.int).to(device)
    char = torch.ones(64, 113, 68).type(dtype=torch.int).to(device)
    lengths = torch.ones(64, ).type(dtype=torch.int)
    print("[Loaded] Model")
    print("[Definition] Model")
    model = model.to(device)
    print(model)
    writer = SummaryWriter("runs/blstm3")
    writer.add_graph(model, (words, char, capital, lengths))
    writer.close()

    print("[---DONE---]")