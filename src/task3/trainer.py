from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .model import BlstmCnn


@dataclass
class BlstmCnnTrainer:
    model: BlstmCnn
    criterion: nn.CrossEntropyLoss

    def __post_init__(self) -> None:
        assert isinstance(self.model, BlstmCnn)
        assert isinstance(self.criterion, nn.CrossEntropyLoss)

    def train_step(self, 
                   dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer, 
                   device: str="cpu") -> float:
        """
        Inputs:
            model is the Neural Network Architecture
            criterion is the loss function
            optimizer is the weight update optimization technique
            dataloder is the training dataset
            device is the device on which the training needs to be run on
        """
        model = self.model
        criterion = self.criterion

        total_loss = 0.0
        # Training
        model.train()
        for batch in tqdm(dataloader):

            word_encodings = batch['word_encodings'].to(device)
            char_encodings = batch['char_encodings'].to(device)

            capital_mask = batch['capital_mask'].to(device)

            lengths = batch['lengths']
            tag_encodings = batch['tag_encodings'].to(device)

            # forward
            out: torch.Tensor = model(word_encodings, char_encodings, capital_mask, lengths)

            # Loss
            loss: torch.Tensor = criterion(out.permute(0, 2, 1), tag_encodings)

            with torch.no_grad():
                total_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return total_loss

    def eval_step(self, 
                   dataloader: DataLoader, 
                   device: str="cpu") -> float:
        """
        Inputs:
            model is the Neural Network Architecture
            criterion is the loss function
            dataloder is the testing dataset
            device is the device on which the training needs to be run on
        """

        model = self.model
        criterion = self.criterion
        
        # Testing
        model.eval()
        num_batches = len(dataloader)
        total_loss: float = 0.0

        for batch in tqdm(dataloader):
            with torch.no_grad():

                word_encodings = batch['word_encodings'].to(device)
                char_encodings = batch['char_encodings'].to(device)
                capital_mask = batch['capital_mask'].to(device)
                lengths = batch['lengths']
                tag_encodings = batch['tag_encodings'].to(device)
                
                # forward
                out = model(word_encodings, char_encodings, capital_mask, lengths)

                # Loss
                loss = criterion(out.permute(0, 2, 1), tag_encodings)
                total_loss += loss.item()

        return total_loss
