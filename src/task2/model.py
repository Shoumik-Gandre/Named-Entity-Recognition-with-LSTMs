from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader

    
class Blstm(nn.Module):

    def __init__(self,
                 embeddings: np.ndarray,
                 embedding_size: int=100, 
                 hidden_size: int=256, 
                 linear_size: int=128, 
                 out_size: int=9,
                 padding_idx:int=0) -> None:
        
        super(Blstm, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), padding_idx=padding_idx)

        # LSTM
        self.bilstm = nn.LSTM(input_size=embedding_size+4, 
                              hidden_size=hidden_size, 
                              num_layers=1, 
                              batch_first=True, 
                              bidirectional=True)
        
        self.lstm_dropout = nn.Dropout(0.33)
        
        # Linear-ELU
        self.linear_elu = nn.Sequential(nn.Linear(2*hidden_size, linear_size),
                                        nn.ELU())

        # Classifier
        self.classifier = nn.Linear(linear_size, out_size)
        

    def forward(self, x: torch.Tensor, capital_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        
        embeddings = self.embedding(x)

        output = torch.cat((embeddings, capital_mask), dim=-1)

        # LSTM
        output = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.lstm_dropout(output)
        # Linear - ELU
        output = self.linear_elu(output)

        # Classifier
        output = self.classifier(output)
        return output
    
    def predict(model: nn.Module,
         dataloader: DataLoader,
         device: str="cpu") -> list[list[int]]:
        """
        Inputs:
            model is the Neural Network Architecture
            dataloder is the testing dataset
            device is the device on which the training needs to be run on
        """
        # Testing
        model.eval()
        y_pred = []

        for batch in tqdm(dataloader):
            with torch.no_grad():
                
                word_encodings = batch['word_encodings'].to(device)
                capital_mask = batch['capital_mask'].to(device)
                lengths = batch['lengths']

                probabilities: torch.Tensor = model(word_encodings, capital_mask, lengths)
                predictions = probabilities.argmax(-1)
                predictions = predictions.tolist()
                predictions = [prediction[:length] for prediction, length in zip(predictions, lengths)]            

                y_pred.extend(predictions)

        return y_pred


def train_step(model: Blstm,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler,
          dataloader: DataLoader,
          device: str="cpu") -> float:
    """
    Inputs:
        model is the Neural Network Architecture
        criterion is the loss function
        optimizer is the weight update optimization technique
        dataloder is the training dataset
        device is the device on which the training needs to be run on
    """
    total_loss = 0.0
    # Training
    model.train()
    for batch in tqdm(dataloader):

        word_encodings = batch['word_encodings'].to(device)
        capital_mask = batch['capital_mask'].to(device)

        lengths = batch['lengths']
        tag_encodings = batch['tag_encodings'].to(device)

        # forward
        out: torch.Tensor = model(word_encodings, capital_mask, lengths)

        # Loss
        loss: torch.Tensor = criterion(out.permute(0, 2, 1), tag_encodings)

        with torch.no_grad():
            total_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)


def eval_step(model: Blstm,
             criterion: nn.Module,
             dataloader: DataLoader, 
             device: str="cpu") -> float:
    """
    Inputs:
        model is the Neural Network Architecture
        criterion is the loss function
        dataloder is the testing dataset
        device is the device on which the training needs to be run on
    """
    # Testing
    model.eval()
    num_batches = len(dataloader)
    total_loss: float = 0.0

    for batch in tqdm(dataloader):
        with torch.no_grad():

            word_encodings = batch['word_encodings'].to(device)
            capital_mask = batch['capital_mask'].to(device)
            lengths = batch['lengths']
            tag_encodings = batch['tag_encodings'].to(device)
            
            # forward
            out = model(word_encodings, capital_mask, lengths)

            # Loss
            loss = criterion(out.permute(0, 2, 1), tag_encodings)
            total_loss += loss.item()

    mean_loss = total_loss / num_batches
    return mean_loss
