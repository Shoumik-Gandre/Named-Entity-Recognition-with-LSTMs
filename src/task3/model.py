import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
    

class TokenCharacterEncoder(nn.Module):
    """
    Encodes tokens into 30 dimensions characterwise
    It requires label encoded characters to function
    """
    
    def __init__(self, num_embeddings: int, character_pad_value: int=0):
        super(TokenCharacterEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim=30)
        self.conv = nn.Conv1d(30, 30, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(30, 30, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(30, 30, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(30, 30, kernel_size=3, padding=1)

        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.character_pad_value = character_pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        char_encodings.shape - (batch size, sentence length, word length)
        """
        pad_mask = (x == self.character_pad_value)
        pad_mask = pad_mask.repeat(30, 1, 1, 1).permute(1, 2, 3, 0)
        pad_mask = pad_mask.flatten(0, 1).permute(0, 2, 1).to(x.device)

        N, S = x.shape[:2]
        x = x.flatten(0, 1)
        # out = self.single_sentence_forward(char_encodings)
        x = self.embedding(x) 
        # x.shape - (sentence length, word length, num_embeddings)
        x = x.permute(0, 2, 1)
        # x.shape - (sentence length, num_embeddings, word length)
        x = F.relu(self.conv(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.masked_fill(pad_mask, float("-inf"))
        # x.shape - (sentence length, conv out dim=30, word length)
        x = self.maxpool(x).squeeze(-1)
        # x.shape - (sentence length, conv out dim=30)
        x = x.unflatten(0, (N, S))
        # out.shape: (batch size, sentence length, 30)
        return x
    
    
class BlstmCnn(nn.Module):

    def __init__(self,
                 glove_embeddings: np.ndarray,
                 num_char_embeddings: int=100,
                 token_padding_idx:int=0) -> None:
        
        super(BlstmCnn, self).__init__()

        # Glove Embeddings
        self.glove_encoder = nn.Embedding.from_pretrained(
            torch.from_numpy(glove_embeddings).float(), 
            padding_idx=token_padding_idx
        )

        # Character Level Embeddings
        self.token_encoder = TokenCharacterEncoder(num_embeddings=num_char_embeddings)

        # LSTM
        self.bilstm = nn.LSTM(input_size=100+30+1+3, hidden_size=256, 
                              num_layers=1, batch_first=True, bidirectional=True)
        
        self.lstm_dropout = nn.Dropout(0.33)
        
        # Linear-ELU
        self.linear_elu = nn.Sequential(nn.Linear(2*256, 128), nn.ELU())

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(128, 9))
        

    def forward(self, 
                word_encodings: torch.Tensor, 
                char_encodings: torch.Tensor, 
                capital_mask: torch.Tensor, 
                lengths: torch.Tensor) -> torch.Tensor:
        glove_embeddings = self.glove_encoder(word_encodings)

        token_embeddings = self.token_encoder(char_encodings)

        output = torch.cat((glove_embeddings, token_embeddings, capital_mask), dim=-1)

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
                char_encodings = batch['char_encodings'].to(device)
                capital_mask = batch['capital_mask'].to(device)
                lengths = batch['lengths']

                probabilities: torch.Tensor = model(word_encodings, char_encodings, capital_mask, lengths)
                predictions = probabilities.argmax(-1)
                predictions = predictions.tolist()
                predictions = [prediction[:length] for prediction, length in zip(predictions, lengths)]            

                y_pred.extend(predictions)

        return y_pred


def train_step(model: BlstmCnn,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
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
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    
    return total_loss # / len(dataloader)


def eval_step(model: nn.Module,
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
            char_encodings = batch['char_encodings'].to(device)
            capital_mask = batch['capital_mask'].to(device)
            lengths = batch['lengths']
            tag_encodings = batch['tag_encodings'].to(device)
            
            # forward
            out = model(word_encodings, char_encodings, capital_mask, lengths)

            # Loss
            loss = criterion(out.permute(0, 2, 1), tag_encodings)
            total_loss += loss.item()

    mean_loss = total_loss #/ num_batches
    return mean_loss
