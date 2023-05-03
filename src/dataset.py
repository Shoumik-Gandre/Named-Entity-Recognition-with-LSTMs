from typing import Callable
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from os import PathLike

from .iodata import read_wordsequences_tagsequences, read_wordsequences
from .preprocessing import LabelEncoder, CharacterLabelEncoder, sequence_condition_mask
from .vocabulary import Vocabulary, CharacterVocabulary


def task1_collate(data: list[dict], use_targets: bool) -> dict[str, torch.Tensor]:
    word_encodings = [torch.Tensor(x['word_encodings']) for x in data]
    word_encodings = pad_sequence(word_encodings, batch_first=True)

    lengths = torch.IntTensor([len(x['word_encodings']) for x in data])

    capital_mask = [torch.Tensor(x['capital_mask']) for x in data]
    capital_mask = pad_sequence(capital_mask, batch_first=True)

    item: dict[str, torch.Tensor] = {
        'word_encodings': word_encodings,
        'lengths': lengths,
        'capital_mask': capital_mask,
    }

    if use_targets: 
        tag_encodings = [torch.Tensor(x['tag_encodings']) for x in data] # type: ignore
        tag_encodings = pad_sequence(tag_encodings, batch_first=True)
        item |= {'tag_encodings': tag_encodings}
    
    return item


def task2_collate(data: list[dict], use_targets: bool) -> dict[str, torch.Tensor]:
    word_encodings = [torch.Tensor(x['word_encodings']) for x in data]
    capital_mask = [torch.Tensor(x['capital_mask']) for x in data]

    lengths = torch.IntTensor([len(x['word_encodings']) for x in data])

    word_encodings = pad_sequence(word_encodings, batch_first=True)
    capital_mask = pad_sequence(capital_mask, batch_first=True)
    
    item: dict[str, torch.Tensor] = {
        'word_encodings': word_encodings,
        'lengths': lengths,
        'capital_mask': capital_mask,
    }

    if use_targets: 
        tag_encodings = [torch.Tensor(x['tag_encodings']) for x in data] # type: ignore
        tag_encodings = pad_sequence(tag_encodings, batch_first=True)
        item |= {'tag_encodings': tag_encodings}
    
    return item


def task3_collate(data: list[dict], use_targets: bool) -> dict[str, torch.Tensor]:
    
    word_encodings = [torch.Tensor(x['word_encodings']) for x in data]
    char_encodings = [torch.Tensor(x['char_encodings']) for x in data]

    capital_mask = [torch.Tensor(x['capital_mask']) for x in data]

    lengths = torch.IntTensor([len(x['word_encodings']) for x in data])

    word_encodings = pad_sequence(word_encodings, batch_first=True)
    char_encodings = pad_sequence(char_encodings, batch_first=True)
    
    capital_mask = pad_sequence(capital_mask, batch_first=True)
    
    item: dict[str, torch.Tensor] = {
        'word_encodings': word_encodings,
        'char_encodings': char_encodings,
        'lengths': lengths,
        'capital_mask': capital_mask,
    }

    if use_targets: 
        tag_encodings = [torch.Tensor(x['tag_encodings']) for x in data] # type: ignore
        tag_encodings = pad_sequence(tag_encodings, batch_first=True)
        item |= {'tag_encodings': tag_encodings}
    
    return item
    
    

def character_feature_extractor(
        sequences: list[list[str]], 
        char_vocab_path: PathLike, 
        word_vocab_path: PathLike) -> list[torch.Tensor]:
    
    word_vocab = Vocabulary.from_file(word_vocab_path)
    char_encoder = CharacterLabelEncoder(CharacterVocabulary.from_file(char_vocab_path))
    char_sequences_le = char_encoder.transform(sequences)
    max_token_len = max(len(x) for x in word_vocab.keys())
    # Create List of Label Encoded Character Tensors
    return [
        torch.tensor([word[:max_token_len] + [0] * max(max_token_len - len(word), 0) for word in sequence])
        for sequence in char_sequences_le
    ]


def case_feature_extractor(
        sequences: list[list[str]],) -> list[torch.Tensor]:
    
    # Capitalization feature extraction
    capital_mask_ = sequence_condition_mask(sequences, lambda word: word[0].isupper())
    uppercase_mask_ = sequence_condition_mask(sequences, lambda word: word.isupper())
    lowercase_mask_ = sequence_condition_mask(sequences, lambda word: word.islower())
    mixedcase_mask_ = sequence_condition_mask(sequences, lambda word: not word.islower() and not word.isupper())
    
    return [
        torch.tensor(sequence).transpose(0, -1) 
        for sequence in zip(capital_mask_, uppercase_mask_, lowercase_mask_, mixedcase_mask_)    
    ]


def word_preprocessor(
        sequences: list[list[str]], 
        word_vocab_path: PathLike) -> list[torch.Tensor]:
        # Label Encode the Word Sequences
        word_encoder = LabelEncoder(Vocabulary.from_file(word_vocab_path))
        word_sequences_le = word_encoder.transform(sequences,
                                                   default_value=word_encoder.vocab['<unk>'],
                                                   key=lambda x: x.lower())
        # Create List of Label Encoded Word Sequence Tensors
        return [torch.tensor(sequence) for sequence in word_sequences_le]


class Conll03Dataset(Dataset):

    def __init__(self, path: str | PathLike, 
                 tag_vocab_path: str | PathLike, 
                 feature_extractors: dict[str, Callable[[list[list[str]]], list[torch.Tensor]]],
                 use_targets=True) -> None:

        # Flags
        self.use_targets = use_targets

        if use_targets:
            word_sequences, tag_sequences = read_wordsequences_tagsequences(path, ' ')
            ### Tag Preprocessing:
            # Label Encode the Tag Sequences
            tag_encoder = LabelEncoder(Vocabulary.from_file(tag_vocab_path))
            tag_sequences_le = tag_encoder.transform(tag_sequences)
            # Create List of Label Encoded Tag Sequence Tensors
            self.tag_encodings:  list[torch.Tensor] = [torch.tensor(sequence) for sequence in tag_sequences_le]
        else:
            word_sequences = read_wordsequences(path, ' ')
        
        # Feature Extraction
        self.feature_names = set()
        for feature_name, feature_extractor in feature_extractors.items():
            self.feature_names.add(feature_name)
            setattr(self, feature_name, feature_extractor(word_sequences))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {feature_name: getattr(self, feature_name)[index] for feature_name in self.feature_names}
        if self.use_targets:
            item |= {'tag_encodings': self.tag_encodings[index]}
        
        return item

    def __len__(self) -> int:
        feature_name = next(iter(self.feature_names))
        return len(getattr(self, feature_name))