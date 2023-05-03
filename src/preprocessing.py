from __future__ import annotations
from typing import Any, Callable, TypeVar
from collections import defaultdict
from typing import TypeVar
import torch
from dataclasses import dataclass

from .vocabulary import Vocabulary, CharacterVocabulary


T = TypeVar('T')
def count_items(sequences: list[list[T]], transform: Callable[[T], T]= lambda x: x) -> defaultdict[T, int]:
    frequency = defaultdict(lambda: 0)

    for sent in sequences:
        for item in sent:
            frequency[transform(item)] += 1

    return frequency


def calc_tag_weights(y: torch.Tensor) -> torch.Tensor:
    items, freqs = y.unique(return_counts=True)
    freqs = freqs[items != -1]

    prob = freqs / torch.sum(freqs)
    tag_weights = 1. / prob
    return tag_weights
    

@dataclass
class LabelEncoder:
    vocab: Vocabulary

    def __post_init__(self):
        self.inverse_vocab: dict[int, str] = self.vocab.get_inverse()

    def transform(self, sequences: list[list[str]], default_value=1, key: Callable[[str], str]=lambda x: x) -> list[list[int]]:
        return [
            [self.vocab.get(key(word), default_value) for word in sequence] 
            for sequence in sequences
        ]
    
    def inverse_transform(self, sequences: list[list[int]], default_value='O') -> list[list[str]]:
        return [
            [self.inverse_vocab.get(num, default_value) for num in sequence] 
            for sequence in sequences
        ]
    
@dataclass
class CharacterLabelEncoder:
    vocab: CharacterVocabulary

    def __post_init__(self):
        self.inverse_vocab: dict[int, str] = self.vocab.get_inverse()

    def transform(self, sequences: list[list[str]], default_value=1, key: Callable[[str], str]=lambda x: x) -> list[list[list[int]]]:
        return [
            [
                [self.vocab.get(key(ch), default_value) for ch in word]
                for word in sequence
            ]
            for sequence in sequences
        ]
    

def sequence_condition_mask(sequences: list[list[str]], condition: Callable[[str], bool]) -> list[list[int]]:
    return [
        [int(condition(word)) for word in sequence] 
        for sequence in sequences
    ]

    
def capital_mask(sequences: list[list[str]]) -> list[list[int]]:
    return [
        [int(word[0].isupper()) for word in sequence] 
        for sequence in sequences
    ]

def uppercase_mask(sequences: list[list[str]]) -> list[list[int]]:
    return [
        [int(word.isupper()) for word in sequence] 
        for sequence in sequences
    ]

def lowercase_mask(sequences: list[list[str]]) -> list[list[int]]:
    return [
        [int(word.islower()) for word in sequence] 
        for sequence in sequences
    ]

def mixedcase_mask(sequences: list[list[str]]) -> list[list[int]]:
    return [
        [int(not word.islower() and not word.isupper()) for word in sequence] 
        for sequence in sequences
    ]


