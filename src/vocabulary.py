from __future__ import annotations
from collections import UserDict
from os import PathLike
import json
from pathlib import Path


class Vocabulary(UserDict[str, int]):
    """A dictionary of keys of type str and values of type int"""

    def get_inverse(self) -> dict[int, str]:
        return {value: key for key, value in self.items()}

    @staticmethod
    def from_sequences(sequences: list[list[str]], start_index=1) -> Vocabulary:
        vocab = Vocabulary()
        index = start_index

        for sent in sequences:
            for item in sent:
                if item not in vocab:
                    vocab[item] = index
                    index += 1

        return vocab
    
    def to_dict(self) -> dict[str, int]:
        return self.data
   
    @staticmethod
    def from_dict(data: dict[str, int]) -> Vocabulary:
        return Vocabulary(data)
    
    def to_file(self, path: str | PathLike) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @staticmethod
    def from_file(path: str | PathLike) -> Vocabulary:
        if Path(path).exists():
            with open(path) as f:
                word_vocab = Vocabulary.from_dict(json.load(f))
            return word_vocab
        else:
            raise FileNotFoundError()
        

class CharacterVocabulary(UserDict[str, int]):
    """A dictionary of keys of type str and values of type int"""

    def get_inverse(self) -> dict[int, str]:
        return {value: key for key, value in self.items()}

    @staticmethod
    def from_sequences(sequences: list[list[str]], start_index=1) -> CharacterVocabulary:
        vocab = CharacterVocabulary()
        index = start_index

        for sequence in sequences:
            for token in sequence:
                for character in token:
                    if character not in vocab.keys():
                        vocab[character] = index
                        index +=1

        return vocab
    
    def to_dict(self) -> dict[str, int]:
        return self.data
   
    @staticmethod
    def from_dict(data: dict[str, int]) -> CharacterVocabulary:
        return CharacterVocabulary(data)
    
    def to_file(self, path: str | PathLike) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @staticmethod
    def from_file(path: str | PathLike) -> CharacterVocabulary:
        if Path(path).exists():
            with open(path) as f:
                word_vocab = CharacterVocabulary.from_dict(json.load(f))
            return word_vocab
        else:
            raise FileNotFoundError()
