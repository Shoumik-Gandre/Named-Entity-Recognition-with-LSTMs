from os import PathLike
import re
import json
import numpy as np


def read_wordsequences_tagsequences(
        path: str | PathLike, 
        delimiter: str,
        word_column: int=1,
        tag_column: int=2) -> tuple[list[list[str]], list[list[str]]]:
    """This is used to read from the train and dev set"""
    with open(path) as f:
        data = f.read()
    tagged_sentences = re.split(r'\n\n', data.strip('\n'))

    word_sequences = [
        [
            wordrecord.split(delimiter)[word_column] 
            for wordrecord in sentence.split('\n')
        ] 
        for sentence in tagged_sentences
    ]

    tag_sequences = [
        [
            tag_word.split(delimiter)[tag_column] 
            for tag_word in sentence.split('\n')
        ] 
        for sentence in tagged_sentences
    ]
    
    return word_sequences, tag_sequences


def read_wordsequences(path: str | PathLike, delimiter: str) -> list[list[str]]:
    """This is used to read from the train and dev set"""
    with open(path) as f:
        data = f.read()
    tagged_sentences = re.split(r'\n\n', data.strip('\n'))

    word_sequences = [
        [
            wordrecord.split(delimiter)[1] 
            for wordrecord in sentence.split('\n')
        ] 
        for sentence in tagged_sentences
    ]
    
    return word_sequences


def write_vocabulary(path: str | PathLike, frequency: list[tuple[str, int]]) -> None:

    result = ""
    for index, (word, count) in enumerate(frequency):
        result += f"{word}\t{index}\t{count}\n"

    with open(path, 'w') as f:
        f.write(result)


def write_wordsequences_tagsequences(path: str | PathLike, wordsequences: list[list[str]], tagsequences: list[list[str]], delimiter: str):
    
    result = ""
    for wordseq, tagseq in zip(wordsequences, tagsequences):
        for i, (word, tag) in enumerate(zip(wordseq, tagseq)):
            result += f"{i+1}{delimiter}{word}{delimiter}{tag}\n"
        result += "\n"
    result = result[:-1]
    
    with open(path, 'w') as f:
        f.write(result)


def write_prediction_file(path: str | PathLike, wordsequences: list[list[str]], tagsequences: list[list[str]], predsequences: list[list[str]], delimiter: str):
    result = ""
    for wordseq, tagseq, predseq in zip(wordsequences, tagsequences, predsequences):
        for i, (word, tag, pred) in enumerate(zip(wordseq, tagseq, predseq)):
            result += f"{i+1}{delimiter}{word}{delimiter}{tag}{delimiter}{pred}\n"
        result += "\n"
    result = result[:-1]
    
    with open(path, 'w') as f:
        f.write(result)


def read_glove_words(path: str | PathLike) -> dict[str, int]:
    index_to_word = np.loadtxt(path, usecols=0, comments=None, encoding='utf-8', dtype='str')
    index_to_word = np.insert(index_to_word, 0, '<pad>')
    index_to_word = np.insert(index_to_word, 1, '<unk>')
    word_to_index = {x: index for index, x in enumerate(index_to_word)}
    
    return word_to_index


def read_glove_embeddings(path: str | PathLike) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Returns the means to use glove embeddings:
    Return:
    1. A dictionary that maps words to its index
    2. An array that maps index to words
    3. A matrix that holds embeddings coresponsding to the word at the index == row in matrix
    """
    index_to_word = np.loadtxt(path, usecols=0, comments=None, encoding='utf-8', dtype='str')
    index_to_word = np.insert(index_to_word, 0, '<pad>')
    index_to_word = np.insert(index_to_word, 1, '<unk>')
    word_to_index = {x: index for index, x in enumerate(index_to_word)}

    embeddings = np.loadtxt(path, usecols=range(1, 100+1), comments=None, encoding='utf-8')
    #embedding for '<pad>' token.
    pad_embedding = np.zeros((1, embeddings.shape[1]))
    #embedding for '<unk>' token.
    unk_embedding = np.mean(embeddings, axis=0, keepdims=True)
    embeddings = np.vstack((pad_embedding, unk_embedding, embeddings))

    return word_to_index, index_to_word, embeddings
