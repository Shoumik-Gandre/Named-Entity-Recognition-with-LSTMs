from argparse import ArgumentParser, Namespace
from os import PathLike
from src.task1.model import BLSTM as Model1
from src.vocabulary import Vocabulary as Vocabulary1
from src.task1.settings import WORD_VOCAB_PATH as task1_WORD_VOCAB_PATH

from src.task2.model import Blstm as Model2
from src.task3.model import BlstmCnn as Model3

import torch
import numpy as np


def main(args: Namespace):
    assert args.input_path
    assert args.output_path
    assert args.model
    match args.model:
        
        case 1:
            model = Model1(len(Vocabulary1.from_file(task1_WORD_VOCAB_PATH))+1)
            model.load_state_dict(torch.load(args.input_path)['model_state_dict'])
            model.eval()

            torch.save(model, args.output_path)

        case 2:
            model = Model2(np.zeros((400_002, 100)))
            model.load_state_dict(torch.load(args.input_path)['model_state_dict'])
            model.eval()

            torch.save(model, args.output_path)

        case 3:
            model = Model3(np.zeros((400_002, 100)), num_char_embeddings=85)
            model.load_state_dict(torch.load(args.input_path)['model_state_dict'])
            model.eval()

            torch.save(model, args.output_path)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-path", "-i", type=str, required=True)
    parser.add_argument("--output-path", "-o", type=str, required=True)
    parser.add_argument("--model", "-m", choices=[1, 2, 3], type=int)
    args = parser.parse_args()
    main(args)
