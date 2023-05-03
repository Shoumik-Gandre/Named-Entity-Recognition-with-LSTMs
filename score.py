from argparse import ArgumentParser, Namespace
from os import PathLike
import subprocess
from src.iodata import read_wordsequences_tagsequences, write_prediction_file
from src.settings import PERL_SCRIPT


def write_scoreable_file(
        true_path: PathLike, 
        pred_path: PathLike, 
        out_path: PathLike,
        word_col: int=1, 
        tag_col: int=2) -> None:
    X, y_true = read_wordsequences_tagsequences(true_path, ' ', word_col, tag_col)
    _, y_pred = read_wordsequences_tagsequences(pred_path, ' ')
    write_prediction_file(out_path, X, y_true, y_pred, ' ')

    p1 = subprocess.run(f"perl \"{PERL_SCRIPT}\" < \"{out_path}\"", 
                        capture_output=True, shell=True, text=True)

    print(p1.stdout)


def main(args: Namespace):
    assert args.true_path
    assert args.pred_path
    assert args.out_path
    match args.mode:
        case "dev":
            write_scoreable_file(args.true_path, args.pred_path, args.out_path)
        case "test":
            write_scoreable_file(args.true_path, args.pred_path, args.out_path, 0, 3)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--true-path", "-t", type=str, required=True)
    parser.add_argument("--pred-path", "-p", type=str, required=True)
    parser.add_argument("--out-path", "-o", type=str, required=True)
    parser.add_argument("--mode", "-m", choices=["dev", "test"], required=True)
    args = parser.parse_args()
    main(args)
