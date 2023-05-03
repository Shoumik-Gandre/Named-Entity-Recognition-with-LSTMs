from argparse import ArgumentParser, Namespace
from src.task1.train import train_model as task1_train_model
from src.task2.train import train_model as task2_train_model
from src.task3.train import train_model as task3_train_model
from src.task1.predict import generate_prediction_file as task1_generate_predictions
from src.task2.predict import generate_prediction_file as task2_generate_predictions
from src.task3.predict import generate_prediction_file as task3_generate_predictions
from src.task1.vocab import generate_vocabulary as task1_generate_vocabulary
from src.task2.vocab import generate_vocabulary as task2_generate_vocabulary
from src.task3.vocab import generate_vocabulary as task3_generate_vocabulary



def main(args: Namespace):
    match args.task:

        case 1:
            match args.mode:

                case "vocab":
                    assert args.input_path
                    task1_generate_vocabulary(args.input_path)

                case "train":
                    assert args.input_path
                    assert args.output_path
                    assert args.model_path
                    assert args.dev_path
                    task1_train_model(args.input_path, args.dev_path, args.output_path, args.model_path)

                case "predict":
                    assert args.input_path
                    assert args.output_path
                    assert args.model_path
                    task1_generate_predictions(args.input_path, args.output_path, args.model_path)

        case 2:
            match args.mode:
                
                case "vocab":
                    assert args.input_path
                    task2_generate_vocabulary(args.input_path)

                case "train":
                    assert args.input_path
                    assert args.output_path
                    assert args.model_path
                    assert args.dev_path
                    task2_train_model(args.input_path, args.dev_path, args.output_path, args.model_path)

                case "predict":
                    assert args.input_path
                    assert args.output_path
                    assert args.model_path
                    task2_generate_predictions(args.input_path, args.output_path, args.model_path)

        case 3:
            match args.mode:

                case "vocab":
                    assert args.input_path
                    task3_generate_vocabulary(args.input_path)

                case "train":
                    assert args.input_path
                    assert args.output_path
                    assert args.model_path
                    assert args.dev_path
                    task3_train_model(args.input_path, args.dev_path, args.output_path, args.model_path)

                case "predict":
                    assert args.input_path
                    assert args.output_path
                    assert args.model_path
                    task3_generate_predictions(args.input_path, args.output_path, args.model_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", "-t", type=int,
                        choices=[1, 2, 3], required=True)
    parser.add_argument("--mode", "-m", type=str,
                        choices=['vocab', 'train', 'predict'], required=True)
    parser.add_argument("--input-path", "-i", type=str)
    parser.add_argument("--output-path", "-o", type=str)
    parser.add_argument("--model-path", "-p", type=str)
    parser.add_argument("--vocab-path", "-v", type=str)
    parser.add_argument("--dev-path", "-d", type=str)

    args = parser.parse_args()
    main(args)
