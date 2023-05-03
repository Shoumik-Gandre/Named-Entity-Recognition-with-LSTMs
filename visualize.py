from argparse import ArgumentParser, Namespace
from src.visualization.blstm1 import visualize_model as task1_visualize_model
from src.visualization.blstm2 import visualize_model as task2_visualize_model
from src.visualization.blstm3 import visualize_model as task3_visualize_model




def main(args: Namespace):
    assert args.model_path
    match args.task:
        
        case 1:
            task1_visualize_model(model_path=args.model_path)

        case 2:
            task2_visualize_model(model_path=args.model_path)


        case 3:
            task3_visualize_model(model_path=args.model_path)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--task", "-t", choices=[1, 2, 3], type=int)
    args = parser.parse_args()
    main(args)
