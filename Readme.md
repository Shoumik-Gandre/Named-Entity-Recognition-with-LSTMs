# Named-Entity-Recognition-with-LSTMs
This was part of my coursework involving Named Entity recognition using Bidirectional LSTMs

### General Instructions:

You need to use python 3.10.4 or higher.
Put your 3 data files (train, test, dev) in the data/ folder or use the pre-existing ones.
The out/ folder contains all the files that you need for grading.
The src folder contains all the code.
The models folder contains the vocabularies and the model checkpoints.
The main.py file in this folder is your interface to my software.

All Commands at your disposal:

To generate any prediction files (test or dev):
```
python main.py -t <task number> -m predict -i <path to data> -o <output prediction path> -p <model checkpoint path>
```

To simulate the training process:
```
python main.py -t <task number> -m train -i <path to train data> -d <path to dev data> -o <output prediction path> -p <model checkpoint path>
```

To generate vocabulary files
```
python .\main.py -m vocab -t <task number> -i <path to train data>
```


Examples:

#### Task 1
Generate dev predictions:
```
python main.py -t 1 -m predict -i "data\dev" -o "out\dev1.out" -p "models\1\blstm1.pt"
```

Generate test predictions:
```
python main.py -t 1 -m predict -i "data\test" -o "out\test1.out" -p "models\1\blstm1.pt"
```

Train: WARNING, this may replace pretrained model
```
python main.py -t 1 -m train -i "data\train" -d "data\dev" -o "out\dev1.out" -p "models\1\blstm1.pt"
```

Generate vocab:
```
python .\main.py -m vocab -t 1 -i "data\train"
```


#### Task 2
Generate dev predictions:
```
python main.py -t 2 -m predict -i "data\dev" -o "out\dev2.out" -p "models\2\blstm2.pt"
```

Generate test predictions:
```
python main.py -t 2 -m predict -i "data\test" -o "out\test2.out" -p "models\2\blstm2.pt"
```

Generate vocab:
```
python .\main.py -m vocab -t 2 -i "data\train"
```

Train: WARNING, this may replace pretrained model
```
python main.py -t 2 -m train -i "data\train" -d "data\dev" -o "out\dev2.out" -p "models\2\blstm2.pt"
```


#### Task 3
Generate dev predictions:
```
python main.py -t 3 -m predict -i "data\dev" -o "out\dev3.out" -p "models\3\blstm3.pt"
```

Generate test predictions:
```
python main.py -t 3 -m predict -i "data\test" -o "out\pred.out" -p "models\3\blstm3.pt"
```

Train: WARNING, this may replace pretrained model
```
python main.py -t 3 -m train -i "data\train" -d "data\dev" -o "out\dev3.out" -p "models\3\blstm3.pt"
```



NOTE:
I output model checkpoints into models folder!
To convert them into models use ctm.py

Command at your disposal:
```
python ctm.py -i <model checkpoint path> -o <model path> -m <task number>
```

Examples:
For blstm1:
```
python ctm.py -i "models\1\blstm1.pt" -o "out\blstm1.pt" -m 1
```
for blstm2:
```
python ctm.py -i "models\2\blstm2.pt" -o "out\blstm2.pt" -m 2
```
for blstm3:
```
python ctm.py -i "models\3\blstm3.pt" -o "out\blstm3.pt" -m 3
```


I have provided a requirements.txt
Make use of it if needed.
