# AutomatedExceptionHandling

## Requirements
* Python 3.6
* pandas 0.20.3
* pytorch 1.3.1
* tqdm 4.30.0
* scikit-learn 0.19.1
* javalang 0.11.0

## Task1: Try block localization
1. Unzip the data file "task1/data/data.tar.gz";
2. Run `python utils.py` to preprocess the data;
3. Run `python train.py train` to train a model and save the checkpoints;
4. Run `python train.py test N` to test the model by specifying the epoch N according to the results of step 3.

## Task2: Catch block generation
1. Run `python prepare.py` to prepare the data;
2. Run `sh preprocess.sh` to preprocess the data;
3. Run `sh train.sh` to train a model;
4. Run `sh infer.sh` to generate handling code and the output will be saved in "testout/multi_slicing.out";
5. Run `perl multi-bleu.perl data/multi_slicing/tgt-test.txt < testout/multi_slicing.out` to get the BLEU score;
6. Run `python evaluate.py data/multi_slicing/tgt-test.txt testout/multi_slicing.out` to get the Accuracy.
