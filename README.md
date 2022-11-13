# Learning to Handle Exceptions
This repository includes the source code and data for our paper "Learning to Handle Exceptions" which has been published in ASE'20. 
Exception handling is an important built-in feature of many modern programming languages such as Java. It allows developers to deal with abnormal or unexpected conditions that may occur at runtime in advance by using try-catch blocks. Missing or improper implementation of exception handling can cause catastrophic consequences such as system crash. However, previous studies reveal that developers are unwilling or feel it hard to adopt exception handling mechanism, and tend to ignore it until a system failure forces them to do so. To help developers with exception handling, existing work produces recommendations such as code examples and exception types, which still requires developers to localize the try blocks and modify the catch block code to fit the context. In this paper, we propose a novel neural approach to automated exception handling, which can predict locations of try blocks and automatically generate the complete catch blocks. We collect a large number of Java methods from GitHub and conduct experiments to evaluate our approach. The evaluation results, including quantitative measurement and human evaluation, show that our approach is highly effective and outperforms all baselines. Our work makes one step further towards automated exception handling.
## Requirements
* Python 3.6
* pandas 0.20.3
* pytorch 1.3.1
* tqdm 4.30.0
* scikit-learn 0.19.1
* javalang 0.11.0

## Dataset
[Updated] The raw data and scripts corresponding to the two tasks were added. 
Please extract it from the raw_data.tar.gz and run the scripts accordingly to get the experimental data.


## Task1: Try block localization
1. Unzip the data file "task1_data.tar.gz" into task1/data;
2. Run `python utils.py` to preprocess the data;
3. Run `python train.py train` to train a model and save the checkpoints;
4. Run `python train.py test N` to test the model by specifying the epoch N according to the results of step 3.

## Task2: Catch block generation
1. Unzip the data file "task2_data.tar.gz" into task2/data;
2. Run `python prepare.py` to prepare the data;
3. Run `sh preprocess.sh` to preprocess the data;
4. Run `sh train.sh` to train a model;
5. Run `sh infer.sh` to generate handling code and the output will be saved in "testout/multi_slicing.out";
6. Run `perl multi-bleu.perl data/multi_slicing/tgt-test.txt < testout/multi_slicing.out` to get the BLEU score;
7. Run `python evaluate.py data/multi_slicing/tgt-test.txt testout/multi_slicing.out` to get the Accuracy.
