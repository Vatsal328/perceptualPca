This file contains instructions to run the attached files.

## Problem 1
### Train file (`train.py`)
It takes `train.txt` as an argument and can be executed using the command
 ```bash 
 python train.py train.txt
 ```
It creates `weights.txt` and confirms the same by giving output as "Training over and weights are saved".

### Test file(`test.py`)
It takes `test.txt` as an argument and can be executed using the command
```bash
python test.py test.txt
```
It creates `predictions.txt`(used to check accuracy, can be ignored by the TA) and outputs the predicted labels in a comma separated form.

### Colab file
The link of this file is provided in the report and this file is not provided along with the above files.

It consists of two files, `main.py` which genearates the files like `data.txt`, `train.txt`, `test.txt` and `test_labels.txt`(required to compute accuracy) and `calAcc.py` which computes the accuracy using `test_labels.txt` and `predictions.txt`.

Please note that this file may not run directly as it will require other files like `predictions.txt` which will not be present in Colab. To run this file, download the `main.py` and `calAcc.py` files from the links provided in the report and run them separately after running `train.py` and `test.py`. 

## Problem 2
### Main file(`problem2.py`)
This file consists of the whole code of problem 2 and performs all the required tasks.

Please note that when run this file may not show the output in a perfect manner, prefer to run it using the colab file link provided in the report.
