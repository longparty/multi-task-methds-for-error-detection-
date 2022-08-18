This folder contains all the data used in the project.  

process_1.py ：This file is used to obtain training samples. The specific operation is to cut out the first X lines of the target file. The purpose of this operation is to get a smaller training set to test model's performance when the training data is insufficient.

process_2.py ：This file is used to convert the original data (end with M2) into standard CoNLL-type tab-separated format for error detection task.

process_3.py : This file is used to convert the original data (end with M2) into standard CoNLL-type tab-separated format for error detection task, with error type as auxiliary task.

process_4.py : This file uses nltk library to annotate CoNLL format files, generate POS tags, and place them in the second to last column

process_5.py : This file uses nltk library to annotate CoNLL format files, generate NER tags, and place them in the second to last column

process_6.py : The purpose of this file is to change the label placed in the last column. Since the main task label is placed in the last column, we need this file to generate the training dataset of different auxiliary tasks to train the auxiliary model when using the soft parameter sharing method.

process_7.py : This file puts different auxiliary labels in the same file




