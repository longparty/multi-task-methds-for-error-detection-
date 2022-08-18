This folder contains all the data used in the project. There are three folders in total. The data are in standard CoNLL-type tab-separated format. One word per line, separate column for token and label, empty line between sentences.

FCE folder:
This folder contains all FCE data, including training set, development set and test set. In this folder, files end with m2 represent unprocessed files. The remaining files end with txt represent the files we use in training and testing process. Next, we will explain them respectively.

fce_train.txt : training set for error detection
fce_test.txt : test set for error detection
fce_dev.txt : development set for error detection
fce_train_error.txt : training set for error detection with error type as auxiliary task
fce_train_NER.txt : training set for error detection with NER as auxiliary task
fce_train_POS.txt : training set for error detection with POS as auxiliary task
fce_train_all.txt : training set for error detection with all three labels above as auxiliary task

CoNLL_2014 folder:
This folder contains CoNLL_2014 test data set annotated by two different annotators. Files end with m2 represent unprocessed files, files end with txt represent the files we used in testing process.

glove folder:
This folder contains 300 dimensional word vector data by glove. 
glove.6B.300d.rar : This file is needed in the project. What you need to do is unzip it and copy the file end with txt to the embeddings\glove folder of the project.