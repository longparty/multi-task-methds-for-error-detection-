In this folder, there are two models for Loss-Balanced Weighting. The first one use one auxiliary task at one time, another one use all auxiliary tasks at the same time.

How to run the code:
step 1 : unzip the glove.6B.300d.rar in 1_data/glove and copy the file into the embeddings/glove
step 2 : Run with: python experiment.py

If you want to change the hyper parameter a:
change the code in line 137，138 of experiment.py in folder one auxiliary task
OR
change the code in line 177-180 of experiment.py in folder all auxiliary task