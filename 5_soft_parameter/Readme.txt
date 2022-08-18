In this folder, there is the model for soft paramter sharing. In order to facilitate the operation, we have placed an auxiliary model in the model folder in advance which is obtained by training the model in 3_singletask_basic with POS task.

How to run the code:
step 1 : unzip the glove.6B.300d.rar in 1_data/glove and copy the file into the embeddings/glove
step 2 : Run with: python experiment.py

If you want to change the hyper parameter a:
change the code in line 132 of experiment.py

If you want to change the regularization method:
change the code in line 129-131 of experiment.py

If you want to change the auxiliary model's path:
change the code in line 94 of experiment.py

If you want to change the parameter to be shared:
change the code in line 95-97 of experiment.py

