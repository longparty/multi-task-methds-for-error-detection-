In this folder, there are four models for multitask learning. The first two models use one auxiliary task at one time, another two models use all auxiliary tasks at the same time.

How to run the code:
step 1 : unzip the glove.6B.300d.rar in 1_data/glove and copy the file into the embeddings/glove
step 2 : if you want to use MMOE model, rename the file model_MMOE.py to model.py; if you want to use MOE model, rename the file model_MOE.py to model.py
step 3 : Run with: python experiment.py
