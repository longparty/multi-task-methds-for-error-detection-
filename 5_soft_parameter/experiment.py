from utils import *
from model import BiLSTM
from labeler import SequenceLabeler
import time


import torch
import torch.nn as nn
import torch.optim as optim

def run_experiment(config_path):
    config = parse_config("config", config_path)


    if config["path_train"] != None and len(config["path_train"]) > 0:
        data_train = read_input_files(config["path_train"])
    if config["path_dev"] != None and len(config["path_dev"]) > 0:
        data_dev = read_input_files(config["path_dev"])
    if config["path_test"] != None and len(config["path_test"]) > 0:
        data_test = []
        for path_test in config["path_test"].strip().split(":"):
            data_test += read_input_files(path_test)


    if config["mode"]=="Train":
        labeler = SequenceLabeler(config)
        labeler.build_vocabs(data_train, data_dev, data_test, config["preload_vectors"])
        labeler.preload_word_embeddings(config["preload_vectors"])
    else:
        labeler = SequenceLabeler.load(config["load_labeler_path"])
    word2id = labeler.word2id
    label2id = labeler.label2id



    id2label = collections.OrderedDict()
    for label in label2id:
        id2label[label2id[label]] = label
    print(label2id)
    print(id2label)


    word_embeddings=labeler.word_embeddings
    word_embeddings = torch.FloatTensor(word_embeddings)


    input_train_data = []
    for i in data_train:
        sentence = []
        for i2 in i:
            sentence.append(translate2id(i2[0], word2id, "<unk>"))
        input_train_data.append(sentence)


    input_train_label = []
    for i in data_train:
        sentence = []
        for i2 in i:
            sentence.append(translate2id_label(i2[-1], label2id, "<unk>"))
        input_train_label.append(sentence)


    input_test_data = []
    for i in data_test:
        sentence = []
        for i2 in i:
            sentence.append(translate2id(i2[0], word2id, "<unk>"))
        input_test_data.append(sentence)

    input_test_label = []
    for i in data_test:
        sentence = []
        for i2 in i:
            sentence.append(translate2id_label(i2[-1], label2id, "<unk>"))
        input_test_label.append(sentence)


    input_dev_data = []
    for i in data_dev:
        sentence = []
        for i2 in i:
            sentence.append(translate2id(i2[0], word2id, "<unk>"))
        input_dev_data.append(sentence)

    input_dev_label = []
    for i in data_dev:
        sentence = []
        for i2 in i:
            sentence.append(translate2id_label(i2[-1], label2id, "<unk>"))
        input_dev_label.append(sentence)

    start_time = time.time()

    model2 = torch.load(open('model/POS.model', 'rb'))
    regularize_layer = ['hidden2tag.weight']
    regularize_param = {}
    for name, param in model2.named_parameters():
        if name in regularize_layer:
            param = param.view(-1)
            regularize_param[name] = param

    if config["mode"]=="Train":
        model = BiLSTM(config, word_embeddings, config["word_embedding_size"], config["hidden_layer_size"], len(word2id), len(label2id))
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1)
        tolerance=0
        max_epoch=0
        max_f05=0
        for epoch in range(config["max_train_epoch"]):
            loss_total = 0
            for i in range(len(input_train_label)):

                model.zero_grad()

                sentence_in = torch.tensor(input_train_data[i], dtype=torch.long)
                targets = torch.tensor(input_train_label[i], dtype=torch.long)

                tag_scores = model(sentence_in)

                loss1 = loss_function(tag_scores, targets)
                regularization_loss = 0

                for name, param in model.named_parameters():
                    if name in regularize_layer:
                        param = param.view(-1)
                        for i2 in range(len(param)):
                            regularization_loss += abs(param[i2] - regularize_param[name][i2])
                            # regularization_loss += (param[i2] - regularize_param[name][i2])**2
                # regularization_loss = regularization_loss**0.5
                loss = loss1 + 0.01 * regularization_loss
                loss.backward()
                optimizer.step()
                loss_total += loss

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss_total))
            f05 = evaluate(model, input_dev_data, id2label, data_dev, input_dev_label, config['main_label'])
            print(f05)
            if f05 > max_f05:
                max_f05=f05
                max_epoch=epoch+1
                tolerance=0
            else:
                tolerance+=1
            if tolerance>=5:
                break
        torch.save(model,open(config['save_model_path'],'wb'))
        print('max_epoch:',max_epoch,'max_f0.5:', max_f05)
        labeler.save(config['save_labeler_path'])
        end_time = time.time()
        seconds = end_time - start_time
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print('time_spend:', "%02d:%02d:%02d" % (h, m, s))
    if config["mode"] == "Test":
        print("test mode")
        model=torch.load(open(config['load_model_path'],'rb'))

    print("test_result")
    print(evaluate(model,input_test_data,id2label,data_test,input_test_label,config['main_label']))

















































if __name__ == "__main__":
    run_experiment('config/fce.conf')



