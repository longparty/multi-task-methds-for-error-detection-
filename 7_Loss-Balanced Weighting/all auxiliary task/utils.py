import configparser
import collections
import re
import torch

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_config(config_section, config_path):
    config_parser = configparser.ConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config

def read_input_files(file_paths):
    sentences = []
    line_length = None
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r",encoding='utf-8') as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    assert(len(line_parts) >= 2)
                    assert(len(line_parts) == line_length or line_length == None)
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                sentences.append(sentence)
    return sentences


def translate2id(token, token2id, unk_token):
    token = token.lower()
    token = re.sub(r'\d', '0', token)

    token_id = None
    if token in token2id:
        token_id = token2id[token]
    elif unk_token != None:
        token_id = token2id[unk_token]
    else:
        raise ValueError("Unable to handle value, no UNK token: " + str(token))
    return token_id

def translate2id_label(token, token2id, unk_token):

    token_id = None
    if token in token2id:
        token_id = token2id[token]
    elif unk_token != None:
        token_id = token2id[unk_token]
    else:
        raise ValueError("Unable to handle value, no UNK token: " + str(token))
    return token_id

def evaluate(model,input_test_data,id2label,data_test,input_test_label,main_label):
    total = 0
    predict_right = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for i in range(len(input_test_label)):
            inputs = torch.tensor(input_test_data[i], dtype=torch.long)
            tag_scores, tag_scores2, tag_scores3, tag_scores4 = model(inputs)
            predict = tag_scores.max(1, keepdim=True)[1]
            predict = [id2label[n.item()] for n in predict.view(-1)]

            for i2 in range(len(input_test_label[i])):
                total += 1
                if data_test[i][i2][-1] == predict[i2]:
                    predict_right += 1
                    if data_test[i][i2][-1] == main_label:
                        TP+=1
                    else:
                        TN+=1
                else:
                    if data_test[i][i2][-1] == main_label:
                        FN+=1
                    else:
                        FP+=1

    p = TP / (TP + FP) if (TP + FP > 0.0) else 0.0
    r = TP / (TP + FN) if (TP + FN > 0.0) else 0.0
    Accuracy = (TP+TN) / (TP+TN+FN+FP) if (TP+TN+FN+FP > 0.0) else 0.0
    f = (2.0 * p * r / (p + r)) if (p + r > 0.0) else 0.0
    f05 = ((1.0 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (p + r > 0.0) else 0.0
    # print("total:", total, "right:", predict_right)
    # print("p:", p, "r:", r)
    # print("Accuracy:", Accuracy, "f:", f)
    # print(TP,TN,FP,FN)
    # print("f05:", f05)
    return f05