# -*- coding: utf-8 -*-
import collections
import re
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

class SequenceLabeler(object):
    def __init__(self, config):
        self.config = config

        self.UNK = "<unk>"

        self.word2id = None
        self.label2id = None
        self.label2id2 = None
        self.label2id3 = None
        self.label2id4 = None

        self.word_embeddings = None


    def build_vocabs(self, data_train, data_dev, data_test, embedding_path=None):
        data_source = list(data_train)
        if self.config["vocab_include_devtest"]:
            if data_dev != None:
                data_source += data_dev
            if data_test != None:
                data_source += data_test


        word_counter = collections.Counter()
        for sentence in data_source:
            for word in sentence:
                w = word[0]
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
        self.word2id = collections.OrderedDict([(self.UNK, 0)])
        for word, count in word_counter.most_common():
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)


        label_counter = collections.Counter()
        for sentence in data_train:
            for word in sentence:
                label_counter[word[-1]] += 1
        self.label2id = collections.OrderedDict([(self.UNK, 0)])
        for label, count in label_counter.most_common():
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)

        label_counter2 = collections.Counter()
        for sentence in data_train:
            for word in sentence:
                label_counter2[word[-2]] += 1
        self.label2id2 = collections.OrderedDict([(self.UNK, 0)])
        for label, count in label_counter2.most_common():
            if label not in self.label2id2:
                self.label2id2[label] = len(self.label2id2)

        label_counter3 = collections.Counter()
        for sentence in data_train:
            for word in sentence:
                label_counter3[word[-3]] += 1
        self.label2id3 = collections.OrderedDict([(self.UNK, 0)])
        for label, count in label_counter3.most_common():
            if label not in self.label2id3:
                self.label2id3[label] = len(self.label2id3)

        label_counter4 = collections.Counter()
        for sentence in data_train:
            for word in sentence:
                label_counter4[word[-4]] += 1
        self.label2id4 = collections.OrderedDict([(self.UNK, 0)])
        for label, count in label_counter4.most_common():
            if label not in self.label2id4:
                self.label2id4[label] = len(self.label2id4)

        if embedding_path != None and self.config["vocab_only_embedded"] == True:
            self.embedding_vocab = set([self.UNK])
            fp=open(embedding_path,'r',encoding='UTF-8')
            lines = fp.readlines()
            for line in lines:
                line_parts = line.strip().split()
                if len(line_parts) <= 2:
                    continue
                w = line_parts[0]
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                self.embedding_vocab.add(w)
            word2id_revised = collections.OrderedDict()
            for word in self.word2id:
                if word in self.embedding_vocab and word not in word2id_revised:
                    word2id_revised[word] = len(word2id_revised)
            self.word2id = word2id_revised

        print("n_words: " + str(len(self.word2id)))
        print("n_labels: " + str(len(self.label2id)))
        print("n_labels: " + str(len(self.label2id2)))
        print("n_labels: " + str(len(self.label2id3)))
        print("n_labels: " + str(len(self.label2id4)))


        self.word_embeddings = np.zeros((len(self.word2id),self.config["word_embedding_size"]))


    def preload_word_embeddings(self, embedding_path):
        loaded_embeddings = set()
        embedding_matrix = np.zeros((len(self.word2id),self.config["word_embedding_size"]))
        fp = open(embedding_path, 'r', encoding='UTF-8')
        lines = fp.readlines()
        for line in lines:
            line_parts = line.strip().split()
            if len(line_parts) <= 2:
                continue
            w = line_parts[0]
            if self.config["lowercase"] == True:
                w = w.lower()
            if self.config["replace_digits"] == True:
                w = re.sub(r'\d', '0', w)
            if w in self.word2id and w not in loaded_embeddings:
                word_id = self.word2id[w]
                embedding = np.array(line_parts[1:])
                embedding_matrix[word_id] = embedding
                loaded_embeddings.add(w)
        self.word_embeddings = embedding_matrix
        print("n_preloaded_embeddings: " + str(len(loaded_embeddings)))

    def load(filename):
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            labeler = SequenceLabeler(dump["config"])
            labeler.UNK = dump["UNK"]
            labeler.word2id = dump["word2id"]
            labeler.label2id = dump["label2id"]
            labeler.label2id2 = dump["label2id2"]
            labeler.label2id3 = dump["label2id3"]
            labeler.label2id4 = dump["label2id4"]
            labeler.word_embeddings = dump["word_embeddings"]

            return labeler

    def save(self, filename):
        dump = {}
        dump["config"] = self.config
        dump["UNK"] = self.UNK
        dump["word2id"] = self.word2id
        dump["label2id"] = self.label2id
        dump["label2id2"] = self.label2id2
        dump["label2id3"] = self.label2id3
        dump["label2id4"] = self.label2id4
        dump["word_embeddings"] = self.word_embeddings

        with open(filename, 'wb') as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)