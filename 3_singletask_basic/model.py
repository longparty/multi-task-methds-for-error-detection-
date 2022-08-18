import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):

    def __init__(self, config, word_embeddings, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings)
        self.birnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, (_,_) = self.birnn(embeds.view(len(sentence), 1, -1))

        lstm_out = torch.tanh(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

