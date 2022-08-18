import torch
import torch.nn as nn
import torch.nn.functional as F



class BiLSTM(nn.Module):

    def __init__(self, config, word_embeddings, embedding_dim, hidden_dim, vocab_size, tagset_size,tagset_size2):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings)
        self.birnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.birnn2 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.birnn3 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)


        self.gate = nn.Linear(embedding_dim, 3)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.hidden2tag2 = nn.Linear(hidden_dim * 2, tagset_size2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

        lstm_out1, (_, _) = self.birnn(embeds.view(len(sentence), 1, -1))
        lstm_out2, (_, _) = self.birnn2(embeds.view(len(sentence), 1, -1))
        lstm_out3, (_, _) = self.birnn3(embeds.view(len(sentence), 1, -1))


        lstm_out1 = torch.tanh(lstm_out1)
        lstm_out2 = torch.tanh(lstm_out2)
        lstm_out3 = torch.tanh(lstm_out3)


        gate_out = self.gate(embeds)
        gate_out = torch.FloatTensor(gate_out)
        gate_wight = gate_out.mean(0)
        Softmax = nn.Softmax(dim=0)
        gate_wight = Softmax(gate_wight)

        lstm_out = lstm_out1 * gate_wight[0] + lstm_out2 * gate_wight[1] + lstm_out3 * gate_wight[2]

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        tag_space2 = self.hidden2tag2(lstm_out.view(len(sentence), -1))
        tag_scores2 = F.log_softmax(tag_space2, dim=1)
        return tag_scores,tag_scores2

