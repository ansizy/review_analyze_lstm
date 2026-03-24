import torch
from torch import nn

from src import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()

        # embedding lstm linear
        # padding_idx 填充字符的index
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM, padding_idx=padding_index)

        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)

        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1)

    def forward(self, x):
        # x shape [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded shape [batch_size, seq_len, embedding_dim]
        lstm_out, (_, _) = self.lstm(embedded)
        # lstm_out shape [batch_size, seq_len, hidden_size]
        # 取最后一个非 pad 的隐藏状态
        # last_hidden_state = lstm_out[:, -1, :]
        batch_indexes = torch.arange(0, lstm_out.shape[0])
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = lstm_out[batch_indexes, lengths - 1]

        output = self.linear(last_hidden_state).squeeze(-1)
        # output shape [batch_size]

        return output
