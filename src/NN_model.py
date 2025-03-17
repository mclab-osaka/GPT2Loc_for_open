import torch
import torch.nn as nn
import torch.nn.functional as F


class NN_estimator(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length=140,
        feature_dim=1,
        seq_len=137
    ):
        super(NN_estimator, self).__init__()
        output_size = 64

        # self.embedding = nn.Embedding(vocab_size, d_model)

        self.fc1 = nn.Linear(feature_dim*seq_len, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.1)

        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, id_input, feature_input, attention_mask=None):
        B = feature_input.size(0)
        feature_input = feature_input.view(B, -1)

        # print(feature_input.size())
        x = F.relu(self.fc1(feature_input))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        output = self.fc3(x)

        return None, output, None
