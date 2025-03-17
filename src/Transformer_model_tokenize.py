import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length=1000,
        feature_dim=1,
    ):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.feature_fc = nn.Linear(feature_dim, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.transformer_encoder = self.transformer.encoder

        self.fc_out = nn.Linear(d_model, 2)  # Output x, y
        self.projection_head = nn.Linear(d_model, 64)  # Output x, y

    def forward(self, id_input, feature_input, attention_mask=None):
        # Embedding and positional encoding
        embedded = self.embedding(id_input)

        # Process feature input
        # print(feature_input.size())
        feature_input_t = feature_input.transpose(1, 2)

        feature_embedded = self.feature_fc(feature_input_t)
        feature_embedded = self.positional_encoding(feature_embedded)
        # print(feature_embedded.size(), embedded.size())

        # Combine ID embedding and feature embedding
        combined_input = embedded + feature_embedded

        # Pass through Transformer
        # attention_mask = attention_mask.to(torch.bool)
        transformer_output = self.transformer_encoder(
            combined_input,
            # src_key_padding_mask=attention_mask
        )

        encoded_feature = transformer_output.mean(dim=1)

        if True:
            encoded_feature = self.projection_head(encoded_feature)
        

        # Output layer
        output = self.fc_out(
            transformer_output.mean(dim=1)
        )  # Aggregate along sequence length

        return output, encoded_feature, None
