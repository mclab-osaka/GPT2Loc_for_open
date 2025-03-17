import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel

# from try_config_hydra import Config
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


def reinitialize_weights(module):
    if hasattr(module, "reset_parameters"):  # リセット可能なモジュールを対象
        module.reset_parameters()


class GPT2_back_Model(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length=1000,
        feature_dim=1,
        will_recover_rssi=False,
        random_init=False,
        is_projection_head=False,
    ):
        super(GPT2_back_Model, self).__init__()

        self.is_projection_head = is_projection_head

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.feature_fc = nn.Linear(feature_dim, d_model)
        self.will_recover_rssi = will_recover_rssi
        self.fc_out = nn.Linear(d_model, 2)  # Output x, y

        if will_recover_rssi:
            self.decode_feature_fc = nn.Linear(d_model, feature_dim)
            

        if False:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())
        else:
            self.gpt2 = GPT2Model.from_pretrained(
                "gpt2", output_hidden_states=True
            )  # loads a pretrained GPT-2 base model

            if random_init:
                print("random init")
                self.gpt2 = GPT2Model(GPT2Config(n_embd=d_model, n_head=nhead))
                self.gpt2.apply(reinitialize_weights)

        self.gpt2.h = self.gpt2.h[:num_layers]

        is_mlp = False

        if True:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if "ln" in name or "wpe" in name:  # or 'mlp' in name:
                    param.requires_grad = True
                elif "mlp" in name and is_mlp:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        if self.is_projection_head:
            self.projection_head = nn.Linear(d_model, 64)  # Output x, y

    def forward(self, id_input, feature_input, attention_mask=None):

        embedded = self.embedding(id_input)

        # Process feature input
        # print(feature_input.size())
        feature_input_t = feature_input.transpose(1, 2)
        feature_embedded = self.feature_fc(feature_input_t)
        feature_embedded = self.positional_encoding(feature_embedded)

        combined_input = embedded + feature_embedded
        # combined_input = feature_embedded

        outputs = self.gpt2(
            inputs_embeds=combined_input, attention_mask=attention_mask
        ).last_hidden_state

        feature = outputs.mean(dim=1)
        loc = self.fc_out(feature)  # Aggregate along sequence length 

        if self.is_projection_head:
            feature = self.projection_head(feature)

        if self.will_recover_rssi:
            
            recovered_rssi = self.decode_feature_fc(outputs).squeeze()
            return loc, feature, recovered_rssi
        else:
            return loc, feature, None
