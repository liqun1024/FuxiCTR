# =========================================================================
# Copyright (C) 2025. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention


class SIM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="SIM", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dropout=0,
                 attention_dim=64,
                 num_heads=1,
                 gsu_type="soft",
                 short_seq_len=50,
                 topk=50,
                 alpha=1,
                 beta=1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(SIM, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.topk = topk
        self.short_seq_len = short_seq_len
        self.alpha = alpha
        self.beta = beta
        assert gsu_type == "soft", "Only support soft search currently!"
        self.item_info_dim = 0
        for _, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.W_a = nn.Linear(self.item_info_dim, attention_dim, bias=False)
        self.W_b = nn.Linear(self.item_info_dim, attention_dim, bias=False)
        self.short_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                        attention_dim,
                                                        num_heads,
                                                        attention_dropout)
        self.long_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                       attention_dim,
                                                       num_heads,
                                                       attention_dropout)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        self.dnn_aux = MLP_Block(input_dim=input_dim,
                                 output_dim=1,
                                 hidden_units=dnn_hidden_units,
                                 hidden_activations=dnn_activations,
                                 output_activation=self.output_activation, 
                                 dropout_rates=net_dropout,
                                 batch_norm=batch_norm)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim * 2
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        item_dict, item_mask, topk_dict, topk_mask = inputs
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = item_mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]

        # short interest attention
        short_seq_emb = item_feat_emb[:, -self.short_seq_len:-1, :]
        short_mask = item_mask[:, -self.short_seq_len:-1]
        short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)

        # long interest attention
        topk_emb = self.embedding_layer(topk_dict, flatten_emb=True)
        long_interest_emb = self.long_attention(target_emb, topk_emb, topk_mask)
        emb_list = emb_list[0:-1] + [short_interest_emb, long_interest_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return y_pred

    def add_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true, reduction='mean')

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def get_loss(self, batch_data):
        y_pred = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(y_pred, y_true)
        loss = loss / self.accumulation_steps
        return loss
