import torch
from torch import nn
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention, Dice
from fuxictr.utils import not_in_whitelist
from fuxictr.pytorch.torch_utils import get_loss

from model_zoo.LongCTR.MISC.layers import CategoryInterestAttention

class MISC(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="MISC", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 short_seq_len=20,
                 net_dropout=0, 
                 batch_norm=False, 
                 din_use_softmax=False,
                 accumulation_steps=1,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(MISC, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.short_seq_len = short_seq_len
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.category_attention = CategoryInterestAttention(self.item_info_dim, max_categories=200)

        self.short_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                        attention_dim=256,
                                                        num_heads=4,
                                                        dropout_rate=0.1)

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

    def get_optimizer(self, optimizer, params, lr=None):
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                optimizer = "Adam"
        try:
            if lr is not None:
                # 传统方式：所有参数使用相同 lr
                optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
            else:
                # params 是参数组（list of dicts），每组有自己的 lr
                optimizer = getattr(torch.optim, optimizer)(params)
        except:
            raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
        return optimizer

    def compile(self, optimizer, loss, lr):    
        
        # 默认学习率
        base_lr = lr
        category_lr = base_lr * 10  # 10倍学习率

        # 获取所有参数，但排除 category_attention 的参数
        base_params = []
        category_params = []

        for name, param in self.named_parameters():
            if 'category_attention' in name:
                category_params.append(param)
            else:
                base_params.append(param)

        # 构建参数组
        param_groups = [
            {'params': base_params, 'lr': base_lr},
            {'params': category_params, 'lr': category_lr}
        ]

        self.optimizer = self.get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            emb_out = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(emb_out)
        
        cate_hist = item_dict["cate_hist"]
        item_dict.pop("cate_hist", None)

        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        cate_hist = cate_hist[:, 0:-1]
        # pooling_emb = self.category_attention(target_emb, sequence_emb, cate_hist, mask)

        # short_seq_emb
        short_seq_emb = sequence_emb[:, -self.short_seq_len:-1, :]
        short_mask = mask[:, -self.short_seq_len:-1]
        short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)

        long_seq_emb = sequence_emb[:, :-self.short_seq_len, :]
        long_mask = mask[:, :-self.short_seq_len]
        long_cate_hist = cate_hist[:, :-self.short_seq_len]
        long_mask = long_mask.float().unsqueeze(-1)
        # pooling_emb = (long_seq_emb * long_mask).sum(dim=1) / long_mask.sum(dim=1).clamp(min=1.0)
        pooling_emb = self.category_attention(target_emb, long_seq_emb, long_cate_hist, long_cate_hist)
        emb_list += [target_emb, pooling_emb, short_interest_emb]

        # emb_list += [target_emb, short_interest_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        if torch.any(torch.isnan(y_pred)):
            raise ValueError("Error")
        return return_dict

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

