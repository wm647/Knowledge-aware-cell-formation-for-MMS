# coding=utf-8

import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from BertTENERCRF.model import CRF
from torch.autograd import Variable
import torch
import ipdb

class BERT_TENER_CRF(nn.Module):
    """
    bert_tener_crf model
    """

    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1,
                 use_cuda=False, num_heads=8):
        super(BERT_TENER_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = rnn_layers  # 重用 rnn_layers 参数作为 transformer 层数
        self.num_heads = num_heads

        # BERT模型加载（保持原有方式）
        self.word_embeds = BertModel.from_pretrained(bert_config)

        # 如果 BERT 输出维度与 Transformer 维度不同，添加投影层
        if embedding_dim != hidden_dim:
            self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.input_projection = None

        # Transformer Encoder 配置
        # 确保hidden_dim能被num_heads整除
        if hidden_dim % num_heads != 0:
            # 自动调整num_heads
            for heads in [8, 4, 2, 1]:
                if hidden_dim % heads == 0:
                    self.num_heads = heads
                    print(f"Auto-adjusted num_heads to {heads} for hidden_dim {hidden_dim}")
                    break

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=hidden_dim * 4,  # 通常是模型维度的4倍
            dropout=dropout_ratio,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # 为了保持与原始LSTM版本的兼容性，输出维度保持为 hidden_dim*2
        self.output_projection = nn.Linear(hidden_dim, hidden_dim * 2)

        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)

        # 修正：liner输出维度应该是tagset_size，而不是tagset_size+2
        self.liner = nn.Linear(hidden_dim * 2, tagset_size)
        self.tagset_size = tagset_size

    def forward(self, sentence, attention_mask=None):
        '''
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            attention_mask: attention mask for BERT

        return:
            transformer output (batch_size, word_seq_len, tag_size)
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)

        # BERT 词嵌入
        embeds, _ = self.word_embeds(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)

        # 投影到 Transformer 维度（如果需要）
        if self.input_projection is not None:
            embeds = self.input_projection(embeds)

        # 为 Transformer 创建 padding mask
        if attention_mask is not None:
            # 将 attention_mask 转换为 padding mask (True 表示 padding 位置)
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Transformer Encoder 前向传播
        transformer_out = self.transformer_encoder(
            embeds,
            src_key_padding_mask=padding_mask
        )

        # 投影到与原始LSTM输出相同的维度
        transformer_out = self.output_projection(transformer_out)

        # 应用 dropout 和线性变换
        transformer_out = transformer_out.contiguous().view(-1, self.hidden_dim * 2)
        d_transformer_out = self.dropout1(transformer_out)
        l_out = self.liner(d_transformer_out)
        transformer_feats = l_out.contiguous().view(batch_size, seq_length, -1)

        return transformer_feats

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
        mask: size=(batch_size, seq_len)
        tags: size=(batch_size, seq_len)
        :return:
        """
        # 修正：调用正确的方法名
        loss_value = self.crf.neg_log_likelihood(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
