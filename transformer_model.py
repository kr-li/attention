import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionEncoding(nn.Module):
    """位置编码实现"""
    def __init__(self, d_model, max_len=5000):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力实现"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.final_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """将输入分割成多个头"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        # 线性变换
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 合并
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        out_put = self.final_linear(scaled_attention)

        return out_put, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """计算注意力权重"""
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))

        dk = torch.tensor(k.size(-1)).float()

        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)


        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        out_put = torch.matmul(attention_weights, v)

        return out_put, attention_weights

class FeedForward(nn.Module):
    """前馈神经网络实现"""
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """编码层实现"""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None):

        # 多头注意力
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(x + attention_output)

        # 前馈神经网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(nn.Module):
    """解码层实现"""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # 自注意力
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        # 编码-解码注意力
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        # 前馈神经网络
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(nn.Module):
    """编码器实现"""
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionEncoding(d_model, maximum_position_encoding)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x

class Decoder(nn.Module):
    """解码器实现"""
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionEncoding(d_model, maximum_position_encoding)

        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        attention_weights = {}
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return x, attention_weights

class Transformer(nn.Module):
    """Transformer模型实现"""
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,  rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_linear = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        enc_output = self.encoder(inp, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_linear(dec_output)
        return final_output, attention_weights

    def create_padding_mask(self, seq):
        seq = torch.eq(seq, 0).float()
        return seq.unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, seq):
        mask = torch.triu(torch.ones(seq, seq), 1)
        return mask.unsqueeze(0).unsqueeze(0)

if __name__ == '__main__':
    transformer = Transformer(
        num_layers=2,
        d_model=512,
        num_heads=8,
        dff=2048,
        input_vocab_size=8500,
        target_vocab_size=8000,
        pe_input=10000,
        pe_target=6000
    )

    temp_input = torch.randint(low=0, high=200, size=(64, 62))
    temp_target = torch.randint(low=0, high=200, size=(64, 52))

    fn_out = transformer(temp_input, temp_target)
    print("模型输出形状", fn_out[0].shape)
    print("注意力权重形状", fn_out[1]['decoder_layer2_block2'].shape)
