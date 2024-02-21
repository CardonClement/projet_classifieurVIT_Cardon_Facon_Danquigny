import math
import torch

class MultiHeadAttention(torch.nn.Module):
    r"""Multi-headed Attention for input Query, Key, Value

    Multi-headed Attention is a module for attention mechanisms which runs through attention in several times in
    parallel, then the multiple outputs are concatenated and linearly transformed

    Args:
        embed_size  (int): Max embedding size
        num_heads   (int): Number of heads in multi-headed attention; Number of splits in the embedding size
        dropout     (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        batch_dim   (int, optional): The dimension in which batch dimensions is

    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.2, batch_dim: int = 0):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_dim = batch_dim

        self.dropout_layer = torch.nn.Dropout(dropout)

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self.Q = torch.nn.Linear(self.embed_size, self.embed_size)
        self.K = torch.nn.Linear(self.embed_size, self.embed_size)
        self.V = torch.nn.Linear(self.embed_size, self.embed_size)

        self.linear = torch.nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        if self.batch_dim == 0:
            out = self.batch_0(q, k, v, mask)
        elif self.batch_dim == 1:
            out = self.batch_1(q, k, v, mask)

        return out

    def batch_0(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, q_embed_size = q.size()
        k_batch_size, k_seq_len, k_embed_size = k.size()
        v_batch_size, v_seq_len, v_embed_size = v.size()

        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self.head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self.head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self.head_size)

        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(v_batch_size, -1, self.embed_size)
        out = self.linear(concatenated)

        return out

    def batch_1(self, q, k, v, mask=None):
        q_seq_len, q_batch_size, q_embed_size = q.size()
        k_seq_len, k_batch_size, k_embed_size = k.size()
        v_seq_len, v_batch_size, v_embed_size = v.size()

        q = self.Q(q).reshape(q_seq_len, q_batch_size, self.num_heads, self.head_size).transpose(0, 1)
        k = self.K(k).reshape(k_seq_len, k_batch_size, self.num_heads, self.head_size).transpose(0, 1)
        v = self.V(v).reshape(v_seq_len, v_batch_size, self.num_heads, self.head_size).transpose(0, 1)

        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(-1, v_batch_size, self.embed_size)

        out = self.linear(concatenated)

        return out

    def attention(self, q, k, v, mask=None):
        scores = torch.einsum("bqhe,bkhe->bhqk", [q, k])

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores /= math.sqrt(self.embed_size)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        attention = torch.einsum("bhql,blhd->bqhd", [scores, v])
        return attention
