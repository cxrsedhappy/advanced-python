import math


def scaled_dot_product_attention(q, k, v, mask, dropout):
    d_k = q.shape[-1]
    attention = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # @ is matmul holy shit

    if mask is not None:
        attention.masked_fill_(mask == 0, -1e9)

    attention_scores = attention.softmax(dim=-1)

    if dropout is not None:
        attention = dropout(attention_scores)

    return (attention @ v), attention
