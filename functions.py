import numpy as np
# Attention is all your needed
def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    d_k = Q.shape[1]
    score = np.matmul(Q, K.T) / np.sqrt(d_k)

    exp_score = np.exp(score - np.max(score, axis = 1, keepdims = True))
    attention_weight = exp_score / np.sum(exp_score, axis = 1, keepdims = True)

    output = np.matmul(attention_weight, V)

    return output