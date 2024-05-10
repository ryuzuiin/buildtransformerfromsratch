from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        
        # Linear layer for transforming input tensor to query, key, and value tensors
        #qkv_projected = input_tensor @ qkv_projection.weight^T + qkv_projection.bias
        #qkv_projection.weigh.shape = (hidden_size, (hidden_size // 4) * 3)
        self.qkv_projection = nn.Linear(hidden_size, (hidden_size // 4) * 3, bias=bias)
        
        # Linear layer for final output projection
        self.output_projection = nn.Linear(hidden_size // 4, hidden_size, bias=bias)

    def forward(self, input_tensor: Tensor):
        batch_size, sequence_length, hidden_size = input_tensor.shape
        
        # Project input tensor to query, key, and value tensors
        qkv_projected = self.qkv_projection(input_tensor)
        qkv_projected = qkv_projected.reshape(batch_size, sequence_length, 3, hidden_size // 4)
        q, k, v = qkv_projected.unbind(dim=2)
        
        # Compute attention weights using query and key tensors
        attention_weights = q @ k.transpose(-2, -1)
        attention_weights = attention_weights / torch.sqrt(torch.tensor(k.size(-1)))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Apply attention weights to value tensor
        attended_values = attention_weights @ v
        
        # Project attended values to final output
        output_tensor = self.output_projection(attended_values)
        
        return output_tensor
    

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        """
        Initialization function for the Multi-Head Attention module.

        Args:
            hidden_size (int): Size of the input and output hidden layers.
            num_heads (int): Number of attention heads.
            dropout (float): Probability of dropout. Default is 0.1.
            bias (bool): Whether to use bias in the linear layers. Default is True.
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.dropout = nn.Dropout(dropout)

        # Linear layer for computing Q, K, V
        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        # Linear layer for computing the final output
        self.output_linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation function for the Multi-Head Attention module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Compute final output
        output = self.output_linear(attn_output)

        return output

class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: BoolTensor = None) -> Tensor:
        batch_size, seq_length, hidden_size = x.size()

        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("bqhd,bkhd->bhqk", [q, k]) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        attn_probs = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        attn_output = torch.einsum("bhqv,bqhd->bqhd", [attn_probs, v])
        attn_output = attn_output.contiguous().view(batch_size, seq_length, hidden_size)
        output = self.proj_dropout(self.out_proj(attn_output))

        return output

class CausalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, context_size: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("causal_mask", torch.triu(torch.ones(context_size, context_size, dtype=torch.bool), diagonal=1))

    def forward(self, x: Tensor, mask: BoolTensor = None) -> Tensor:
        batch_size, seq_length, hidden_size = x.size()

        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("bqhd,bkhd->bhqk", [q, k]) / math.sqrt(self.head_dim)

        causal_mask = self.causal_mask[:seq_length, :seq_length].unsqueeze(0).unsqueeze(1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask | mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        else:
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))
        attn_output = torch.einsum("bhqv,bqhd->bqhd", [attn_probs, v])
        attn_output = attn_output.contiguous().view(batch_size, seq_length, hidden_size)
        output = self.out_proj(attn_output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor = None) -> Tensor:
        batch_size, q_length, hidden_size = q.size()
        _, kv_length, _ = k.size()

        q = self.q_proj(q).view(batch_size, q_length, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, kv_length, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, kv_length, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("bqhd,bkhd->bhqk", [q, k]) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        attn_probs = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        attn_output = torch.einsum("bhqv,bvhd->bqhd", [attn_probs, v])

        attn_output = attn_output.contiguous().view(batch_size, q_length, hidden_size)
        output = self.proj_dropout(self.out_proj(attn_output))

        return output
    
