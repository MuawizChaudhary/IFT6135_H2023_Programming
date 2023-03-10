from torch.nn.modules import dropout
from torch import nn
from transformer_solution import LayerNorm, MultiHeadedAttention, Transformer
import unittest
import random

atol=1e-6
rtol=1e-4

class TestLayerNorm(unittest.TestCase):

    def test_forward(self):

      batch, sentence_length, embedding_dim = 64, 256, 256

      embedding = torch.randn(batch, sentence_length, embedding_dim)

      # la base de référence
      layer_norm = nn.LayerNorm(embedding_dim)
      o_base = layer_norm(embedding)

      # mon implémentation
      my_layer_norm = LayerNorm(embedding_dim)
      o_test = my_layer_norm(embedding)

      assert o_test.size() == o_base.size()
      assert torch.allclose(o_test, o_base, atol=atol, rtol=rtol)

class TestMultiHeadedAttention(unittest.TestCase):

    def test_1_merge_heads_shape(self):
      batch_size, num_heads, sequence_length, head_size = 64, 12, 128, 256
      t = torch.randn(batch_size, num_heads, sequence_length, head_size)
      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)

      t_test = my_multi_head_attn.merge_heads(t)

      assert t_test.size() == torch.Size([batch_size, sequence_length, num_heads * head_size])

    def test_2_split_heads_shape(self):
      batch_size, num_heads, sequence_length, head_size = 64, 12, 128, 256
      t = torch.randn(batch_size, sequence_length, num_heads*head_size)
      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)

      t_test = my_multi_head_attn.split_heads(t)

      assert t_test.size() == torch.Size([batch_size, num_heads, sequence_length, head_size])

    def test_3_merge_split_heads_identity(self):
      batch_size, num_heads, sequence_length, head_size = 64, 12, 128, 256
      t = torch.randn(batch_size, num_heads, sequence_length, head_size)
      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)

      t_merged = my_multi_head_attn.merge_heads(t)
      t_merged_splited = my_multi_head_attn.split_heads(t_merged)

      assert torch.allclose(t_merged_splited, t, atol=atol, rtol=rtol)

    def test_21_merge_heads_value(self):
      batch_size, num_heads, sequence_length, head_size = 2, 2, 1, 2
      t = torch.randn(batch_size, num_heads, sequence_length, head_size)
      t[0,0,0,:] = torch.tensor([1,2])
      t[0,1,0,:] = torch.tensor([3,4])
      t[1,0,0,:] = torch.tensor([5,6])
      t[1,1,0,:] = torch.tensor([7,8])
      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)

      t_test = my_multi_head_attn.merge_heads(t)

      t_result = torch.tensor([1,2,3,4,5,6,7,8]).float()
      t_result = t_result.reshape(2,1,4)

      assert t_test.size() == t_result.size()
      assert torch.allclose(t_test, t_result, atol=atol, rtol=rtol)

    def test_4_get_attention_weights_shape(self):
      batch_size, num_heads, sequence_length, head_size = 64, 12, 128, 256
      queries = torch.randn(batch_size, num_heads, sequence_length, head_size)
      keys = torch.randn(batch_size, num_heads, sequence_length, head_size)
      mask = torch.randint(0,2,(batch_size, sequence_length-1)).double()

      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)

      t_test = my_multi_head_attn.get_attention_weights(queries, keys, mask)

      assert t_test.size() == torch.Size([batch_size, num_heads, sequence_length, sequence_length])      

    def test_5_get_attention_weights_values(self):
      batch_size, num_heads, sequence_length, head_size = 2, 3, 4, 5

      x = torch.randn(batch_size, num_heads, sequence_length, head_size)
      multihead_attn = nn.MultiheadAttention(num_heads*head_size, num_heads, batch_first=True, bias=False)

      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)
      x_m = my_multi_head_attn.merge_heads(x)

      attn_output, attn_output_weights = multihead_attn(x_m, x_m, x_m, average_attn_weights=False)

      W_q = multihead_attn.in_proj_weight[0:num_heads*head_size,:]
      W_k = multihead_attn.in_proj_weight[num_heads*head_size:num_heads*head_size*2, :]
      W_v = multihead_attn.in_proj_weight[num_heads*head_size*2:num_heads*head_size*3, :]

      queries_m = torch.matmul(x_m,W_q)
      queries = my_multi_head_attn.split_heads(queries_m)

      keys_m = torch.matmul(x_m,W_k)
      keys = my_multi_head_attn.split_heads(keys_m)

      t_test = my_multi_head_attn.get_attention_weights(queries, keys)

      assert t_test.size() == attn_output_weights.size()

    def test_6_apply_attention_shape(self):
      batch_size, num_heads, sequence_length, head_size = 64, 12, 128, 256
      queries = torch.randn(batch_size, num_heads, sequence_length, head_size)
      keys = torch.randn(batch_size, num_heads, sequence_length, head_size)
      values = torch.randn(batch_size, num_heads, sequence_length, head_size)
      mask = torch.randint(0,2,(batch_size, sequence_length-1)).double()

      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)

      t_test = my_multi_head_attn.apply_attention(queries, keys, values, mask)

      assert t_test.size() == torch.Size([batch_size, sequence_length, num_heads * head_size])

    def test_7_forward_shape(self):
      batch_size, num_heads, sequence_length, head_size = 64, 4, 256, 128
      X = torch.randn(batch_size, sequence_length, num_heads * head_size)
      mask = torch.randint(0,2,(batch_size, sequence_length-1)).double()

      my_multi_head_attn = MultiHeadedAttention(head_size, num_heads, sequence_length)

      outputs = my_multi_head_attn(X, mask)

      assert outputs.size() == torch.Size([batch_size, sequence_length, num_heads * head_size])      
      
class TestTransformer(unittest.TestCase):      
    def test_1_forward_shape(self):
      vocabulary_size=30522
      embed_dim=256
      hidden_dim=256
      num_heads=1
      num_layers=2
      block='prenorm'
      dropout=0.3

      batch_size = 2
      sequence_length = 256

      X = torch.randint(0,vocabulary_size,(batch_size, sequence_length))
      mask = torch.randint(0,2,(batch_size, sequence_length))

      my_transformer = Transformer(vocabulary_size, embed_dim, hidden_dim, num_heads,
                                  num_layers, block, dropout)

      outputs = my_transformer(X, mask)

      assert outputs.size() == torch.Size([batch_size, embed_dim])      
