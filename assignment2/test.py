import unittest

import torch

from transformer_solution_template import MultiHeadedAttention, Transformer
from encoder_decoder_solution_template import Encoder, Attn
# from transformer_solution import MultiHeadedAttention, Transformer
# from encoder_decoder_solution import Encoder, Attn

class TestEncoderDecoder(unittest.TestCase):
    def test_attn(self):

        batch_size = 32
        sequence_length = 16
        hidden_size = 8
        num_layers = 1

        attn = Attn(hidden_size=hidden_size)

        inputs = torch.rand(batch_size, sequence_length, hidden_size)
        hidden_states = torch.rand(num_layers, batch_size, hidden_size)

        output, _ = attn(inputs, hidden_states)
        
        assert inputs.shape == output.shape

    def test_encoder(self):

        batch_size = 32
        sequence_length = 16
        hidden_size = 8
        num_layers = 1

        enc = Encoder(hidden_size=hidden_size)

        inputs = torch.randint(0, 10, (batch_size, sequence_length))
        hidden_states = torch.rand(num_layers*2, batch_size, hidden_size)

        output, hidden_states = enc(inputs, hidden_states)
        
        assert output.shape == (batch_size, sequence_length, hidden_size)
        assert hidden_states.shape == (num_layers, batch_size, hidden_size)

class TestTransformer(unittest.TestCase):
    def test_get_attention_weights_no_mask(self):

        head_size = 4
        num_heads = 8
        sequence_length = 16
        batch_size = 32

        mha = MultiHeadedAttention(head_size, num_heads, sequence_length)

        q = torch.rand(batch_size, num_heads, sequence_length, head_size)
        k = torch.rand(batch_size, num_heads, sequence_length, head_size)

        attention_weights = mha.get_attention_weights(q, k)
        
        assert attention_weights.shape == (batch_size, num_heads, sequence_length, sequence_length)

    def test_split_heads(self):

        dim = 3
        head_size = 4
        num_heads = 8
        sequence_length = 16
        batch_size = 32

        mha = MultiHeadedAttention(head_size, num_heads, sequence_length)

        x = torch.rand(batch_size, sequence_length, num_heads * dim)

        splited_x = mha.split_heads(x)
        
        assert splited_x.shape == (batch_size, num_heads, sequence_length, dim)

    def test_merge_heads(self):

        dim = 3
        head_size = 4
        num_heads = 8
        sequence_length = 16
        batch_size = 32

        mha = MultiHeadedAttention(head_size, num_heads, sequence_length)

        x = torch.rand(batch_size, num_heads, sequence_length, dim)

        merged_x = mha.merge_heads(x)
        
        assert merged_x.shape == (batch_size, sequence_length, num_heads * dim)


if __name__ == '__main__':
    unittest.main()