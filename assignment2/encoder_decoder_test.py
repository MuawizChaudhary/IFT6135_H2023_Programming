from torch.nn.modules import dropout
from torch import nn
from encoder_decoder_solution import GRU, Encoder, DecoderAttn, Attn
import unittest
import random

dtype=torch.float32
atol=1e-6
rtol=1e-4

class TestEncoder(unittest.TestCase):

    def test_1_forward_shapes(self):
      embedding_size = 256
      input_size = 3
      sequence_length = 128
      vocabulary_size = 30522
      batch_size = 4
      hidden_size = 256
      num_layers = 10
      dropout = 0.1

      input = torch.randint(low=0, high=vocabulary_size-1, 
                            size=(batch_size, sequence_length))

      my_encoder = Encoder(vocabulary_size=vocabulary_size,
              embedding_size=embedding_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              dropout=dropout)
      h0 = my_encoder.initial_states(batch_size)
      x, h = my_encoder(input, h0)

      assert x.size() == torch.Size([batch_size, sequence_length, hidden_size])
      assert h.size() == torch.Size([num_layers, batch_size, hidden_size])

class TestAttn(unittest.TestCase):

    def test_1_forward_shapes(self):
      embedding_size = 256
      input_size = 3
      sequence_length = 128
      vocabulary_size = 30522
      batch_size = 4
      hidden_size = 256
      num_layers = 10
      dropout = 0.1

      input = torch.randint(low=0, high=vocabulary_size-1, 
                            size=(batch_size, sequence_length))
      
      mask = torch.randint(low=0, high=2, 
                            size=(batch_size, sequence_length))

      my_encoder = Encoder(vocabulary_size=vocabulary_size,
              embedding_size=embedding_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              dropout=dropout)
      h0 = my_encoder.initial_states(batch_size)
      x, h = my_encoder(input, h0)

      assert x.size() == torch.Size([batch_size, sequence_length, hidden_size])
      assert h.size() == torch.Size([num_layers, batch_size, hidden_size])

      my_attn = Attn(hidden_size, dropout)
      x_assisted, x_attn = my_attn(x, h, mask=mask)

      assert x_assisted.size() == torch.Size([batch_size, sequence_length, hidden_size])
      assert x_attn.size() == torch.Size([batch_size, 1, hidden_size])

class TestDecoderAttn(unittest.TestCase):

    def test_1_forward_shapes(self):
      embedding_size = 256
      input_size = 3
      sequence_length = 128
      vocabulary_size = 30522
      batch_size = 4
      hidden_size = 256
      num_layers = 10
      dropout = 0.2

      input = torch.randint(low=0, high=vocabulary_size-1, 
                            size=(batch_size, sequence_length))

      my_encoder = Encoder(vocabulary_size=vocabulary_size,
              embedding_size=embedding_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              dropout=dropout)
      h0 = my_encoder.initial_states(batch_size)
      x_encoded, h_encoder = my_encoder(input, h0)

      assert x_encoded.size() == torch.Size([batch_size, sequence_length, hidden_size])
      assert h_encoder.size() == torch.Size([num_layers, batch_size, hidden_size])

      my_decoder = DecoderAttn(vocabulary_size=vocabulary_size,
              embedding_size=embedding_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              dropout=dropout)
      x, h = my_decoder(x_encoded, h_encoder)

      assert x.size() == torch.Size([batch_size, sequence_length, hidden_size])
      assert h.size() == torch.Size([num_layers, batch_size, hidden_size])

if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromModule(TestEncoder())
  unittest.TextTestRunner(verbosity=2).run(suite)

  suite = unittest.TestLoader().loadTestsFromModule(TestAttn())
  unittest.TextTestRunner(verbosity=2).run(suite)
  
  suite = unittest.TestLoader().loadTestsFromModule(TestDecoderAttn())
  unittest.TextTestRunner(verbosity=2).run(suite)
