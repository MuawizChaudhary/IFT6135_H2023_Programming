from torch.nn.modules import dropout
from torch import nn
from encoder_decoder_solution import GRU, Encoder, DecoderAttn, Attn
import unittest
import random

dtype=torch.float32
atol=1e-6
rtol=1e-4

def TransfertParam_nn_GRU_2_GRU(gru, my_gru, hidden_size):
  my_gru.w_ir.data = gru.weight_ih_l0[0:hidden_size]
  my_gru.w_iz.data = gru.weight_ih_l0[hidden_size:2*hidden_size]
  my_gru.w_in.data = gru.weight_ih_l0[2*hidden_size:3*hidden_size]

  my_gru.w_hr.data = gru.weight_hh_l0[0:hidden_size]
  my_gru.w_hz.data = gru.weight_hh_l0[hidden_size:2*hidden_size]
  my_gru.w_hn.data = gru.weight_hh_l0[2*hidden_size:3*hidden_size]

  my_gru.b_ir.data = gru.bias_ih_l0[0:hidden_size]
  my_gru.b_iz.data = gru.bias_ih_l0[hidden_size:2*hidden_size]
  my_gru.b_in.data = gru.bias_ih_l0[2*hidden_size:3*hidden_size]

  my_gru.b_hr.data = gru.bias_hh_l0[0:hidden_size]
  my_gru.b_hz.data = gru.bias_hh_l0[hidden_size:2*hidden_size]
  my_gru.b_hn.data = gru.bias_hh_l0[2*hidden_size:3*hidden_size]
  
class TestGRU(unittest.TestCase):

    def test_forward_compared_nn_GRU(self):

      input_size = 256
      sequence_length = 128
      batch_size = 4
      hidden_size = 256
      num_layers = 1

      input = torch.randn(batch_size, sequence_length, input_size, dtype=dtype)
      h0 = torch.randn(num_layers, batch_size, hidden_size, dtype=dtype)

      gru_base_test = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=num_layers, batch_first=True)
      
      my_gru = GRU(input_size=input_size, hidden_size=hidden_size)

      TransfertParam_nn_GRU_2_GRU(gru_base_test, my_gru, hidden_size)

      output_base_test, hn_base_test = gru_base_test(input, h0)
      output_my_gru, hn_my_gru = my_gru(input, h0)

      assert torch.allclose(hn_my_gru, hn_base_test, atol=atol, rtol=rtol)
      assert torch.allclose(output_my_gru, output_base_test, atol=atol, rtol=rtol)
 
if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromModule(TestGRU())
  unittest.TextTestRunner(verbosity=2).run(suite)
