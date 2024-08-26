import math
import torch
import torch_mlu
from bangtransformer.torch import bt_ops
from itertools import product
from infer_ext.vendor import vendor_ops_registry
from infer_ext.vendor.camb import camb_ops
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple

passed = 0
failed = 0

class SelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale

    # special alibi only support causal
    def build_alibi(self, slopes, block_size, n_heads, dtype):
        device ='mlu'
        tril = torch.tril(torch.ones(1,1 , block_size, block_size, device = device))
        bias_rows = torch.arange( block_size, device=device).view(1, -1)
        bias_cols = torch.arange( block_size, device=device).view(-1, 1)
        bias = - torch.sqrt(bias_cols - bias_rows)
        bias = bias.view(1, block_size, block_size) * slopes.view(-1, 1, 1)
        bias = bias.masked_fill(tril == 0, float('-inf'))
        return bias.type(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cur_seq_len_t:torch.Tensor, alibi_slope:torch.Tensor, attn_bias:torch.Tensor):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D)
            k: The tensor containing the key.   (B, T, H, D)
            v: The tensor containing the value. (B, T, H, D)
            cur_seq_len_t: true_seq_lens. (B+1)
            alibi_slope: (H) or (B, H)
            attn_bias: (B,H,T,T) or (B,T,T)
        """
        batch = q.shape[0]
        seq_q = q.shape[1]
        seq_k = k.shape[1]
        head = q.shape[2]
        scores = torch.einsum('bthd,bshd->bhts', q, k )* self.softmax_scale
        # mask
        if alibi_slope is not None:
            slope = torch.zeros((batch, head)).mlu()
            if len(alibi_slope.shape) == 1 :
                slope[:,]=alibi_slope
            else:
                slope=alibi_slope
            slope = slope.reshape(batch, head, 1, 1)
            slope_bias = torch.zeros(batch, head, seq_q, seq_k).mlu()
            if self.causal:
                relative_pos = torch.arange(-seq_k + 1, 1, dtype=torch.float32).mlu()
                slope_bias = relative_pos * slope
            else:
                row_idx = torch.arange(seq_q, dtype=torch.long).reshape(-1, 1)
                col_idx = torch.arange(seq_k, dtype=torch.long)
                relative_pos = torch.abs(row_idx + seq_k - seq_q - col_idx).mlu()
                slope_bias = -slope * relative_pos.to(dtype=slope.dtype)
            # if use special alibi
            # slope_bias = self.build_alibi(alibi_slope, seq_k, head, dtype=torch.float32)

            scores += slope_bias
        if attn_bias is not None:
            if len(attn_bias.shape) == 3:
                scores += attn_bias.unsqueeze(1)
            else:
                scores +=attn_bias
        if self.causal:
            causal_mask = torch.triu(torch.full((seq_q, seq_k), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        else: # fill -inf in pad_area
            for b in range(batch):
                true_seq_len = cur_seq_len_t[b + 1] - cur_seq_len_t[b]
                scores[b, ..., true_seq_len:] = -10000.0
                scores[b, :, true_seq_len:, :] = -10000.0
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        output = torch.einsum('bhts,bshd->bthd', attention, v)
        return output.contiguous()

class TestFlashAttnOp():
    # @unittest.skip("not test")
    def test_flash_attention(self):
        t_len_sequence_list = [(38, 64, 128), (34, 56, 78), (32, 47, 58)]
        head_num = 32
        head_size_list = [64, 256]
        alibi_lst = [True, False]
        mask_lst = [False, False]
        causal_lst = [True, True]
        def copy_pack_data_to_pad_data(pad_input: torch.Tensor,
                                       packed_input_list: list,
                                       t_len_sequence: list,
                                       max_sequence_len: int):
            for index in range(len(t_len_sequence)):
                start_index = index * max_sequence_len
                end_index = start_index + t_len_sequence[index]
                pad_input[start_index:end_index, ...] = packed_input_list[index]

        def get_pack_and_pad_input(suquence_list: list,
                                   head_num: int, head_size: int):
            q_tensor_list = []
            k_tensor_list = []
            v_tensor_list = []
            for i in range(len(suquence_list)):
                tensor_q = torch.randn((suquence_list[i], head_num, head_size)).mlu().half()
                q_tensor_list.append(tensor_q)
                tensor_k = torch.randn((suquence_list[i], head_num, head_size)).mlu().half()
                k_tensor_list.append(tensor_k)
                tensor_v = torch.randn((suquence_list[i], head_num, head_size)).mlu().half()
                v_tensor_list.append(tensor_v)
            max_sequence_len = max(t_len_sequence)
            batch = len(t_len_sequence)
            pad_input_q = torch.zeros((max_sequence_len*batch, head_num, head_size)).mlu().half()
            pad_input_k = torch.zeros((max_sequence_len*batch, head_num, head_size)).mlu().half()
            pad_input_v = torch.zeros((max_sequence_len*batch, head_num, head_size)).mlu().half()
            copy_pack_data_to_pad_data(pad_input_q, q_tensor_list,
                                       t_len_sequence, max_sequence_len)
            copy_pack_data_to_pad_data(pad_input_k, k_tensor_list,
                                       t_len_sequence, max_sequence_len)
            copy_pack_data_to_pad_data(pad_input_v, v_tensor_list,
                                       t_len_sequence, max_sequence_len)
            packed_q = torch.cat(q_tensor_list, dim=0)
            packed_k = torch.cat(k_tensor_list, dim=0)
            packed_v = torch.cat(v_tensor_list, dim=0)
            return pad_input_q, pad_input_k, pad_input_v, packed_q, packed_k, packed_v


        args = product(t_len_sequence_list, head_size_list, alibi_lst, mask_lst, causal_lst)
        for t_len_sequence, head_size, has_alibi, has_mask, is_causal in args:
            # prepare input
            pad_and_pack_input = get_pack_and_pad_input(t_len_sequence, head_num, head_size)
            max_sequence_len = max(t_len_sequence)
            batch = len(t_len_sequence)
            print("batch={}, seq_lens={}, head_num={}, head_size={}, has_alibi={}, has_mask={}, is_causal={}, testing...".format(
                batch, t_len_sequence, head_num, head_size, has_alibi, has_mask, is_causal), flush=True)
            # cur_seq_len
            cur_seq_len = [0]
            for value in t_len_sequence:
                cur_seq_len.append(cur_seq_len[-1] + value)
            cur_seq_len_t = torch.tensor(cur_seq_len, dtype = torch.int32).mlu()
            softmax_scale = 1 / math.sqrt(head_size)
            alibi_slope = torch.zeros((head_num)).uniform_(0, 0.1).to(torch.float32).mlu()
            # if use special alibi
            # n = 2 ** math.floor(math.log2(head_num))
            # m0 = 2.0 ** (-8.0/n)
            # slopes = torch.pow(m0, torch.arange(1, n+1))
            # if n < head_num:
            #     m1 = 2.0 ** (-4.0 / n)
            #     mm = torch.pow(m1, torch.arange(1, 1+2* (head_num - n), 2))
            #     slopes = torch.cat([slopes, mm])
            # alibi_slope = slopes.to('mlu')

            # attn_bias = torch.randn((batch, max_sequence_len, max_sequence_len), dtype=torch.float16).mlu()
            attn_bias = torch.randn((batch, head_num, max_sequence_len, max_sequence_len), dtype=torch.float16).mlu()
            attn_bias = None

            # small op module
            attention = SelfAttention(causal = is_causal, softmax_scale=softmax_scale)
            torch_output = attention(pad_and_pack_input[0].view(batch, max_sequence_len, head_num, head_size),
                                     pad_and_pack_input[1].view(batch, max_sequence_len, head_num, head_size),
                                     pad_and_pack_input[2].view(batch, max_sequence_len, head_num, head_size),
                                     cur_seq_len_t,
                                     alibi_slope if has_alibi else None,
                                     attn_bias if has_mask else None)
            bt_output = torch.zeros_like(pad_and_pack_input[3])
            bt_ops.flash_attention(pad_and_pack_input[3],
                               pad_and_pack_input[4],
                               pad_and_pack_input[5],
                               cur_seq_len_t,
                               alibi_slope if has_alibi else None,
                               attn_bias if has_mask else None,
                               bt_output,
                               max_sequence_len,
                               softmax_scale,
                               is_causal, -1, -1)
            query = pad_and_pack_input[3]
            key = pad_and_pack_input[4]
            value = pad_and_pack_input[5]
            q_start_loc = None
            seq_len = cur_seq_len_t
            num_q_heads = query.shape[1]
            num_kv_heads = key.shape[1]
            attn_mask = None
            attn_qk_scale = softmax_scale
            alibi_slopes = alibi_slope if has_alibi else None
            attn_output = bt_output

            attn_output = vendor_ops_registry["context_attention"](query, key, value, q_start_loc, seq_len, num_q_heads, num_kv_heads, attn_mask, attn_qk_scale, alibi_slopes, attn_output) 
            
            # copy torch_output to pack mode
            torch_pack_output = torch.zeros_like(bt_output)
            torch_output_viewed = torch_output.view((batch*max_sequence_len, head_num, head_size))
            for index in range(len(t_len_sequence)):
                start_index = index * max_sequence_len
                end_index = start_index + t_len_sequence[index]
                torch_pack_output[cur_seq_len[index]:cur_seq_len[index+1], ...] \
                    = torch_output_viewed[start_index:end_index, ...]
            diff = torch_pack_output.cpu() - bt_output.cpu()
            global passed
            global failed
            if diff.mean() < 1e-5:
                passed = passed + 1
            else:
                failed = failed + 1
            # self.assertTensorsEqual(torch_pack_output.cpu().float(),
                                    # bt_output.cpu().float(),
                                    # 0.003, use_MSE=True, use_RAE=True)

if __name__ == '__main__':
    test = TestFlashAttnOp()
    test.test_flash_attention()
    print(f"Total number of test is {passed + failed}, pass: {passed}, fail: {failed}")
