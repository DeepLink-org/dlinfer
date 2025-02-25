// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include "../quantization/fp8/amd/quant_utils.cuh"
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include "../quantization/fp8/nvidia/quant_utils.cuh"
#endif

#ifndef USE_ROCM
  #define WARP_SIZE 32
#else
  #define WARP_SIZE warpSize
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

template<int NUM_WARPS>
inline __device__ float mxblock_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / MXWARP_SIZE;
  int lane = threadIdx.x % MXWARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = MXWARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += MXVLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }
 // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += MXVLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return MXVLLM_SHFL_SYNC(sum, 0);
}

template<typename scalar_t>
__device__  float __forceinline__ atten_mul(scalar_t *a, float b, int j) {
}

template<>
__device__ float __forceinline__ atten_mul(uint16_t *a, float b, int j) {
    return __half2float(*((half*)a + j)) * __half2float(__float2half(b));
}

template<>
__device__ float __forceinline__ atten_mul(__nv_bfloat16 *a, float b, int j) {
    return __bfloat162float(*(a + j)) * __bfloat162float(__float2bfloat16(b));
}

template<typename scalar_t, typename cache_t>
__device__ float __forceinline__ atten_dot(scalar_t* a, cache_t *b ,int i){

}
template<>
__device__ float __forceinline__ atten_dot(uint16_t* a, uint16_t *b ,int i){
  return __half2float(*((half*)a + i)) * __half2float(*((half*)b + i));
}

template<>
__device__ float __forceinline__ atten_dot(__nv_bfloat16* a, __nv_bfloat16 *b ,int i){
  return __bfloat162float(a[i]) * __bfloat162float(b[i]);
}


// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float k_scale, const float v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 32 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

  int block_idx0 = start_block_idx + warp_idx;
  int kv_offset0, kv_offset1;
  K_vec load_k[NUM_VECS_PER_THREAD];
  K_vec compute_k[NUM_VECS_PER_THREAD];

  kv_offset0 = block_table[block_idx0];
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  
  for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
    const int token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
    const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                      + kv_head_idx * kv_head_stride
                                      + physical_block_offset * x;

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
	const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
#if 0
        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
              k_vec_quant, k_scale);
        }
#endif
	load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }
  }

   for (int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    for(int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        compute_k[j] = load_k[j];
      }
      if(block_idx < end_block_idx - NUM_WARPS) {
          kv_offset0 = kv_offset1;
          if(block_idx < end_block_idx - (NUM_WARPS << 1)) {
            kv_offset1 = block_table[block_idx + (NUM_WARPS<<1)];
          }

          const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                      + kv_head_idx * kv_head_stride
                                      + physical_block_offset * x;
          #pragma unroll NUM_VECS_PER_THREAD
          for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
              const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
              const int offset1 = (vec_idx * VEC_SIZE) / x;
              const int offset2 = (vec_idx * VEC_SIZE) % x;
#if 0        
	      if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
                  k_vecs[j] = *reinterpret_cast<const K_vec*>(
                  k_ptr + offset1 * BLOCK_SIZE * x + offset2);
              } else {
                  // Vector conversion from Quant_vec to K_vec.
                  Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
                      k_ptr + offset1 * BLOCK_SIZE * x + offset2);
                  k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
                      k_vec_quant, k_scale);
              }
#endif
	      load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      // Compute the parallel products for Q*K^T (treat vector lanes separately).
      float qk = 0.0f;
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        scalar_t *ptr_q = (scalar_t*)&q_vecs[thread_group_offset][j];
        cache_t *ptr_c = (cache_t*)&compute_k[j];
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; k++) {
          qk += atten_dot(ptr_q,ptr_c,k);
        }
      }

      #pragma unroll
      for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += VLLM_SHFL_XOR_SYNC(qk, mask);
      }
      qk = scale * qk;

      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int NUM_V_VECS_PER_THREAD = HEAD_SIZE / V_VEC_SIZE;
  constexpr int NUM_COLS_PER_ITER = MAX(WARP_SIZE / NUM_V_VECS_PER_THREAD,1);
  constexpr int NUM_VALID_THREAD = NUM_COLS_PER_ITER * NUM_V_VECS_PER_THREAD;
  constexpr int NUM_LGT_PER_COL = (BLOCK_SIZE + NUM_COLS_PER_ITER - 1) / NUM_COLS_PER_ITER;
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  const int physical_block_offset = lane / NUM_V_VECS_PER_THREAD;
  const int laneid = lane % NUM_V_VECS_PER_THREAD;
  V_vec v_vecs[NUM_LGT_PER_COL];
  V_vec v_prev_vecs[NUM_LGT_PER_COL];
  float accs[V_VEC_SIZE];
  float reg_log[NUM_LGT_PER_COL];
  float reg_prev_log[NUM_LGT_PER_COL];
  #pragma unroll
  for(int i = 0; i < V_VEC_SIZE; i++) {
    accs[i] = 0.0f;
  }
  int token_idx, kv_stride, block_offset;
  kv_stride = BLOCK_SIZE * HEAD_SIZE ;
  kv_offset0 = block_table[block_idx0];
  block_offset = NUM_COLS_PER_ITER * HEAD_SIZE;
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
  const cache_t *v_ptr0 = v_cache + kv_head_idx * kv_stride + physical_block_offset * HEAD_SIZE; 
  const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
  float *ptr_logits = logits + token_idx - start_token_idx;
  if(lane < NUM_VALID_THREAD) {
    if(block_idx0 == num_seq_blocks - 1) {
    #pragma unroll
      for(int i = 0; i < NUM_LGT_PER_COL; i++) {
        if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
          const int idx = laneid * V_VEC_SIZE + i * block_offset;
          v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
          reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
        }
      }
    } else {
      #pragma unroll
      for(int i = 0; i < NUM_LGT_PER_COL; i++) {
        if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
          const int idx = laneid * V_VEC_SIZE + i * block_offset;
          v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
          reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
        }
      }
    }


  for(int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
      int next_block = block_idx + NUM_WARPS;
      int nnext_block = next_block + NUM_WARPS;
      for(int i = 0; i < NUM_LGT_PER_COL; i++) {
          v_vecs[i] = v_prev_vecs[i];
          reg_log[i] = reg_prev_log[i];
      }
      if(next_block < end_block_idx) {
          kv_offset0 = kv_offset1;
          if(nnext_block < end_block_idx) {
              kv_offset1 = block_table[nnext_block];
          }
          token_idx = next_block * BLOCK_SIZE + physical_block_offset;
          const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
          ptr_logits = logits + token_idx - start_token_idx;
          if(next_block == num_seq_blocks - 1) {
              #pragma unroll
              for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                  if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
                      const int idx = laneid * V_VEC_SIZE + i * block_offset;
#if 0
  		      if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          	          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
                      } else {
                          V_quant_vec v_quant_vec =
                              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
                          // Vector conversion from V_quant_vec to V_vec.
                          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    v_scale);
        	      }
#endif
	              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                      reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                  }
              }
          } else {
              #pragma unroll
              for(int i = 0; i < NUM_LGT_PER_COL; i++) {
                  if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
                      const int idx = laneid * V_VEC_SIZE + i * block_offset;
#if 0
  		      if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          	          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
                      } else {
                          V_quant_vec v_quant_vec =
                              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
                          // Vector conversion from V_quant_vec to V_vec.
                          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    v_scale);
        	      }
#endif
	              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
                      reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
                  }
              }
          }
      }
      token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      for(int i = 0; i < NUM_LGT_PER_COL; i++) {
          if(token_idx + i * NUM_COLS_PER_ITER < seq_len && i * NUM_COLS_PER_ITER + physical_block_offset < BLOCK_SIZE) {
              scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vecs[i]);
              for(int j = 0; j < V_VEC_SIZE; j++) {
                  accs[j] += atten_mul(v_vec_ptr, reg_log[i], j);
	      }
           }
        }
     }
  }
  __syncthreads();
  //need move
  float* out_smem = reinterpret_cast<float*>(shared_mem);
  for(int i = threadIdx.x; i < NUM_WARPS * NUM_COLS_PER_ITER * HEAD_SIZE; i += blockDim.x) {
    out_smem[i] = 0.0f;
  }
  __syncthreads(); 

  if(lane < NUM_VALID_THREAD) {
    float*ptr_out_smem = out_smem + warp_idx * HEAD_SIZE*NUM_COLS_PER_ITER + physical_block_offset * HEAD_SIZE + laneid* V_VEC_SIZE;
    for(int i = 0; i < V_VEC_SIZE; i++) {
      ptr_out_smem[i] = accs[i];
    }
  }
   __syncthreads();
  for(int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
    float r = 0;
    #pragma unroll
    for(int j = 0; j < NUM_WARPS * NUM_COLS_PER_ITER; j++){
        r += out_smem[j * HEAD_SIZE + i];
    }
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                        + head_idx * max_num_partitions * HEAD_SIZE
                        + partition_idx * HEAD_SIZE;
    from_float(*(out_ptr + i), r);
  }

}
/*
  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}
*/

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel_32N(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]->[num_blocks, num_kv_heads, head_size/16, block_size, 16]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]->[num_blocks, num_kv_heads, block_size, head_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }
  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;
  constexpr int THREAD_GROUP_SIZE = MAX(MXWARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
	      DIVIDE_ROUND_UP(BLOCK_SIZE, MXWARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / MXWARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / MXWARP_SIZE;
  const int lane = thread_idx % MXWARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs
  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 32 / sizeof(cache_t);      // VLLM_0.4.0  x=32
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
#if 0
  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }
#endif
  int block_idx0 = start_block_idx + warp_idx;
  int kv_offset0, kv_offset1;
  K_vec load_k[NUM_VECS_PER_THREAD];
  K_vec compute_k[NUM_VECS_PER_THREAD];
  kv_offset0 = block_table[block_idx0];
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  
  for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) % BLOCK_SIZE;
    const int token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
    const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                      + kv_head_idx * kv_head_stride
                                      + physical_block_offset * x;

#pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      const int vec_idx = (thread_group_offset + j * THREAD_GROUP_SIZE) * VEC_SIZE;
      const int offset1 = vec_idx / x;
      const int offset2 = vec_idx % x;
#if 0
      if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
	  load_k[j] = fp8_e5m2_unscaled::vec_conversion<K_vec, Quant_vec>(k_vec_quant);
        } else {
	  load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        }
#endif
      load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
    }
  }

   for (int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    for(int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * MXWARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        compute_k[j] = load_k[j];
      }
      if(block_idx < end_block_idx - NUM_WARPS) {
          kv_offset0 = kv_offset1;
          if(block_idx < end_block_idx - (NUM_WARPS << 1)) {
            kv_offset1 = block_table[block_idx + (NUM_WARPS<<1)];
          }

          const cache_t* k_ptr = k_cache + static_cast<int64_t>(kv_offset0) * kv_block_stride
                                      + kv_head_idx * kv_head_stride
                                      + physical_block_offset * x;
          #pragma unroll NUM_VECS_PER_THREAD
          for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
	      const int vec_idx = (thread_group_offset + j * THREAD_GROUP_SIZE) * VEC_SIZE;
              const int offset1 = vec_idx / x;
              const int offset2 = vec_idx % x;
#if 0
              if constexpr (IS_FP8_E5M2_KV_CACHE) {
      #ifdef ENABLE_FP8_E5M2
                Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
                // Vector conversion from Quant_vec to K_vec.
                load_k[j] = fp8_e5m2_unscaled::vec_conversion<K_vec, Quant_vec>(k_vec_quant);
      #else
                assert(false);
      #endif
              } else {
                load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
              }
#endif
                load_k[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          }
      }
      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      // Compute the parallel products for Q*K^T (treat vector lanes separately).
      float qk = 0.0f;
      #pragma unroll
      for(int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        scalar_t *ptr_q = (scalar_t*)&q_vecs[thread_group_offset][j];
        cache_t *ptr_c = (cache_t*)&compute_k[j];
        #pragma unroll
        for(int k = 0; k < VEC_SIZE; k++) {
          qk += atten_dot(ptr_q,ptr_c,k);
        }
      }
  
      #pragma unroll
      for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
	qk += MXVLLM_SHFL_XOR_SYNC(qk, mask);
      }
      qk = scale * qk;
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }
  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = MXWARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, MXVLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
   red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, MXVLLM_SHFL_XOR_SYNC(qk_max, mask));  // XuBW
  }
  // Broadcast the max qk value to all threads.
  qk_max = MXVLLM_SHFL_SYNC(qk_max, 0);  // XuBW

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = mxblock_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);  // XuBW

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }
  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int NUM_V_VECS_PER_THREAD = HEAD_SIZE / V_VEC_SIZE;
  constexpr int NUM_COLS_PER_ITER = MAX(MXWARP_SIZE / NUM_V_VECS_PER_THREAD,1);  // XuBW
  //constexpr int NUM_VALID_THREAD = NUM_COLS_PER_ITER * NUM_V_VECS_PER_THREAD;
  constexpr int NUM_LGT_PER_COL = BLOCK_SIZE / NUM_COLS_PER_ITER;
  //constexpr int NUM_LGT_PER_COL = (BLOCK_SIZE + NUM_COLS_PER_ITER - 1) / NUM_COLS_PER_ITER;
  constexpr int NUM_LANE = NUM_WARPS * NUM_COLS_PER_ITER; // XuBW
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  //using Float_L_vec = typename FloatVec<L_vec>::Type;
  const int physical_block_offset = lane / NUM_V_VECS_PER_THREAD;
  const int laneid = lane % NUM_V_VECS_PER_THREAD;
  V_vec v_vecs[NUM_LGT_PER_COL];
  V_vec v_prev_vecs[NUM_LGT_PER_COL];
  float accs[V_VEC_SIZE];
  float reg_log[NUM_LGT_PER_COL];
  float reg_prev_log[NUM_LGT_PER_COL];
  #pragma unroll
  for(int i = 0; i < V_VEC_SIZE; i++) {
    accs[i] = 0.0f;
  }
  int token_idx, kv_stride, block_offset;
  kv_stride = BLOCK_SIZE * HEAD_SIZE ;
  kv_offset0 = block_table[block_idx0];
  block_offset = NUM_COLS_PER_ITER * HEAD_SIZE;
  if(block_idx0 + NUM_WARPS < end_block_idx) {
    kv_offset1 = block_table[block_idx0 + NUM_WARPS];
  }
  token_idx = block_idx0 * BLOCK_SIZE + physical_block_offset;
  const cache_t *v_ptr0 = v_cache + kv_head_idx * kv_stride + physical_block_offset * HEAD_SIZE; 
  const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
  float *ptr_logits = logits + token_idx - start_token_idx;
  #pragma unroll
  for(int i = 0; i < NUM_LGT_PER_COL; i++) {
     // XuBW
     if(token_idx + i * NUM_COLS_PER_ITER < seq_len ) {
       const int idx = laneid * V_VEC_SIZE + i * block_offset;
       v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
       reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
     }
  } 

  for(int block_idx = block_idx0; block_idx < end_block_idx; block_idx += NUM_WARPS) {
      int next_block = block_idx + NUM_WARPS;
      int nnext_block = next_block + NUM_WARPS;
      for(int i = 0; i < NUM_LGT_PER_COL; i++) {
          v_vecs[i] = v_prev_vecs[i];
          reg_log[i] = reg_prev_log[i];
      }
      if(next_block < end_block_idx) {
        kv_offset0 = kv_offset1;
        if(nnext_block < end_block_idx) {
          kv_offset1 = block_table[nnext_block];
        }
        token_idx = next_block * BLOCK_SIZE + physical_block_offset;
        const cache_t* v_ptr = v_ptr0 + static_cast<int64_t>(kv_offset0) * kv_block_stride;
        ptr_logits = logits + token_idx - start_token_idx;
        if(next_block == num_seq_blocks - 1) {
          #pragma unroll
          for(int i = 0; i < NUM_LGT_PER_COL; i++) {
            if(token_idx + i * NUM_COLS_PER_ITER < seq_len) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
#if 0
              if constexpr (IS_FP8_E5M2_KV_CACHE) {
    #ifdef ENABLE_FP8_E5M2
            V_quant_vec v_quant_vec = *reinterpret_cast<const V_quant_vec*>(v_ptr + idx);
            // Vector conversion from V_quant_vec to V_vec.
            v_prev_vecs[i] = fp8_e5m2_unscaled::vec_conversion<V_vec, V_quant_vec>(v_quant_vec);
    #else
            assert(false);
    #endif
              } else {
                  v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              }
#endif
	      v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
            }
          }
        } else {
          #pragma unroll
          for(int i = 0; i < NUM_LGT_PER_COL; i++) {
              const int idx = laneid * V_VEC_SIZE + i * block_offset;
#if 0
              if constexpr (IS_FP8_E5M2_KV_CACHE) {
      #ifdef ENABLE_FP8_E5M2
              V_quant_vec v_quant_vec = *reinterpret_cast<const V_quant_vec*>(v_ptr + idx);
              // Vector conversion from V_quant_vec to V_vec.
              v_prev_vecs[i] = fp8_e5m2_unscaled::vec_conversion<V_vec, V_quant_vec>(v_quant_vec);
      #else
              assert(false);
      #endif
            } else {
              v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
            }
#endif
	      v_prev_vecs[i] = *reinterpret_cast<const V_vec*>(v_ptr + idx);
              reg_prev_log[i] = ptr_logits[i * NUM_COLS_PER_ITER];
            }
          }
        }

      token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      float *ptr_logits = logits + token_idx - start_token_idx;
      for(int i = 0; i < NUM_LGT_PER_COL; i++) {
        if(token_idx + i * NUM_COLS_PER_ITER < seq_len) {
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vecs[i]);
          for(int j = 0; j < V_VEC_SIZE; j++) {
            accs[j] += atten_mul(v_vec_ptr, reg_log[i], j);
          }
        }
      }
    }

  __syncthreads();
  //need move
  float* out_smem = reinterpret_cast<float*>(shared_mem);
  float*ptr_out_smem = out_smem + warp_idx * HEAD_SIZE*NUM_COLS_PER_ITER + physical_block_offset * HEAD_SIZE + laneid* V_VEC_SIZE;
  for(int i = 0; i < V_VEC_SIZE; i++) {
      ptr_out_smem[i] = accs[i];
  }
  __syncthreads();
 scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                        + head_idx * max_num_partitions * HEAD_SIZE
                        + partition_idx * HEAD_SIZE;

  for(int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
    float r = 0;
    #pragma unroll
    for(int j = 0; j < NUM_LANE; j++){
        r += out_smem[j * HEAD_SIZE + i];
    }
    from_float(*(out_ptr + i), r);
  }
}





// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float k_scale, const float v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, k_scale, v_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_32N_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  paged_attention_kernel_32N<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, kv_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}


// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float k_scale, const float v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride, k_scale, v_scale, tp_rank,
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_32N_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  paged_attention_kernel_32N<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride, kv_scale, tp_rank,
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}


// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums =
      reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums +
                              seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

}  // namespace vllm

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          k_scale, v_scale, tp_rank, blocksparse_local_blocks,              \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);

#define LAUNCH_PAGED_ATTENTION_V1_32N(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_32N_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_32N_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          k_scale, tp_rank, blocksparse_local_blocks,                      \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);


template< typename scalar_t>
__global__ void reshape_k_layout_new(scalar_t * __restrict__ k_buffer, scalar_t* k_output,int num_blocks,int num_kv_heads, int head_size,int block_size, int x,int dst_x) {
  int k_head_stride = head_size * block_size;
  scalar_t *ptr_k_buffer = k_buffer + blockIdx.x * k_head_stride;
  scalar_t *ptr_output = k_output + blockIdx.x * k_head_stride;
  for(int t = threadIdx.x; t < k_head_stride; t += blockDim.x) {
    int heightId = t / (block_size * dst_x);
    int remain = t % (block_size * dst_x);
    int blockId = remain / dst_x;
    int wId = remain % dst_x;
    int inId = heightId * dst_x + wId;
    int in_y = inId / x;
    int in_x = inId % x;
    int inIndex = in_y  * block_size * x + blockId * x + in_x;
    ptr_output[t] = ptr_k_buffer[inIndex];
  }
}

// [num_blocks, num_kv_heads, head_size, block_size] -->   [num_blocks,  num_kv_heads, block_size,head_size]
template<typename scalar_t>
__global__ void reshape_v_layout(scalar_t * __restrict__ v_buffer, scalar_t* v_output,int num_blocks,int num_kv_heads, int head_size,int block_size) {
      int v_block_stride = head_size * block_size * num_kv_heads;
      int v_head_stride = head_size * block_size;
      scalar_t *ptr_in = v_buffer + blockIdx.x * v_block_stride;
      scalar_t *ptr_output = v_output + blockIdx.x * v_block_stride;
      for(int t = threadIdx.x; t < v_block_stride; t += blockDim.x) {
        int num_kv_headIdx = t / v_head_stride;
        int remain = t % v_head_stride;
        int headId_H = remain / block_size;
        remain = remain % block_size;
        int out_idx = num_kv_headIdx * head_size * block_size + remain * head_size + headId_H;
        ptr_output[out_idx] = ptr_in[t];
      }
}

template<
  typename CACHE_T,
  int BLOCK_SIZE>
void reshape_kv_cache(
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& key_cache_new_layer,
  torch::Tensor& value_cache_new_layer,
  int num_seqs,
  int num_heads,
  int head_size,
  int num_kv_heads) {
  int kv_block_stride = key_cache.stride(0); // NU ,BLC ,HEAD, HEAD_DIM
  int kv_head_stride = key_cache.stride(1);

  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  CACHE_T* key_cache_tmp = reinterpret_cast<CACHE_T*>(key_cache_new_layer.data_ptr());
  CACHE_T* value_cache_tmp = reinterpret_cast<CACHE_T*>(value_cache_new_layer.data_ptr());

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  reshape_k_layout_new<CACHE_T><<<dim3(key_cache.size(0)*num_kv_heads,1,1),dim3(256,1,1),0,stream>>>(key_cache_ptr,key_cache_tmp,key_cache.size(0),num_kv_heads,head_size,BLOCK_SIZE,8,16);
  reshape_v_layout<CACHE_T><<<dim3(key_cache.size(0),1,1),dim3(256,1,1),0,stream>>>(value_cache_ptr,value_cache_tmp,key_cache.size(0),num_kv_heads,head_size,BLOCK_SIZE);
}

#define CALL_RESHAPE_LAUNCHER(CACHE_T, BLOCK_SIZE)       \
  reshape_kv_cache<CACHE_T, BLOCK_SIZE>( \
    key_cache,                                                               \
    value_cache,                                                             \
    key_cache_new_layer,                                                     \
    value_cache_new_layer,                                                   \
    num_seqs,\
    num_heads,\
    head_size,\
    num_kv_heads);

#define CALL_RESHAPE_BLOCK_SIZE(CACHE_T) \
  switch (block_size) {                                               \
    case 8:                                                           \
      CALL_RESHAPE_LAUNCHER(CACHE_T, 8);          \
      break;                                                          \
    case 16:                                                          \
      CALL_RESHAPE_LAUNCHER(CACHE_T, 16);         \
      break;                                                          \
    case 32:                                                          \
      CALL_RESHAPE_LAUNCHER(CACHE_T, 32);         \
      break;                                                          \
    default:                                                          \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);     \
      break;                                                          \
  }

void page_reshape_kv_cache(
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& key_cache_new_layer, //[num_blocks, num_heads, head_size/16, block_size, 16]
  torch::Tensor& value_cache_new_layer,//[num_blocks, num_heads, block_size, head_size]
  // XuBW int -> int64_t
  int num_seqs,
  int num_heads,
  int head_size,
  int num_kv_heads,               // [num_heads]
  int block_size,
  const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto") {
    if (sizeof(key_cache.dtype())==4) {
      //CALL_RESHAPE_BLOCK_SIZE(float);
    } else if (sizeof(key_cache.dtype()) == 2) {
      //CALL_RESHAPE_BLOCK_SIZE(uint16_t);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", key_cache.dtype());
    }
  }  else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
}



// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, torch::Tensor& max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float k_scale,
    float v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);  // num head head_dim
  int kv_block_stride = key_cache.stride(0); // NU ,BLC ,HEAD, HEAD_DIM
  int kv_head_stride = key_cache.stride(1);

  [[maybe_unused]] int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len =
      DIVIDE_ROUND_UP(512, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);

  int V_VEC_SIZE = 16 / sizeof(CACHE_T);
  int NUM_V_VECS_PER_THREAD = head_size / V_VEC_SIZE;
  int NUM_COLS_PER_ITER = MAX(WARP_SIZE / NUM_V_VECS_PER_THREAD, 1);
  int outputs_size = NUM_WARPS * head_size * sizeof(float) * NUM_COLS_PER_ITER;

  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1_32N(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V1(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1_32N(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1_32N(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)  \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,              \
                              IS_BLOCK_SPARSE>(                              \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes, k_scale, v_scale, tp_rank,        \
      blocksparse_local_blocks, blocksparse_vert_stride,                     \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  switch (is_block_sparse) {                                               \
    case true:                                                             \
      CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);     \
      break;                                                               \
    case false:                                                            \
      CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);    \
      break;                                                               \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 8:                                                       \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);         \
      break;                                                      \
    case 16:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    case 32:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

void paged_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, torch::Tensor& max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V1_LAUNCHER_BLOCK_SIZE)
}

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                   \
  vllm::paged_attention_v2_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,           \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE,      \
                                  PARTITION_SIZE>                              \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr, \
          value_cache_ptr, num_kv_heads, scale, block_tables_ptr,              \
          seq_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,    \
          kv_block_stride, kv_head_stride, k_scale, v_scale, tp_rank,          \
          blocksparse_local_blocks, blocksparse_vert_stride,                   \
          blocksparse_block_size, blocksparse_head_sliding_step);              \
  vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS,            \
                                         PARTITION_SIZE>                       \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr,    \
          max_num_partitions);

#define LAUNCH_PAGED_ATTENTION_V2_32N(HEAD_SIZE)                                   \
  vllm::paged_attention_v2_32N_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,           \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE,      \
                                  PARTITION_SIZE>                              \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr, \
          value_cache_ptr, num_kv_heads, scale, block_tables_ptr,              \
          seq_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,    \
          kv_block_stride, kv_head_stride, k_scale, tp_rank,                  \
          blocksparse_local_blocks, blocksparse_vert_stride,                   \
          blocksparse_block_size, blocksparse_head_sliding_step);              \
  vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS,            \
                                         PARTITION_SIZE>                       \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr,    \
          max_num_partitions);


template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float k_scale,
    float v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  [[maybe_unused]] int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int V_VEC_SIZE = 16 / sizeof(CACHE_T);
  int NUM_V_VECS_PER_THREAD = head_size / V_VEC_SIZE;
  int NUM_COLS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_THREAD;
  int outputs_size = NUM_WARPS * head_size * sizeof(float) * NUM_COLS_PER_ITER;


  // For paged attention v2 kernel.
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V2_32N(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V2(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2_32N(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V2(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2_32N(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)   \
  paged_attention_v2_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,               \
                              IS_BLOCK_SPARSE>(                               \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,      \
      num_kv_heads, scale, block_tables, seq_lens, max_seq_len, alibi_slopes, \
      k_scale, v_scale, tp_rank, blocksparse_local_blocks,                    \
      blocksparse_vert_stride, blocksparse_block_size,                        \
      blocksparse_head_sliding_step);

#define CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  switch (is_block_sparse) {                                               \
    case true:                                                             \
      CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);     \
      break;                                                               \
    case false:                                                            \
      CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);    \
      break;                                                               \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 8:                                                       \
      CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);         \
      break;                                                      \
    case 16:                                                      \
      CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    case 32:                                                      \
      CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

void paged_attention_v2(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor&
        tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);
  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V2_LAUNCHER_BLOCK_SIZE)
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
