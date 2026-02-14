// SPDX-License-Identifier: BSD-3-Clause
// CUDA Argon2id lane-fill kernel specialized for Seine's PoW profile.

extern "C" {

constexpr unsigned int ARGON2_COOP_THREADS = 32U;
constexpr unsigned int ARGON2_COOP_WARP_MASK = 0xFFFFFFFFU;
#ifndef SEINE_FIXED_M_BLOCKS
#define SEINE_FIXED_M_BLOCKS 0U
#endif
#ifndef SEINE_FIXED_T_COST
#define SEINE_FIXED_T_COST 0U
#endif

__device__ __forceinline__ void coop_sync() {
    __syncwarp(ARGON2_COOP_WARP_MASK);
}

__device__ __forceinline__ unsigned long long blamka(
    unsigned long long x,
    unsigned long long y
) {
    const unsigned long long low = 0xFFFFFFFFULL;
    return x + y + 2ULL * (x & low) * (y & low);
}

__device__ __forceinline__ void permute_step(
    unsigned long long &a,
    unsigned long long &b,
    unsigned long long &c,
    unsigned long long &d
) {
    a = blamka(a, b);
    d = ((d ^ a) >> 32) | ((d ^ a) << (64 - 32));
    c = blamka(c, d);
    b = ((b ^ c) >> 24) | ((b ^ c) << (64 - 24));

    a = blamka(a, b);
    d = ((d ^ a) >> 16) | ((d ^ a) << (64 - 16));
    c = blamka(c, d);
    b = ((b ^ c) >> 63) | ((b ^ c) << 1);
}

__device__ __forceinline__ void permute_round(
    unsigned long long &v0,
    unsigned long long &v1,
    unsigned long long &v2,
    unsigned long long &v3,
    unsigned long long &v4,
    unsigned long long &v5,
    unsigned long long &v6,
    unsigned long long &v7,
    unsigned long long &v8,
    unsigned long long &v9,
    unsigned long long &v10,
    unsigned long long &v11,
    unsigned long long &v12,
    unsigned long long &v13,
    unsigned long long &v14,
    unsigned long long &v15
) {
    permute_step(v0, v4, v8, v12);
    permute_step(v1, v5, v9, v13);
    permute_step(v2, v6, v10, v14);
    permute_step(v3, v7, v11, v15);
    permute_step(v0, v5, v10, v15);
    permute_step(v1, v6, v11, v12);
    permute_step(v2, v7, v8, v13);
    permute_step(v3, v4, v9, v14);
}

__device__ __forceinline__ void compress_block_coop(
    const unsigned long long *__restrict__ rhs,
    const unsigned long long *__restrict__ lhs,
    unsigned long long *__restrict__ out,
    bool xor_out,
    unsigned long long *__restrict__ scratch_r,
    unsigned long long *__restrict__ scratch_q,
    unsigned int tid
) {
    for (unsigned int i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        const unsigned long long r = rhs[i] ^ lhs[i];
        scratch_r[i] = r;
        scratch_q[i] = r;
    }
    coop_sync();

    // Use all 32 threads: 8 independent states x 4 threads/state.
    {
        const unsigned int state = tid >> 2; // 0..7
        const unsigned int lane = tid & 3U;  // 0..3
        const unsigned int base = state * 16U;

        unsigned int i0 = 0U;
        unsigned int i1 = 0U;
        unsigned int i2 = 0U;
        unsigned int i3 = 0U;
        switch (lane) {
            case 0U:
                i0 = base + 0U;
                i1 = base + 4U;
                i2 = base + 8U;
                i3 = base + 12U;
                break;
            case 1U:
                i0 = base + 1U;
                i1 = base + 5U;
                i2 = base + 9U;
                i3 = base + 13U;
                break;
            case 2U:
                i0 = base + 2U;
                i1 = base + 6U;
                i2 = base + 10U;
                i3 = base + 14U;
                break;
            default:
                i0 = base + 3U;
                i1 = base + 7U;
                i2 = base + 11U;
                i3 = base + 15U;
                break;
        }

        unsigned long long a = scratch_q[i0];
        unsigned long long b = scratch_q[i1];
        unsigned long long c = scratch_q[i2];
        unsigned long long d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a;
        scratch_q[i1] = b;
        scratch_q[i2] = c;
        scratch_q[i3] = d;
    }
    coop_sync();

    {
        const unsigned int state = tid >> 2; // 0..7
        const unsigned int lane = tid & 3U;  // 0..3
        const unsigned int base = state * 16U;

        unsigned int i0 = 0U;
        unsigned int i1 = 0U;
        unsigned int i2 = 0U;
        unsigned int i3 = 0U;
        switch (lane) {
            case 0U:
                i0 = base + 0U;
                i1 = base + 5U;
                i2 = base + 10U;
                i3 = base + 15U;
                break;
            case 1U:
                i0 = base + 1U;
                i1 = base + 6U;
                i2 = base + 11U;
                i3 = base + 12U;
                break;
            case 2U:
                i0 = base + 2U;
                i1 = base + 7U;
                i2 = base + 8U;
                i3 = base + 13U;
                break;
            default:
                i0 = base + 3U;
                i1 = base + 4U;
                i2 = base + 9U;
                i3 = base + 14U;
                break;
        }

        unsigned long long a = scratch_q[i0];
        unsigned long long b = scratch_q[i1];
        unsigned long long c = scratch_q[i2];
        unsigned long long d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a;
        scratch_q[i1] = b;
        scratch_q[i2] = c;
        scratch_q[i3] = d;
    }
    coop_sync();

    // Column round: 8 independent interleaved states x 4 threads/state.
    {
        const unsigned int state = tid >> 2; // 0..7
        const unsigned int lane = tid & 3U;  // 0..3
        const unsigned int b = state * 2U;

        unsigned int i0 = 0U;
        unsigned int i1 = 0U;
        unsigned int i2 = 0U;
        unsigned int i3 = 0U;
        switch (lane) {
            case 0U:
                i0 = b;
                i1 = b + 32U;
                i2 = b + 64U;
                i3 = b + 96U;
                break;
            case 1U:
                i0 = b + 1U;
                i1 = b + 33U;
                i2 = b + 65U;
                i3 = b + 97U;
                break;
            case 2U:
                i0 = b + 16U;
                i1 = b + 48U;
                i2 = b + 80U;
                i3 = b + 112U;
                break;
            default:
                i0 = b + 17U;
                i1 = b + 49U;
                i2 = b + 81U;
                i3 = b + 113U;
                break;
        }

        unsigned long long a = scratch_q[i0];
        unsigned long long c = scratch_q[i1];
        unsigned long long d = scratch_q[i2];
        unsigned long long e = scratch_q[i3];
        permute_step(a, c, d, e);
        scratch_q[i0] = a;
        scratch_q[i1] = c;
        scratch_q[i2] = d;
        scratch_q[i3] = e;
    }
    coop_sync();

    {
        const unsigned int state = tid >> 2; // 0..7
        const unsigned int lane = tid & 3U;  // 0..3
        const unsigned int b = state * 2U;

        unsigned int i0 = 0U;
        unsigned int i1 = 0U;
        unsigned int i2 = 0U;
        unsigned int i3 = 0U;
        switch (lane) {
            case 0U:
                i0 = b;
                i1 = b + 33U;
                i2 = b + 80U;
                i3 = b + 113U;
                break;
            case 1U:
                i0 = b + 1U;
                i1 = b + 48U;
                i2 = b + 81U;
                i3 = b + 96U;
                break;
            case 2U:
                i0 = b + 16U;
                i1 = b + 49U;
                i2 = b + 64U;
                i3 = b + 97U;
                break;
            default:
                i0 = b + 17U;
                i1 = b + 32U;
                i2 = b + 65U;
                i3 = b + 112U;
                break;
        }

        unsigned long long a = scratch_q[i0];
        unsigned long long c = scratch_q[i1];
        unsigned long long d = scratch_q[i2];
        unsigned long long e = scratch_q[i3];
        permute_step(a, c, d, e);
        scratch_q[i0] = a;
        scratch_q[i1] = c;
        scratch_q[i2] = d;
        scratch_q[i3] = e;
    }
    coop_sync();

    for (unsigned int i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        const unsigned long long mixed = scratch_q[i] ^ scratch_r[i];
        if (xor_out) {
            out[i] ^= mixed;
        } else {
            out[i] = mixed;
        }
    }
}

__device__ __forceinline__ void update_address_block_coop(
    unsigned long long *__restrict__ address_block,
    unsigned long long *__restrict__ input_block,
    const unsigned long long *__restrict__ zero_block,
    unsigned long long *__restrict__ scratch_r,
    unsigned long long *__restrict__ scratch_q,
    unsigned int tid
) {
    if (tid == 0U) {
        input_block[6] += 1ULL;
    }
    coop_sync();
    compress_block_coop(
        zero_block,
        input_block,
        address_block,
        false,
        scratch_r,
        scratch_q,
        tid
    );
    compress_block_coop(
        zero_block,
        address_block,
        address_block,
        false,
        scratch_r,
        scratch_q,
        tid
    );
}

__global__ void touch_lane_memory_kernel(
    unsigned long long *lane_memory,
    unsigned int lanes_active,
    unsigned long long lane_stride_words
) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes_active || lane_stride_words == 0ULL) {
        return;
    }

    volatile unsigned long long *lane_ptr =
        lane_memory + static_cast<unsigned long long>(lane) * lane_stride_words;

    // Touch one 4 KiB page at a time (512 x u64) to force physical commit.
    for (unsigned long long word = 0ULL; word < lane_stride_words; word += 512ULL) {
        unsigned long long v = lane_ptr[word];
        lane_ptr[word] = v;
    }

    const unsigned long long tail = lane_stride_words - 1ULL;
    unsigned long long tail_v = lane_ptr[tail];
    lane_ptr[tail] = tail_v;
}

__global__ void argon2id_fill_kernel(
    const unsigned long long *__restrict__ seed_blocks,
    unsigned int lanes_active,
    unsigned int active_hashes,
    unsigned int lane_launch_iters,
    unsigned int m_blocks,
    unsigned int t_cost,
    unsigned long long *__restrict__ lane_memory,
    unsigned long long *__restrict__ out_last_blocks
) {
    const unsigned int lane = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int effective_m_blocks =
        (SEINE_FIXED_M_BLOCKS > 0U) ? SEINE_FIXED_M_BLOCKS : m_blocks;
    const unsigned int effective_t_cost =
        (SEINE_FIXED_T_COST > 0U) ? SEINE_FIXED_T_COST : t_cost;
    if (lane >= lanes_active || effective_m_blocks < 8U) {
        return;
    }

    const unsigned long long lane_stride_words =
        static_cast<unsigned long long>(effective_m_blocks) * 128ULL;
    unsigned long long *memory =
        lane_memory + static_cast<unsigned long long>(lane) * lane_stride_words;
    const unsigned int segment_length = effective_m_blocks / 4U;
    const unsigned int lane_length = effective_m_blocks;

    __shared__ unsigned long long address_block[128];
    __shared__ unsigned long long input_block[128];
    __shared__ unsigned long long zero_block[128];
    __shared__ unsigned long long scratch_r[128];
    __shared__ unsigned long long scratch_q[128];

    for (unsigned int iter = 0U; iter < lane_launch_iters; ++iter) {
        const unsigned long long global_hash_idx =
            static_cast<unsigned long long>(iter) * lanes_active +
            static_cast<unsigned long long>(lane);
        if (global_hash_idx >= static_cast<unsigned long long>(active_hashes)) {
            break;
        }

        const unsigned long long *seed = seed_blocks + global_hash_idx * 256ULL;
        for (unsigned int i = tid; i < 256U; i += ARGON2_COOP_THREADS) {
            memory[i] = seed[i];
        }
        coop_sync();

        for (unsigned int i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
            address_block[i] = 0ULL;
            input_block[i] = 0ULL;
            zero_block[i] = 0ULL;
        }
        coop_sync();

        for (unsigned int pass = 0; pass < effective_t_cost; ++pass) {
            for (unsigned int slice = 0; slice < 4U; ++slice) {
                const bool data_independent = (pass == 0U && slice < 2U);
                if (data_independent) {
                    for (unsigned int i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
                        address_block[i] = 0ULL;
                        input_block[i] = 0ULL;
                    }
                    if (tid == 0U) {
                        input_block[0] = static_cast<unsigned long long>(pass);
                        input_block[1] = 0ULL; // lane index (p=1)
                        input_block[2] = static_cast<unsigned long long>(slice);
                        input_block[3] = static_cast<unsigned long long>(effective_m_blocks);
                        input_block[4] = static_cast<unsigned long long>(effective_t_cost);
                        input_block[5] = 2ULL; // Argon2id
                    }
                    coop_sync();
                }

                const unsigned int first_block =
                    (pass == 0U && slice == 0U) ? 2U : 0U;

                unsigned int cur_index =
                    slice * segment_length + first_block;
                unsigned int prev_index =
                    (slice == 0U && first_block == 0U)
                        ? (cur_index + lane_length - 1U)
                        : (cur_index - 1U);

                for (unsigned int block = first_block; block < segment_length; ++block) {
                    unsigned long long rand64;
                    if (data_independent) {
                        const unsigned int address_index = block & 127U;
                        if (address_index == 0U) {
                            update_address_block_coop(
                                address_block,
                                input_block,
                                zero_block,
                                scratch_r,
                                scratch_q,
                                tid
                            );
                        }
                        coop_sync();
                        rand64 = address_block[address_index];
                    } else {
                        rand64 = memory[static_cast<unsigned long long>(prev_index) * 128ULL];
                    }

                    unsigned long long reference_area_size;
                    if (pass == 0U) {
                        if (slice == 0U) {
                            reference_area_size = static_cast<unsigned long long>(block - 1U);
                        } else {
                            reference_area_size =
                                static_cast<unsigned long long>(slice) * segment_length +
                                static_cast<unsigned long long>(block) - 1ULL;
                        }
                    } else {
                        reference_area_size =
                            static_cast<unsigned long long>(lane_length - segment_length) +
                            static_cast<unsigned long long>(block) - 1ULL;
                    }

                    unsigned long long map = rand64 & 0xFFFFFFFFULL;
                    map = (map * map) >> 32;
                    const unsigned long long relative_position =
                        reference_area_size - 1ULL -
                        ((reference_area_size * map) >> 32);

                    const unsigned int start_position =
                        (pass != 0U && slice != 3U) ? ((slice + 1U) * segment_length) : 0U;
                    const unsigned int ref_index =
                        static_cast<unsigned int>(
                            (static_cast<unsigned long long>(start_position) +
                             relative_position) %
                            static_cast<unsigned long long>(lane_length)
                        );

                    const unsigned long long dst_offset =
                        static_cast<unsigned long long>(cur_index) * 128ULL;
                    const unsigned long long rhs_offset =
                        static_cast<unsigned long long>(prev_index) * 128ULL;
                    const unsigned long long lhs_offset =
                        static_cast<unsigned long long>(ref_index) * 128ULL;
                    if (pass == 0U) {
                        compress_block_coop(
                            memory + rhs_offset,
                            memory + lhs_offset,
                            memory + dst_offset,
                            false,
                            scratch_r,
                            scratch_q,
                            tid
                        );
                    } else {
                        compress_block_coop(
                            memory + rhs_offset,
                            memory + lhs_offset,
                            memory + dst_offset,
                            true,
                            scratch_r,
                            scratch_q,
                            tid
                        );
                    }
                    coop_sync();

                    prev_index = cur_index;
                    cur_index += 1U;
                }
            }
        }

        const unsigned long long *last =
            memory + (static_cast<unsigned long long>(lane_length) - 1ULL) * 128ULL;
        unsigned long long *out = out_last_blocks + global_hash_idx * 128ULL;
        for (unsigned int i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
            out[i] = last[i];
        }
        coop_sync();
    }
}

} // extern "C"
