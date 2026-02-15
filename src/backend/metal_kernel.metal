// SPDX-License-Identifier: BSD-3-Clause
// Metal Argon2id lane-fill kernel specialized for Seine's PoW profile.
// Ported from nvidia_kernel.cu.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint ARGON2_COOP_THREADS = 32U;
constant uint SEED_KERNEL_THREADS = 64U;
constant uint EVAL_KERNEL_THREADS = 64U;
constant uint CANCEL_CHECK_BLOCK_INTERVAL = 64U;
constant uint BLAKE2B_BLOCK_BYTES = 128U;
constant uint BLAKE2B_OUT_BYTES = 64U;
constant uint POW_OUTPUT_BYTES = 32U;
constant uint ARGON2_VERSION_V13 = 0x13U;
constant uint ARGON2_ALGORITHM_ID = 2U;

constant uint SEINE_M_BLOCKS [[function_constant(0)]];
constant uint SEINE_T_COST [[function_constant(1)]];

constant ulong BLAKE2B_IV[8] = {
    0x6A09E667F3BCC908ULL,
    0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL,
    0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL,
    0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL,
    0x5BE0CD19137E2179ULL,
};

constant uchar BLAKE2B_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

// ---------- utility helpers ----------

inline void store_u32_le(thread uchar *dst, uint value) {
    dst[0] = uchar(value & 0xFFU);
    dst[1] = uchar((value >> 8) & 0xFFU);
    dst[2] = uchar((value >> 16) & 0xFFU);
    dst[3] = uchar((value >> 24) & 0xFFU);
}

inline void store_u64_le(thread uchar *dst, ulong value) {
    for (uint i = 0U; i < 8U; ++i) {
        dst[i] = uchar((value >> (8U * i)) & 0xFFULL);
    }
}

inline ulong load_u64_le(thread const uchar *src) {
    ulong value = 0ULL;
    for (uint i = 0U; i < 8U; ++i) {
        value |= ulong(src[i]) << (8U * i);
    }
    return value;
}

inline ulong load_u64_le_device(device const uchar *src) {
    ulong value = 0ULL;
    for (uint i = 0U; i < 8U; ++i) {
        value |= ulong(src[i]) << (8U * i);
    }
    return value;
}

inline ulong rotr64(ulong x, uint n) {
    return (x >> n) | (x << (64U - n));
}

inline bool hash_meets_target_be(thread const uchar *hash, device const uchar *target) {
    for (uint i = 0U; i < POW_OUTPUT_BYTES; ++i) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;
}

// ---------- Blake2b ----------

struct Blake2bState {
    ulong h[8];
    ulong t0;
    ulong t1;
    uchar buf[128];
    uint buflen;
};

inline void blake2b_g(thread ulong v[16], uint a, uint b, uint c, uint d, ulong x, ulong y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr64(v[d] ^ v[a], 32U);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 24U);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr64(v[d] ^ v[a], 16U);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 63U);
}

inline void blake2b_compress(thread Blake2bState &state, thread const uchar *block, ulong last_block_flag) {
    ulong m[16];
    ulong v[16];
    for (uint i = 0U; i < 16U; ++i) {
        m[i] = load_u64_le(block + (i * 8U));
    }
    for (uint i = 0U; i < 8U; ++i) {
        v[i] = state.h[i];
        v[i + 8U] = BLAKE2B_IV[i];
    }
    v[12] ^= state.t0;
    v[13] ^= state.t1;
    v[14] ^= last_block_flag;

    for (uint round = 0U; round < 12U; ++round) {
        constant const uchar *s = BLAKE2B_SIGMA[round];
        blake2b_g(v, 0U, 4U, 8U, 12U, m[s[0]], m[s[1]]);
        blake2b_g(v, 1U, 5U, 9U, 13U, m[s[2]], m[s[3]]);
        blake2b_g(v, 2U, 6U, 10U, 14U, m[s[4]], m[s[5]]);
        blake2b_g(v, 3U, 7U, 11U, 15U, m[s[6]], m[s[7]]);
        blake2b_g(v, 0U, 5U, 10U, 15U, m[s[8]], m[s[9]]);
        blake2b_g(v, 1U, 6U, 11U, 12U, m[s[10]], m[s[11]]);
        blake2b_g(v, 2U, 7U, 8U, 13U, m[s[12]], m[s[13]]);
        blake2b_g(v, 3U, 4U, 9U, 14U, m[s[14]], m[s[15]]);
    }

    for (uint i = 0U; i < 8U; ++i) {
        state.h[i] ^= v[i] ^ v[i + 8U];
    }
}

inline void blake2b_init(thread Blake2bState &state, uint out_len) {
    for (uint i = 0U; i < 8U; ++i) {
        state.h[i] = BLAKE2B_IV[i];
    }
    state.h[0] ^= 0x01010000ULL ^ ulong(out_len);
    state.t0 = 0ULL;
    state.t1 = 0ULL;
    state.buflen = 0U;
    for (uint i = 0U; i < BLAKE2B_BLOCK_BYTES; ++i) {
        state.buf[i] = 0U;
    }
}

inline void blake2b_increment_counter(thread Blake2bState &state, uint increment) {
    ulong prev = state.t0;
    state.t0 += ulong(increment);
    if (state.t0 < prev) {
        state.t1 += 1ULL;
    }
}

inline void blake2b_update(thread Blake2bState &state, thread const uchar *input, uint input_len) {
    uint offset = 0U;
    while (offset < input_len) {
        uint space = BLAKE2B_BLOCK_BYTES - state.buflen;
        uint take = (input_len - offset < space) ? (input_len - offset) : space;
        for (uint i = 0U; i < take; ++i) {
            state.buf[state.buflen + i] = input[offset + i];
        }
        state.buflen += take;
        offset += take;

        if (state.buflen == BLAKE2B_BLOCK_BYTES) {
            blake2b_increment_counter(state, BLAKE2B_BLOCK_BYTES);
            blake2b_compress(state, state.buf, 0ULL);
            state.buflen = 0U;
        }
    }
}

inline void blake2b_update_device(thread Blake2bState &state, device const uchar *input, uint input_len) {
    uint offset = 0U;
    while (offset < input_len) {
        uint space = BLAKE2B_BLOCK_BYTES - state.buflen;
        uint take = (input_len - offset < space) ? (input_len - offset) : space;
        for (uint i = 0U; i < take; ++i) {
            state.buf[state.buflen + i] = input[offset + i];
        }
        state.buflen += take;
        offset += take;

        if (state.buflen == BLAKE2B_BLOCK_BYTES) {
            blake2b_increment_counter(state, BLAKE2B_BLOCK_BYTES);
            blake2b_compress(state, state.buf, 0ULL);
            state.buflen = 0U;
        }
    }
}

inline void blake2b_final(thread Blake2bState &state, thread uchar *out, uint out_len) {
    blake2b_increment_counter(state, state.buflen);
    for (uint i = state.buflen; i < BLAKE2B_BLOCK_BYTES; ++i) {
        state.buf[i] = 0U;
    }
    blake2b_compress(state, state.buf, 0xFFFFFFFFFFFFFFFFULL);

    uchar digest[BLAKE2B_OUT_BYTES];
    for (uint i = 0U; i < 8U; ++i) {
        store_u64_le(digest + (i * 8U), state.h[i]);
    }
    for (uint i = 0U; i < out_len; ++i) {
        out[i] = digest[i];
    }
}

inline void blake2b_hash(thread const uchar *input, uint input_len, thread uchar *out, uint out_len) {
    Blake2bState state;
    blake2b_init(state, out_len);
    blake2b_update(state, input, input_len);
    blake2b_final(state, out, out_len);
}

inline void blake2b_long_1024(thread const uchar *input, uint input_len, thread uchar *out) {
    uchar out_len_le[4];
    store_u32_le(out_len_le, 1024U);

    uchar last_output[64];
    Blake2bState state;
    blake2b_init(state, 64U);
    blake2b_update(state, out_len_le, 4U);
    blake2b_update(state, input, input_len);
    blake2b_final(state, last_output, 64U);

    uint counter = 0U;
    for (uint i = 0U; i < 32U; ++i) {
        out[counter + i] = last_output[i];
    }
    counter += 32U;

    while ((1024U - counter) > 64U) {
        uchar next_output[64];
        blake2b_hash(last_output, 64U, next_output, 64U);
        for (uint i = 0U; i < 64U; ++i) {
            last_output[i] = next_output[i];
        }
        for (uint i = 0U; i < 32U; ++i) {
            out[counter + i] = last_output[i];
        }
        counter += 32U;
    }

    uint tail_len = 1024U - counter;
    blake2b_hash(last_output, 64U, out + counter, tail_len);
}

inline void blake2b_long_32_from_words(device const ulong *block_words, thread uchar *out_hash) {
    uchar out_len_le[4];
    store_u32_le(out_len_le, POW_OUTPUT_BYTES);

    Blake2bState state;
    blake2b_init(state, POW_OUTPUT_BYTES);
    blake2b_update(state, out_len_le, 4U);

    uchar chunk[128];
    for (uint chunk_idx = 0U; chunk_idx < 8U; ++chunk_idx) {
        device const ulong *chunk_words = block_words + chunk_idx * 16U;
        for (uint i = 0U; i < 16U; ++i) {
            store_u64_le(chunk + i * 8U, chunk_words[i]);
        }
        blake2b_update(state, chunk, 128U);
    }
    blake2b_final(state, out_hash, POW_OUTPUT_BYTES);
}

// ---------- Kernel 1: build_seed_blocks ----------

kernel void build_seed_blocks_kernel(
    device const uchar  *header_base          [[buffer(0)]],
    device const ulong  *nonces               [[buffer(1)]],
    device ulong        *seed_blocks          [[buffer(2)]],
    constant uint       &header_base_len      [[buffer(3)]],
    constant uint       &active_hashes        [[buffer(4)]],
    constant uint       &m_cost_kib           [[buffer(5)]],
    constant uint       &t_cost               [[buffer(6)]],
    uint tid_global [[thread_position_in_grid]]
) {
    uint hash_idx = tid_global;
    if (hash_idx >= active_hashes) {
        return;
    }

    uchar h0[64];
    {
        uchar tmp4[4];
        uchar tmp8[8];
        Blake2bState state;
        blake2b_init(state, 64U);

        store_u32_le(tmp4, 1U);            // parallelism = 1
        blake2b_update(state, tmp4, 4U);

        store_u32_le(tmp4, POW_OUTPUT_BYTES); // output length = 32
        blake2b_update(state, tmp4, 4U);

        store_u32_le(tmp4, m_cost_kib);
        blake2b_update(state, tmp4, 4U);

        store_u32_le(tmp4, t_cost);
        blake2b_update(state, tmp4, 4U);

        store_u32_le(tmp4, ARGON2_VERSION_V13);
        blake2b_update(state, tmp4, 4U);

        store_u32_le(tmp4, ARGON2_ALGORITHM_ID);
        blake2b_update(state, tmp4, 4U);

        // Password: 8-byte nonce
        store_u32_le(tmp4, 8U);            // password length
        blake2b_update(state, tmp4, 4U);
        store_u64_le(tmp8, nonces[hash_idx]);
        blake2b_update(state, tmp8, 8U);

        // Salt: header_base
        store_u32_le(tmp4, header_base_len);
        blake2b_update(state, tmp4, 4U);
        blake2b_update_device(state, header_base, header_base_len);

        // Secret and associated data: empty
        store_u32_le(tmp4, 0U);
        blake2b_update(state, tmp4, 4U);
        blake2b_update(state, tmp4, 4U);

        blake2b_final(state, h0, 64U);
    }

    uchar seed_input[72];
    for (uint i = 0U; i < 64U; ++i) {
        seed_input[i] = h0[i];
    }
    // Lane index for p=1 is always zero.
    seed_input[68] = 0U;
    seed_input[69] = 0U;
    seed_input[70] = 0U;
    seed_input[71] = 0U;

    uchar seed_bytes[1024];
    for (uint seed_idx = 0U; seed_idx < 2U; ++seed_idx) {
        store_u32_le(seed_input + 64U, seed_idx);
        blake2b_long_1024(seed_input, 72U, seed_bytes);
        device ulong *out_words =
            seed_blocks + ulong(hash_idx) * 256ULL +
            ulong(seed_idx) * 128ULL;
        for (uint word = 0U; word < 128U; ++word) {
            out_words[word] = load_u64_le(seed_bytes + word * 8U);
        }
    }
}

// ---------- Argon2 compress primitives ----------

inline ulong blamka(ulong x, ulong y) {
    ulong low = 0xFFFFFFFFULL;
    return x + y + 2ULL * (x & low) * (y & low);
}

inline void permute_step(thread ulong &a, thread ulong &b, thread ulong &c, thread ulong &d) {
    a = blamka(a, b);
    d = ((d ^ a) >> 32) | ((d ^ a) << 32);
    c = blamka(c, d);
    b = ((b ^ c) >> 24) | ((b ^ c) << 40);

    a = blamka(a, b);
    d = ((d ^ a) >> 16) | ((d ^ a) << 48);
    c = blamka(c, d);
    b = ((b ^ c) >> 63) | ((b ^ c) << 1);
}

inline void compress_block_coop(
    device const ulong *rhs,
    device const ulong *lhs,
    device ulong *out,
    bool xor_out,
    threadgroup ulong *scratch_r,
    threadgroup ulong *scratch_q,
    uint tid
) {
    for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        ulong r = rhs[i] ^ lhs[i];
        scratch_r[i] = r;
        scratch_q[i] = r;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Row round 1: 8 independent states x 4 threads/state (columns)
    {
        uint state = tid >> 2;
        uint lane = tid & 3U;
        uint base = state * 16U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = base + 0U;  i1 = base + 4U;  i2 = base + 8U;  i3 = base + 12U; break;
            case 1U: i0 = base + 1U;  i1 = base + 5U;  i2 = base + 9U;  i3 = base + 13U; break;
            case 2U: i0 = base + 2U;  i1 = base + 6U;  i2 = base + 10U; i3 = base + 14U; break;
            default: i0 = base + 3U;  i1 = base + 7U;  i2 = base + 11U; i3 = base + 15U; break;
        }
        ulong a = scratch_q[i0], b = scratch_q[i1], c = scratch_q[i2], d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a; scratch_q[i1] = b; scratch_q[i2] = c; scratch_q[i3] = d;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Row round 2: diagonals
    {
        uint state = tid >> 2;
        uint lane = tid & 3U;
        uint base = state * 16U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = base + 0U;  i1 = base + 5U;  i2 = base + 10U; i3 = base + 15U; break;
            case 1U: i0 = base + 1U;  i1 = base + 6U;  i2 = base + 11U; i3 = base + 12U; break;
            case 2U: i0 = base + 2U;  i1 = base + 7U;  i2 = base + 8U;  i3 = base + 13U; break;
            default: i0 = base + 3U;  i1 = base + 4U;  i2 = base + 9U;  i3 = base + 14U; break;
        }
        ulong a = scratch_q[i0], b = scratch_q[i1], c = scratch_q[i2], d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a; scratch_q[i1] = b; scratch_q[i2] = c; scratch_q[i3] = d;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Column round 1: interleaved
    {
        uint state = tid >> 2;
        uint lane = tid & 3U;
        uint b = state * 2U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = b;       i1 = b + 32U;  i2 = b + 64U;  i3 = b + 96U;  break;
            case 1U: i0 = b + 1U;  i1 = b + 33U;  i2 = b + 65U;  i3 = b + 97U;  break;
            case 2U: i0 = b + 16U; i1 = b + 48U;  i2 = b + 80U;  i3 = b + 112U; break;
            default: i0 = b + 17U; i1 = b + 49U;  i2 = b + 81U;  i3 = b + 113U; break;
        }
        ulong a2 = scratch_q[i0], c2 = scratch_q[i1], d2 = scratch_q[i2], e2 = scratch_q[i3];
        permute_step(a2, c2, d2, e2);
        scratch_q[i0] = a2; scratch_q[i1] = c2; scratch_q[i2] = d2; scratch_q[i3] = e2;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Column round 2: interleaved diagonals
    {
        uint state = tid >> 2;
        uint lane = tid & 3U;
        uint b = state * 2U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = b;       i1 = b + 33U;  i2 = b + 80U;  i3 = b + 113U; break;
            case 1U: i0 = b + 1U;  i1 = b + 48U;  i2 = b + 81U;  i3 = b + 96U;  break;
            case 2U: i0 = b + 16U; i1 = b + 49U;  i2 = b + 64U;  i3 = b + 97U;  break;
            default: i0 = b + 17U; i1 = b + 32U;  i2 = b + 65U;  i3 = b + 112U; break;
        }
        ulong a2 = scratch_q[i0], c2 = scratch_q[i1], d2 = scratch_q[i2], e2 = scratch_q[i3];
        permute_step(a2, c2, d2, e2);
        scratch_q[i0] = a2; scratch_q[i1] = c2; scratch_q[i2] = d2; scratch_q[i3] = e2;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        ulong mixed = scratch_q[i] ^ scratch_r[i];
        if (xor_out) {
            out[i] ^= mixed;
        } else {
            out[i] = mixed;
        }
    }
}

inline void update_address_block_coop(
    threadgroup ulong *address_block,
    threadgroup ulong *input_block,
    threadgroup const ulong *zero_block,
    threadgroup ulong *scratch_r,
    threadgroup ulong *scratch_q,
    device ulong *address_device, // temporary device backing for zero_block → address compress
    device ulong *input_device,   // temporary device backing for input_block compress
    uint tid
) {
    // We need device-space pointers for compress_block_coop, so copy to/from threadgroup.
    // Write input_block and zero_block into device scratch, do compress, read back.
    // Actually, let's create a version that works with threadgroup memory directly.
    // Since compress_block_coop expects device pointers, we'll use a different approach.

    if (tid == 0U) {
        input_block[6] += 1ULL;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // First compress: zero ^ input → address
    for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        ulong r = input_block[i]; // zero ^ input = 0 ^ input = input
        scratch_r[i] = r;
        scratch_q[i] = r;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Row round 1
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint base = state * 16U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = base;      i1 = base + 4U;  i2 = base + 8U;  i3 = base + 12U; break;
            case 1U: i0 = base + 1U; i1 = base + 5U;  i2 = base + 9U;  i3 = base + 13U; break;
            case 2U: i0 = base + 2U; i1 = base + 6U;  i2 = base + 10U; i3 = base + 14U; break;
            default: i0 = base + 3U; i1 = base + 7U;  i2 = base + 11U; i3 = base + 15U; break;
        }
        ulong a = scratch_q[i0], b = scratch_q[i1], c = scratch_q[i2], d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a; scratch_q[i1] = b; scratch_q[i2] = c; scratch_q[i3] = d;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Row round 2
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint base = state * 16U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = base;      i1 = base + 5U;  i2 = base + 10U; i3 = base + 15U; break;
            case 1U: i0 = base + 1U; i1 = base + 6U;  i2 = base + 11U; i3 = base + 12U; break;
            case 2U: i0 = base + 2U; i1 = base + 7U;  i2 = base + 8U;  i3 = base + 13U; break;
            default: i0 = base + 3U; i1 = base + 4U;  i2 = base + 9U;  i3 = base + 14U; break;
        }
        ulong a = scratch_q[i0], b = scratch_q[i1], c = scratch_q[i2], d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a; scratch_q[i1] = b; scratch_q[i2] = c; scratch_q[i3] = d;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Column round 1
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint bb = state * 2U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = bb;       i1 = bb + 32U;  i2 = bb + 64U;  i3 = bb + 96U;  break;
            case 1U: i0 = bb + 1U;  i1 = bb + 33U;  i2 = bb + 65U;  i3 = bb + 97U;  break;
            case 2U: i0 = bb + 16U; i1 = bb + 48U;  i2 = bb + 80U;  i3 = bb + 112U; break;
            default: i0 = bb + 17U; i1 = bb + 49U;  i2 = bb + 81U;  i3 = bb + 113U; break;
        }
        ulong a = scratch_q[i0], c = scratch_q[i1], d = scratch_q[i2], e = scratch_q[i3];
        permute_step(a, c, d, e);
        scratch_q[i0] = a; scratch_q[i1] = c; scratch_q[i2] = d; scratch_q[i3] = e;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Column round 2
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint bb = state * 2U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = bb;       i1 = bb + 33U;  i2 = bb + 80U;  i3 = bb + 113U; break;
            case 1U: i0 = bb + 1U;  i1 = bb + 48U;  i2 = bb + 81U;  i3 = bb + 96U;  break;
            case 2U: i0 = bb + 16U; i1 = bb + 49U;  i2 = bb + 64U;  i3 = bb + 97U;  break;
            default: i0 = bb + 17U; i1 = bb + 32U;  i2 = bb + 65U;  i3 = bb + 112U; break;
        }
        ulong a = scratch_q[i0], c = scratch_q[i1], d = scratch_q[i2], e = scratch_q[i3];
        permute_step(a, c, d, e);
        scratch_q[i0] = a; scratch_q[i1] = c; scratch_q[i2] = d; scratch_q[i3] = e;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        address_block[i] = scratch_q[i] ^ scratch_r[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Second compress: zero ^ address → address (in-place)
    for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        ulong r = address_block[i]; // zero ^ address = address
        scratch_r[i] = r;
        scratch_q[i] = r;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Row round 1
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint base = state * 16U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = base;      i1 = base + 4U;  i2 = base + 8U;  i3 = base + 12U; break;
            case 1U: i0 = base + 1U; i1 = base + 5U;  i2 = base + 9U;  i3 = base + 13U; break;
            case 2U: i0 = base + 2U; i1 = base + 6U;  i2 = base + 10U; i3 = base + 14U; break;
            default: i0 = base + 3U; i1 = base + 7U;  i2 = base + 11U; i3 = base + 15U; break;
        }
        ulong a = scratch_q[i0], b = scratch_q[i1], c = scratch_q[i2], d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a; scratch_q[i1] = b; scratch_q[i2] = c; scratch_q[i3] = d;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Row round 2
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint base = state * 16U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = base;      i1 = base + 5U;  i2 = base + 10U; i3 = base + 15U; break;
            case 1U: i0 = base + 1U; i1 = base + 6U;  i2 = base + 11U; i3 = base + 12U; break;
            case 2U: i0 = base + 2U; i1 = base + 7U;  i2 = base + 8U;  i3 = base + 13U; break;
            default: i0 = base + 3U; i1 = base + 4U;  i2 = base + 9U;  i3 = base + 14U; break;
        }
        ulong a = scratch_q[i0], b = scratch_q[i1], c = scratch_q[i2], d = scratch_q[i3];
        permute_step(a, b, c, d);
        scratch_q[i0] = a; scratch_q[i1] = b; scratch_q[i2] = c; scratch_q[i3] = d;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Column round 1
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint bb = state * 2U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = bb;       i1 = bb + 32U;  i2 = bb + 64U;  i3 = bb + 96U;  break;
            case 1U: i0 = bb + 1U;  i1 = bb + 33U;  i2 = bb + 65U;  i3 = bb + 97U;  break;
            case 2U: i0 = bb + 16U; i1 = bb + 48U;  i2 = bb + 80U;  i3 = bb + 112U; break;
            default: i0 = bb + 17U; i1 = bb + 49U;  i2 = bb + 81U;  i3 = bb + 113U; break;
        }
        ulong a = scratch_q[i0], c = scratch_q[i1], d = scratch_q[i2], e = scratch_q[i3];
        permute_step(a, c, d, e);
        scratch_q[i0] = a; scratch_q[i1] = c; scratch_q[i2] = d; scratch_q[i3] = e;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Column round 2
    {
        uint state = tid >> 2; uint lane = tid & 3U; uint bb = state * 2U;
        uint i0, i1, i2, i3;
        switch (lane) {
            case 0U: i0 = bb;       i1 = bb + 33U;  i2 = bb + 80U;  i3 = bb + 113U; break;
            case 1U: i0 = bb + 1U;  i1 = bb + 48U;  i2 = bb + 81U;  i3 = bb + 96U;  break;
            case 2U: i0 = bb + 16U; i1 = bb + 49U;  i2 = bb + 64U;  i3 = bb + 97U;  break;
            default: i0 = bb + 17U; i1 = bb + 32U;  i2 = bb + 65U;  i3 = bb + 112U; break;
        }
        ulong a = scratch_q[i0], c = scratch_q[i1], d = scratch_q[i2], e = scratch_q[i3];
        permute_step(a, c, d, e);
        scratch_q[i0] = a; scratch_q[i1] = c; scratch_q[i2] = d; scratch_q[i3] = e;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
        address_block[i] = scratch_q[i] ^ scratch_r[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------- Kernel 2: argon2id_fill ----------

kernel void argon2id_fill_kernel(
    device const ulong  *seed_blocks              [[buffer(0)]],
    device ulong        *lane_memory               [[buffer(1)]],
    device ulong        *out_last_blocks            [[buffer(2)]],
    device volatile uint *cancel_flag               [[buffer(3)]],
    device atomic_uint  *completed_iters            [[buffer(4)]],
    device const uchar  *target                     [[buffer(5)]],
    device atomic_uint  *found_index_one_based      [[buffer(6)]],
    constant uint       &lanes_active               [[buffer(7)]],
    constant uint       &active_hashes              [[buffer(8)]],
    constant uint       &lane_launch_iters          [[buffer(9)]],
    constant uint       &evaluate_target            [[buffer(10)]],
    uint lane [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {
    uint effective_m_blocks = SEINE_M_BLOCKS;
    uint effective_t_cost = SEINE_T_COST;
    if (lane >= lanes_active || effective_m_blocks < 8U) {
        return;
    }

    ulong lane_stride_words = ulong(effective_m_blocks) * 128ULL;
    device ulong *memory = lane_memory + ulong(lane) * lane_stride_words;
    uint segment_length = effective_m_blocks / 4U;
    uint lane_length = effective_m_blocks;

    threadgroup ulong address_block[128];
    threadgroup ulong input_block[128];
    threadgroup ulong zero_block[128];
    threadgroup ulong scratch_r[128];
    threadgroup ulong scratch_q[128];
    threadgroup uint tg_cancel_requested;
    bool abort_requested = false;

    for (uint iter = 0U; iter < lane_launch_iters; ++iter) {
        if (tid == 0U) {
            tg_cancel_requested = cancel_flag[0];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (tg_cancel_requested != 0U) {
            break;
        }

        ulong global_hash_idx = ulong(iter) * lanes_active + ulong(lane);
        if (global_hash_idx >= ulong(active_hashes)) {
            break;
        }

        // Load seed blocks into lane memory
        device const ulong *seed = seed_blocks + global_hash_idx * 256ULL;
        for (uint i = tid; i < 256U; i += ARGON2_COOP_THREADS) {
            memory[i] = seed[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Initialize shared blocks to zero
        for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
            address_block[i] = 0ULL;
            input_block[i] = 0ULL;
            zero_block[i] = 0ULL;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint pass = 0; pass < effective_t_cost && !abort_requested; ++pass) {
            for (uint slice = 0; slice < 4U && !abort_requested; ++slice) {
                bool data_independent = (pass == 0U && slice < 2U);
                if (data_independent) {
                    for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
                        address_block[i] = 0ULL;
                        input_block[i] = 0ULL;
                    }
                    if (tid == 0U) {
                        input_block[0] = ulong(pass);
                        input_block[1] = 0ULL; // lane index (p=1)
                        input_block[2] = ulong(slice);
                        input_block[3] = ulong(effective_m_blocks);
                        input_block[4] = ulong(effective_t_cost);
                        input_block[5] = 2ULL; // Argon2id
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }

                uint first_block = (pass == 0U && slice == 0U) ? 2U : 0U;
                uint cur_index = slice * segment_length + first_block;
                uint prev_index = (slice == 0U && first_block == 0U)
                    ? (cur_index + lane_length - 1U)
                    : (cur_index - 1U);

                for (uint block = first_block; block < segment_length; ++block) {
                    // Cancel check
                    if (((block - first_block) % CANCEL_CHECK_BLOCK_INTERVAL) == 0U) {
                        if (tid == 0U) {
                            tg_cancel_requested = cancel_flag[0];
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        if (tg_cancel_requested != 0U) {
                            abort_requested = true;
                            break;
                        }
                    }

                    ulong rand64;
                    if (data_independent) {
                        uint address_index = block & 127U;
                        if (address_index == 0U) {
                            // Inline address block update (all threadgroup)
                            if (tid == 0U) {
                                input_block[6] += 1ULL;
                            }
                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            // compress: zero ^ input → address
                            for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
                                ulong r = input_block[i];
                                scratch_r[i] = r;
                                scratch_q[i] = r;
                            }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Row 1
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bs = st * 16U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bs; j1=bs+4U; j2=bs+8U; j3=bs+12U; break;
                                            case 1U: j0=bs+1U; j1=bs+5U; j2=bs+9U; j3=bs+13U; break;
                                            case 2U: j0=bs+2U; j1=bs+6U; j2=bs+10U; j3=bs+14U; break;
                                            default: j0=bs+3U; j1=bs+7U; j2=bs+11U; j3=bs+15U; break; }
                              ulong a=scratch_q[j0],b=scratch_q[j1],c=scratch_q[j2],d=scratch_q[j3];
                              permute_step(a,b,c,d);
                              scratch_q[j0]=a; scratch_q[j1]=b; scratch_q[j2]=c; scratch_q[j3]=d; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Row 2
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bs = st * 16U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bs; j1=bs+5U; j2=bs+10U; j3=bs+15U; break;
                                            case 1U: j0=bs+1U; j1=bs+6U; j2=bs+11U; j3=bs+12U; break;
                                            case 2U: j0=bs+2U; j1=bs+7U; j2=bs+8U; j3=bs+13U; break;
                                            default: j0=bs+3U; j1=bs+4U; j2=bs+9U; j3=bs+14U; break; }
                              ulong a=scratch_q[j0],b=scratch_q[j1],c=scratch_q[j2],d=scratch_q[j3];
                              permute_step(a,b,c,d);
                              scratch_q[j0]=a; scratch_q[j1]=b; scratch_q[j2]=c; scratch_q[j3]=d; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Col 1
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bb = st * 2U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bb; j1=bb+32U; j2=bb+64U; j3=bb+96U; break;
                                            case 1U: j0=bb+1U; j1=bb+33U; j2=bb+65U; j3=bb+97U; break;
                                            case 2U: j0=bb+16U; j1=bb+48U; j2=bb+80U; j3=bb+112U; break;
                                            default: j0=bb+17U; j1=bb+49U; j2=bb+81U; j3=bb+113U; break; }
                              ulong a=scratch_q[j0],c=scratch_q[j1],d=scratch_q[j2],e=scratch_q[j3];
                              permute_step(a,c,d,e);
                              scratch_q[j0]=a; scratch_q[j1]=c; scratch_q[j2]=d; scratch_q[j3]=e; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Col 2
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bb = st * 2U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bb; j1=bb+33U; j2=bb+80U; j3=bb+113U; break;
                                            case 1U: j0=bb+1U; j1=bb+48U; j2=bb+81U; j3=bb+96U; break;
                                            case 2U: j0=bb+16U; j1=bb+49U; j2=bb+64U; j3=bb+97U; break;
                                            default: j0=bb+17U; j1=bb+32U; j2=bb+65U; j3=bb+112U; break; }
                              ulong a=scratch_q[j0],c=scratch_q[j1],d=scratch_q[j2],e=scratch_q[j3];
                              permute_step(a,c,d,e);
                              scratch_q[j0]=a; scratch_q[j1]=c; scratch_q[j2]=d; scratch_q[j3]=e; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
                                address_block[i] = scratch_q[i] ^ scratch_r[i];
                            }
                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            // Second compress: zero ^ address → address
                            for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
                                ulong r = address_block[i];
                                scratch_r[i] = r;
                                scratch_q[i] = r;
                            }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Row 1
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bs = st * 16U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bs; j1=bs+4U; j2=bs+8U; j3=bs+12U; break;
                                            case 1U: j0=bs+1U; j1=bs+5U; j2=bs+9U; j3=bs+13U; break;
                                            case 2U: j0=bs+2U; j1=bs+6U; j2=bs+10U; j3=bs+14U; break;
                                            default: j0=bs+3U; j1=bs+7U; j2=bs+11U; j3=bs+15U; break; }
                              ulong a=scratch_q[j0],b=scratch_q[j1],c=scratch_q[j2],d=scratch_q[j3];
                              permute_step(a,b,c,d);
                              scratch_q[j0]=a; scratch_q[j1]=b; scratch_q[j2]=c; scratch_q[j3]=d; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Row 2
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bs = st * 16U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bs; j1=bs+5U; j2=bs+10U; j3=bs+15U; break;
                                            case 1U: j0=bs+1U; j1=bs+6U; j2=bs+11U; j3=bs+12U; break;
                                            case 2U: j0=bs+2U; j1=bs+7U; j2=bs+8U; j3=bs+13U; break;
                                            default: j0=bs+3U; j1=bs+4U; j2=bs+9U; j3=bs+14U; break; }
                              ulong a=scratch_q[j0],b=scratch_q[j1],c=scratch_q[j2],d=scratch_q[j3];
                              permute_step(a,b,c,d);
                              scratch_q[j0]=a; scratch_q[j1]=b; scratch_q[j2]=c; scratch_q[j3]=d; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Col 1
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bb = st * 2U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bb; j1=bb+32U; j2=bb+64U; j3=bb+96U; break;
                                            case 1U: j0=bb+1U; j1=bb+33U; j2=bb+65U; j3=bb+97U; break;
                                            case 2U: j0=bb+16U; j1=bb+48U; j2=bb+80U; j3=bb+112U; break;
                                            default: j0=bb+17U; j1=bb+49U; j2=bb+81U; j3=bb+113U; break; }
                              ulong a=scratch_q[j0],c=scratch_q[j1],d=scratch_q[j2],e=scratch_q[j3];
                              permute_step(a,c,d,e);
                              scratch_q[j0]=a; scratch_q[j1]=c; scratch_q[j2]=d; scratch_q[j3]=e; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            // Col 2
                            { uint st = tid >> 2; uint ln = tid & 3U; uint bb = st * 2U;
                              uint j0, j1, j2, j3;
                              switch (ln) { case 0U: j0=bb; j1=bb+33U; j2=bb+80U; j3=bb+113U; break;
                                            case 1U: j0=bb+1U; j1=bb+48U; j2=bb+81U; j3=bb+96U; break;
                                            case 2U: j0=bb+16U; j1=bb+49U; j2=bb+64U; j3=bb+97U; break;
                                            default: j0=bb+17U; j1=bb+32U; j2=bb+65U; j3=bb+112U; break; }
                              ulong a=scratch_q[j0],c=scratch_q[j1],d=scratch_q[j2],e=scratch_q[j3];
                              permute_step(a,c,d,e);
                              scratch_q[j0]=a; scratch_q[j1]=c; scratch_q[j2]=d; scratch_q[j3]=e; }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                            for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
                                address_block[i] = scratch_q[i] ^ scratch_r[i];
                            }
                            simdgroup_barrier(mem_flags::mem_threadgroup);
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        rand64 = address_block[address_index];
                    } else {
                        rand64 = memory[ulong(prev_index) * 128ULL];
                    }

                    // Reference index calculation
                    ulong reference_area_size;
                    if (pass == 0U) {
                        if (slice == 0U) {
                            reference_area_size = ulong(block - 1U);
                        } else {
                            reference_area_size =
                                ulong(slice) * segment_length +
                                ulong(block) - 1ULL;
                        }
                    } else {
                        reference_area_size =
                            ulong(lane_length - segment_length) +
                            ulong(block) - 1ULL;
                    }

                    ulong map = rand64 & 0xFFFFFFFFULL;
                    map = (map * map) >> 32;
                    ulong relative_position =
                        reference_area_size - 1ULL -
                        ((reference_area_size * map) >> 32);

                    uint start_position =
                        (pass != 0U && slice != 3U) ? ((slice + 1U) * segment_length) : 0U;
                    uint ref_index = uint(
                        (ulong(start_position) + relative_position) %
                        ulong(lane_length)
                    );

                    ulong dst_offset = ulong(cur_index) * 128ULL;
                    ulong rhs_offset = ulong(prev_index) * 128ULL;
                    ulong lhs_offset = ulong(ref_index) * 128ULL;
                    if (pass == 0U) {
                        compress_block_coop(
                            memory + rhs_offset,
                            memory + lhs_offset,
                            memory + dst_offset,
                            false,
                            scratch_r, scratch_q, tid
                        );
                    } else {
                        compress_block_coop(
                            memory + rhs_offset,
                            memory + lhs_offset,
                            memory + dst_offset,
                            true,
                            scratch_r, scratch_q, tid
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    prev_index = cur_index;
                    cur_index += 1U;
                }
            }
        }
        if (abort_requested) {
            break;
        }

        // Copy last block to output
        device const ulong *last = memory + (ulong(lane_length) - 1ULL) * 128ULL;
        device ulong *out = out_last_blocks + global_hash_idx * 128ULL;
        for (uint i = tid; i < 128U; i += ARGON2_COOP_THREADS) {
            out[i] = last[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Optional fused target check
        if (evaluate_target != 0U && tid == 0U) {
            uchar hash[POW_OUTPUT_BYTES];
            blake2b_long_32_from_words(last, hash);
            if (hash_meets_target_be(hash, target)) {
                atomic_fetch_min_explicit(
                    found_index_one_based,
                    uint(global_hash_idx) + 1U,
                    memory_order_relaxed
                );
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0U) {
            atomic_fetch_max_explicit(completed_iters, iter + 1U, memory_order_relaxed);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------- Kernel 3: evaluate_hashes ----------

kernel void evaluate_hashes_kernel(
    device const ulong  *last_blocks              [[buffer(0)]],
    device const uchar  *target                    [[buffer(1)]],
    device atomic_uint  *found_index_one_based     [[buffer(2)]],
    constant uint       &active_hashes             [[buffer(3)]],
    uint hash_idx [[thread_position_in_grid]]
) {
    if (hash_idx >= active_hashes) {
        return;
    }

    device const ulong *last = last_blocks + ulong(hash_idx) * 128ULL;
    uchar hash[POW_OUTPUT_BYTES];
    blake2b_long_32_from_words(last, hash);
    if (hash_meets_target_be(hash, target)) {
        atomic_fetch_min_explicit(found_index_one_based, hash_idx + 1U, memory_order_relaxed);
    }
}
