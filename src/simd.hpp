#pragma once
// ISA selection shared by the kernel translation units.
//
// DECODE_X86:  AVX-512 kernels are compiled (selected at run time).
// DECODE_NEON: AArch64 NEON kernels are compiled (dot product via run-time check).
// CPU_DECODE_EMULATE_AVX512 compiles the AVX-512 kernels through SIMDe so their
// arithmetic can be tested on hosts without AVX-512 VNNI (CMake option of the same name).
#if defined(CPU_DECODE_EMULATE_AVX512)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>
#include <simde/x86/f16c.h>
#define DECODE_X86 1
#define DECODE_AVX512
#ifndef _MM_FROUND_NO_EXC
#define _MM_FROUND_NO_EXC SIMDE_MM_FROUND_NO_EXC
#endif
#elif defined(__x86_64__)
#include <immintrin.h>
#define DECODE_X86 1
#define DECODE_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2,f16c,fma")))
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define DECODE_NEON 1
#define DECODE_NEON_DOT __attribute__((target("+dotprod")))
#endif

namespace decode::detail {
bool avx512_supported();
bool neon_supported();
#ifdef DECODE_X86
// Horizontal sum (SIMDe lacks _mm512_reduce_add_ps; GCC expands it the same way).
DECODE_AVX512 inline float sum512(__m512 v) {
    __m256 h = _mm256_add_ps(_mm512_castps512_ps256(v), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)));
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
    q = _mm_add_ps(q, _mm_movehl_ps(q, q));
    q = _mm_add_ss(q, _mm_movehdup_ps(q));
    return _mm_cvtss_f32(q);
}
#endif
#ifdef DECODE_NEON
// exp(x) for x in [-87, 88] (clamped outside): Cody-Waite reduction to r in
// [-ln2/2, ln2/2], degree-6 Taylor polynomial (relative error below 2e-7), times 2^n.
inline float32x4_t exp_neon(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.0f / 720);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t scale = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(scale));
}
#endif
}
