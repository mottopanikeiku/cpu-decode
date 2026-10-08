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
}
