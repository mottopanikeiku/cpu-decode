#pragma once
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

namespace decode::detail {
// Softmax supplies x <= 0. Range reduction leaves |r| <= log(2)/2;
// a degree-seven Taylor polynomial is evaluated with explicit fused operations.
// Keep subnormal outputs rather than clipping at the normal-float boundary.
__attribute__((target("avx512f"), always_inline))
inline __m512 exp_nonpositive(__m512 x) {
    __mmask16 underflow = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-104.0f), _CMP_LT_OQ);
    __m512 bounded = _mm512_mask_mov_ps(x, underflow, _mm512_set1_ps(-104.0f));
    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(bounded, _mm512_set1_ps(1.4426950408889634f)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693145751953125f), bounded);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(1.428606765330187e-6f), r);
    __m512 p = _mm512_set1_ps(1.0f / 5040.0f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 720.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_mask_mov_ps(_mm512_scalef_ps(p, n), underflow, _mm512_setzero_ps());
}
// Float-typed quarters preserve the same pairwise addition tree without
// reinterpretation as double vectors at the reduction boundary.
__attribute__((target("avx512f"), always_inline))
inline float reduce_add(__m512 x) {
    __m128 lower = _mm_add_ps(_mm512_castps512_ps128(x), _mm512_extractf32x4_ps(x, 2));
    __m128 upper = _mm_add_ps(_mm512_extractf32x4_ps(x, 1), _mm512_extractf32x4_ps(x, 3));
    __m128 sum = _mm_add_ps(lower, upper);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55)));
}
} // namespace decode::detail
#endif
