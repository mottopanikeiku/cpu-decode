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
} // namespace decode::detail
#endif
