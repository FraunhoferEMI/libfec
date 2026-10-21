/* Reed-Solomon encoder
 * Copyright 2004, Phil Karn, KA9Q
 * May be used under the terms of the GNU Lesser General Public License (LGPL)
 */
#include <string.h>
#include <stdint.h>
#include "fixed.h"

#if defined(__VEC__) && defined(__vec__)
    #define POWERPC
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
    #define ARM_NEON
#endif
#if defined(__SSSE3__) && (defined(__x86_64__) || defined(__i386__))
    #define SSSE3
#endif

#ifdef POWERPC
#include <sys/sysctl.h>
static void encode_rs_8_av(data_t *data, data_t *parity,int pad);
#endif

#ifdef ARM_NEON
#include <arm_neon.h>
static void encode_rs_8_neon(data_t *data, data_t *parity,int pad);
#endif

#ifdef SSSE3
#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
static void encode_rs_8_ssse3(data_t *data, data_t *parity,int pad);
#endif

static void encode_rs_8_c(data_t *data, data_t *parity,int pad);

void encode_rs_8(data_t *data, data_t *parity,int pad){
    #ifdef ARM_NEON
    encode_rs_8_neon(data,parity,pad);
    return;
    #endif
    #ifdef SSSE3
    __builtin_cpu_init();
    if (__builtin_cpu_supports("ssse3")) {
        encode_rs_8_ssse3(data, parity, pad);
        return;
    }
    #endif
    #ifdef POWERPC
    int selectors[2] = { CTL_HW, HW_VECTORUNIT };
    int hasVectorUnit = 0;
    size_t length = sizeof(hasVectorUnit);
    int error = sysctl(selectors, 2, &hasVectorUnit, &length, NULL, 0);
    if(0 == error && hasVectorUnit){
        encode_rs_8_av(data,parity,pad);
        return;
    }
    #endif

    encode_rs_8_c(data,parity,pad);
    return;

}

#if defined(ARM_NEON) || defined(SSSE3)

static data_t gf_mul(data_t a, data_t b)
{
    if (a == 0 || b == 0) return 0;
    return CCSDS_alpha_to[mod255((int)CCSDS_index_of[a] + (int)CCSDS_index_of[b])];
}

static data_t G_poly[NROOTS];
static data_t T_lo[16][NROOTS] __attribute__((aligned(32)));
static data_t T_hi[16][NROOTS] __attribute__((aligned(32)));
static int    tables_ready = 0;

void rs_simd_init(void)
{
    int j, n;
    if (tables_ready) return;

    for (j = 0; j < NROOTS; j++) {
        int gi = NROOTS - 1 - j;
        G_poly[j] = CCSDS_alpha_to[CCSDS_poly[gi]];
    }

    for (n = 0; n < 16; n++)
        for (j = 0; j < NROOTS; j++) {
            T_lo[n][j] = gf_mul((data_t)n,                  G_poly[j]);
            T_hi[n][j] = gf_mul((data_t)((unsigned)n << 4), G_poly[j]);
        }

    tables_ready = 1;
}
#endif
#ifdef SSSE3
static void encode_rs_8_ssse3(data_t *data, data_t *parity, int pad)
{
    const int nsym = NN - NROOTS - pad;
    int i;

    rs_simd_init();

    __m128i p0 = _mm_setzero_si128();   /* parity[0..15]  */
    __m128i p1 = _mm_setzero_si128();   /* parity[16..31] */

    for (i = 0; i < nsym; i++) {
        /* parity[0] = Byte 0 von p0 */
        unsigned fb = (unsigned)data[i] ^ (unsigned)(_mm_cvtsi128_si32(p0) & 0xFF);

        /* Shift: parity[j] <- parity[j+1], parity[31] <- 0 */
        __m128i carry = _mm_slli_si128(p1, 15);          /* p1[0] nach Byte 15 */
        p0 = _mm_or_si128(_mm_srli_si128(p0, 1), carry);
        p1 = _mm_srli_si128(p1, 1);

        /* XOR mit fb * G, zerlegt in Nibbles */
        const __m128i *lo = (const __m128i *)&T_lo[fb & 15][0];
        const __m128i *hi = (const __m128i *)&T_hi[fb >> 4][0];
        p0 = _mm_xor_si128(p0, _mm_xor_si128(_mm_load_si128(lo + 0), _mm_load_si128(hi + 0)));
        p1 = _mm_xor_si128(p1, _mm_xor_si128(_mm_load_si128(lo + 1), _mm_load_si128(hi + 1)));
    }

    _mm_storeu_si128((__m128i *)parity + 0, p0);
    _mm_storeu_si128((__m128i *)parity + 1, p1);
}
#endif

#ifdef ARM_NEON
_Static_assert(NROOTS == 32, "NEON-Layout expects exactly 32 Paritybytes");

void encode_rs_8_neon(data_t *data, data_t *parity, int pad)
{
    const int nsym = NN - NROOTS - pad;
    const uint8x16_t zero = vdupq_n_u8(0);
    uint8x16_t p0 = zero;   /* parity[0..15]  */
    uint8x16_t p1 = zero;   /* parity[16..31] */
    int i;

    rs_simd_init();

    for (i = 0; i < nsym; i++) {
        const unsigned fb = (unsigned)data[i] ^ (unsigned)vgetq_lane_u8(p0, 0);
        const uint8_t *lo = (const uint8_t *)&T_lo[fb & 15][0];
        const uint8_t *hi = (const uint8_t *)&T_hi[fb >> 4][0];

        /* Shift: parity[j] <- parity[j+1], parity[31] <- 0 */
        p0 = vextq_u8(p0, p1,   1);     /* MUSS vor der p1-Zeile stehen */
        p1 = vextq_u8(p1, zero, 1);

        p0 = veorq_u8(p0, veorq_u8(vld1q_u8(lo),      vld1q_u8(hi)));
        p1 = veorq_u8(p1, veorq_u8(vld1q_u8(lo + 16), vld1q_u8(hi + 16)));
    }

    vst1q_u8(parity,      p0);
    vst1q_u8(parity + 16, p1);
}
#endif

#if POWERPC /* PowerPC G4/G5 Altivec instructions are available */

static vector unsigned char reverse = (vector unsigned char)(0,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
static vector unsigned char shift_right = (vector unsigned char)(15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);

/* Lookup table for feedback multiplications
 * These are the low half of the coefficients. Since the generator polynomial is
 * palindromic, we form the other half by reversing this one
 */
extern static union { vector unsigned char v; unsigned char c[16]; } table[256];

static void encode_rs_8_av(data_t *data, data_t *parity,int pad){
  union { vector unsigned char v[2]; unsigned char c[32]; } shift_register;
  int i;

  shift_register.v[0] = (vector unsigned char)(0);
  shift_register.v[1] = (vector unsigned char)(0);

  for(i=0;i<NN-NROOTS-pad;i++){
    vector unsigned char feedback0,feedback1;
    unsigned char f;

    f = data[i] ^ shift_register.c[31];
    feedback1 = table[f].v;
    feedback0 = vec_perm(feedback1,feedback1,reverse);

    /* Shift right one byte */
    shift_register.v[1] = vec_perm(shift_register.v[0],shift_register.v[1],shift_right) ^ feedback1;
    shift_register.v[0] = vec_sro(shift_register.v[0],(vector unsigned char)(8)) ^ feedback0;
    shift_register.c[0] = f;
  }
  for(i=0;i<NROOTS;i++)
    parity[NROOTS-i-1] = shift_register.c[i];
}
#endif

/* Portable C version */
static void encode_rs_8_c(data_t *data, data_t *parity,int pad){

#include "encode_rs.h"

}
