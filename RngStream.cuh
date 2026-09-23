/**
 * @file RngStream.cuh
 *
 * @brief CUDA-compatible implementation of Pierre L'Ecuyer's
 * MRG32k3a random number generator.
 *
 * This implementation provides:
 * - generation of uniform random variates;
 * - stream advancement by 2^127 steps;
 * - substream advancement by 2^76 steps;
 * - efficient advancement by arbitrary numbers of streams
 *   and substreams using precomputed matrix powers.
 *
 * The core modular arithmetic routines are available on both
 * host and device through the HD macro.
 *
 * The implementation is compatible with the stream and
 * substream structure used by R's "L'Ecuyer-CMRG" generator
 * and the `parallel` package.
 */

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#define __device__
#endif

namespace rngstream {

  using Int = int64_t;
  using UInt = uint32_t;
  using Seed = UInt[6];
  using Matrix = UInt[3][3];
  
  constexpr Int m1 = 4294967087LL,
    m2 = 4294944443LL;
  
  constexpr UInt a12 = 1403580U,
    a13n = 810728U,
    a21 = 527612U,
    a23n = 1370589U;
  
  constexpr double normc = 2.328306549295727688e-10;

  constexpr float normcf = 2.3283065e-10f;

  inline constexpr Matrix A1p76 = {
    {      82758667U, 1871391091U, 4127413238U },
      {    3672831523U,   69195019U, 1871391091U },
      {    3672091415U, 3528743235U,   69195019U }
  };

  inline constexpr Matrix A2p76 = {
    {    1511326704U, 3759209742U, 1610795712U },
    {    4292754251U, 1511326704U, 3889917532U },
    {    3859662829U, 4292754251U, 3708466080U }
  };

  inline constexpr Matrix A1p127 = {
    {    2427906178U, 3580155704U,  949770784U },
    {     226153695U, 1230515664U, 3580155704U },
    {    1988835001U,  986791581U, 1230515664U }
  };

  inline constexpr Matrix A2p127 = {
    {    1464411153U,  277697599U, 1610723613U },
    {      32183930U, 1464411153U, 1022607788U },
    {    2824425944U,   32183930U, 2093834863U }
  };
  
  __device__ constexpr Matrix A1p76_dev = {
    {      82758667U, 1871391091U, 4127413238U },
    {    3672831523U,   69195019U, 1871391091U },
    {    3672091415U, 3528743235U,   69195019U }
  };

  __device__ constexpr Matrix A2p76_dev = {
    {    1511326704U, 3759209742U, 1610795712U },
    {    4292754251U, 1511326704U, 3889917532U },
    {    3859662829U, 4292754251U, 3708466080U }
  };

  __device__ constexpr Matrix A1p127_dev = {
    {    2427906178U, 3580155704U,  949770784U },
    {     226153695U, 1230515664U, 3580155704U },
    {    1988835001U,  986791581U, 1230515664U }
  };

  __device__ constexpr Matrix A2p127_dev = {
    {    1464411153U,  277697599U, 1610723613U },
    {      32183930U, 1464411153U, 1022607788U },
    {    2824425944U,   32183930U, 2093834863U }
  };
  
  /**
   * @brief Computes (a*s + c) modulo m.
   *
   * Uses 128-bit intermediate arithmetic to avoid overflow.
   * Parameters a, s, c and m must be < 2^35.
   *
   * @param a Multiplicative coefficient.
   * @param s State value.
   * @param c Additive accumulator.
   * @param m Modulus.
   *
   * @return (a*s + c) mod m.
   */
  HD 
  inline UInt MultModM (UInt a, UInt s, UInt c, Int m)
  {

#if defined(__CUDA_ARCH__)
    __int128 x = (__int128)a * (__int128)s + c;
#else
    __int128_t x = (__int128_t)a * (__int128_t)s + c;
#endif

    x %= m;
    if (x < 0)
      x += m;
    return static_cast<UInt>(x);
  }

  /**
   * @brief Computes the matrix-vector product A*s modulo m.
   *
   * The vector @p s is modified in place.
   * Assume that -m < s[i] < m.
   * Works also when v = s.
   *
   * @param A Transition matrix.
   * @param s Input/output state vector.
   * @param m Modulus.
   */
  HD
  inline void MatVecModM (const Matrix A, UInt s[3], Int m)
  {
    int i;
    UInt x[3];

    for (i = 0; i < 3; ++i) {
      UInt t = 0;
      t = MultModM (A[i][0], s[0], t, m);
      t = MultModM (A[i][1], s[1], t, m);
      t = MultModM (A[i][2], s[2], t, m);

      x[i] = t;
    }
    s[0] = x[0];
    s[1] = x[1];
    s[2] = x[2];
  }

  /**
   * @brief Computes C = A*B modulo m.
   *
   * The routine supports aliasing, i.e. C may be identical
   * to A or B.
   *
   * @param A Left matrix.
   * @param B Right matrix.
   * @param C Output matrix.
   * @param m Modulus.
   */
  HD
  inline void MatMatModM (const Matrix A, const Matrix B,
			  Matrix C, Int m)
  {
    int i, j;
    UInt V[3];
    Matrix W;

    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j)
	V[j] = B[j][i];
      MatVecModM (A, V, m);
      for (j = 0; j < 3; ++j)
	W[j][i] = V[j];
    }
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
	C[i][j] = W[i][j];
  }
  
  /**
   * @brief Computes A^n modulo m.
   *
   * Uses binary exponentiation.
   *
   * @param A Input matrix.
   * @param B Output matrix.
   * @param m Modulus.
   * @param n Non-negative exponent.
   */
  HD
  inline void MatPowModM (const Matrix A, Matrix B, Int m, Int n)
  {
    int i, j;
    Matrix W;

    /* initialize: W = A; B = I */
    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j) {
	W[i][j] = A[i][j];
	B[i][j] = 0;
      }
    for (j = 0; j < 3; ++j)
      B[j][j] = 1;

    /* Compute B = A^n mod m using the binary decomposition of n */
    while (n > 0) {
      if (n & 1) MatMatModM (W, B, B, m);
      MatMatModM (W, W, W, m);
      n >>= 1;
    }
  }

  /**
   * @brief Computes A^(2^e) modulo m.
   *
   * Repeatedly squares the matrix to obtain a power-of-two
   * advancement operator.
   *
   * @param A Input matrix.
   * @param B Output matrix.
   * @param m Modulus.
   * @param e Exponent of two.
   */
  HD
  inline void MatTwoPowModM (const Matrix A, Matrix B, Int m, int e)
  {
    int i, j;

    /* initialize: B = A */
    if (A != B) {
      for (i = 0; i < 3; ++i)
	for (j = 0; j < 3; ++j)
	  B[i][j] = A[i][j];
    }
    /* Compute B = A^(2^e) mod m */
    for (i = 0; i < e; i++)
      MatMatModM (B, B, B, m);
  }
  
  /**
   * @brief State of an MRG32k3a random-number stream.
   *
   * Contains:
   * - the start of the current stream;
   * - the start of the current substream;
   * - the current state of the generator;
   * - an optional antithetic variate flag.
   *
   * Stream and substream resets reproduce the behaviour of
   * R's L'Ecuyer-CMRG implementation.
   */
  struct RngStream {
    /**
     * @brief Start state of the current stream.
     */
    Seed start_stream;

    /**
     * @brief Start state of the current substream.
     */
    Seed start_substream;

    /**
     * @brief Current generator state.
     */
    Seed seed;

    /**
     * @brief Antithetic variate flag.
     *
     * If enabled, U01() returns 1-u rather than u.
     */
    bool anti;

    /**
     * @brief Constructor with an input seed and anti=false.
     *
     * @param inseed a six-dimensional array for the seed
     */
    HD
    RngStream(const Seed inseed) {
#pragma unroll
      for (int i = 0; i < 6; ++i)
	start_stream[i] = start_substream[i] = seed[i] = inseed[i];
      anti = false;
    }

    /**
     * @brief Constructor with anti=false and an integer seed
     *   as per R's set.seed() with RNGkind("L'Ecuyer-CMRG")
     *
     * @param inseed an unsigned integer
     */
    RngStream(UInt inseed) 
    {
      for (int j = 0; j < 50; ++j)
	inseed = (69069U * inseed + 1U);
      for (int i = 0; i < 6; ++i) {
	inseed = 69069U * inseed + 1U;
	while (inseed >= m2) inseed = 69069U * inseed + 1U;
	start_stream[i] = start_substream[i] = seed[i] = inseed;
      }
    }

    /**
     * @brief Default constructor with anti=false and a seed of
     *   {12345U,12345U,12345U,12345U,12345U,12345U}.
     *
     * @param inseed a six-dimensional array for the seed
     */
    HD
    RngStream() {
#pragma unroll
      for(int i=0;i<6;i++)
        start_stream[i] = start_substream[i] = seed[i] = 12345U;
      anti = false;
    }

  };

  /**
   * @brief Precomputed stream or substream advancement matrices.
   *
   * Entry k contains the matrix:
   *
   *     A^(2^k)
   *
   * where A is either the stream advancement matrix
   * (2^127 steps) or the substream advancement matrix
   * (2^76 steps).
   *
   * These tables allow advancement by an arbitrary number of
   * streams or substreams in O(log n) matrix-vector products.
   *
   * @tparam N Number of precomputed powers.
   */
  template<int N>
  struct AdvanceArrays {
    UInt B1[N][3][3], B2[N][3][3];
    AdvanceArrays(const Matrix A1, const Matrix A2) {
      for (int i = 0; i < 3; ++i)
	for (int j = 0; j < 3; ++j) {
	  B1[0][i][j] = A1[i][j];
	  B2[0][i][j] = A2[i][j];
	}
      for(int k=1; k<N; ++k) {
	MatMatModM(B1[k-1], B1[k-1],
		   B1[k], m1);
	MatMatModM(B2[k-1], B2[k-1],
		   B2[k], m2);
      }
    }
  };

  /**
   * @brief Precomputed substream advancement operators.
   *
   * Represents repeated powers of the matrices A1p76 and
   * A2p76.
   */
  inline const AdvanceArrays<32> advance76(A1p76, A2p76);
  
  /**
   * @brief Precomputed stream advancement operators.
   *
   * Represents repeated powers of the matrices A1p127 and
   * A2p127.
   */
  inline const AdvanceArrays<32> advance127(A1p127, A2p127);
  
  /**
   * @brief Generates the next uniform random variate.
   *
   * Advances the generator by one step and returns a
   * double-precision variate in the interval (0,1).
   *
   * @param st Generator state.
   *
   * @return Uniform random variate.
   */
  HD
  inline double U01(RngStream& st) {
    Int p1, p2; // signed!
  
    p1 = a12 * static_cast<Int>(st.seed[1]) - a13n * static_cast<Int>(st.seed[0]);
    p1 %= m1;
    if (p1 < 0) p1 += m1;
    st.seed[0] = st.seed[1]; st.seed[1] = st.seed[2]; st.seed[2] = static_cast<UInt>(p1);
  
    p2 = a21 * static_cast<Int>(st.seed[5]) - a23n * static_cast<Int>(st.seed[3]);
    p2 %= m2;
    if (p2 < 0) p2 += m2;
    st.seed[3] = st.seed[4]; st.seed[4] = st.seed[5]; st.seed[5] = static_cast<UInt>(p2);
  
    double u = static_cast<double>((p1 > p2) ? (p1 - p2) : (p1 - p2 + m1)) * normc;
    return (st.anti ? 1.0-u : u);
  }

  /**
   * @brief Single-precision variant of U01().
   *
   * Advances the generator by one step and returns a
   * single-precision variate in the interval (0,1).
   *
   * @param st Generator state.
   *
   * @return Uniform random variate.
   */
  HD
  inline float U01f(RngStream& st) {
    Int p1, p2; // signed!
  
    p1 = a12 * static_cast<Int>(st.seed[1]) - a13n * static_cast<Int>(st.seed[0]);
    p1 %= m1;
    if (p1 < 0) p1 += m1;
    st.seed[0] = st.seed[1]; st.seed[1] = st.seed[2]; st.seed[2] = static_cast<UInt>(p1);
  
    p2 = a21 * static_cast<Int>(st.seed[5]) - a23n * static_cast<Int>(st.seed[3]);
    p2 %= m2;
    if (p2 < 0) p2 += m2;
    st.seed[3] = st.seed[4]; st.seed[4] = st.seed[5]; st.seed[5] = static_cast<UInt>(p2);
  
    float u = static_cast<float>((p1 > p2) ? (p1 - p2) : (p1 - p2 + m1)) *
      normcf;
    return (st.anti ? 1.0f-u : u);
  }

  /**
   * @brief Resets the generator to the beginning of the
   * current substream.
   *
   * @param st Generator state.
   */
  HD
  inline void ResetStartSubstream(RngStream& st)
  {
#pragma unroll
    for(int i=0;i<6;i++)
      st.seed[i] = st.start_substream[i];
  }  

  /**
   * @brief Resets the generator to the beginning of the
   * current stream.
   *
   * @param st Generator state.
   */
  HD
  inline void ResetStartStream(RngStream& st)
  {
#pragma unroll
    for(int i=0;i<6;i++)
      st.seed[i] = st.start_substream[i] = st.start_stream[i];
  }  

  /**
   * @brief Advances to the next substream.
   *
   * Equivalent to multiplying the state by the matrices
   * A1p76 and A2p76.
   *
   * The resulting state is positioned at the beginning
   * of the next substream.
   *
   * @param st Generator state.
   */
  HD
  inline void ResetNextSubstream(RngStream& st)
  {
#ifdef __CUDA_ARCH__
    MatVecModM(A1p76_dev, st.start_substream, m1);
    MatVecModM(A2p76_dev, &st.start_substream[3], m2);
#else 
    MatVecModM(A1p76, st.start_substream, m1);
    MatVecModM(A2p76, &st.start_substream[3], m2);
#endif
#pragma unroll
    for (int i = 0;  i < 6; i++)
      st.seed[i] = st.start_substream[i];
  }

  /**
   * @brief Advances to the next stream.
   *
   * Equivalent to multiplying the state by the matrices
   * A1p127 and A2p127.
   *
   * The resulting state is positioned at the beginning
   * of the next stream.
   *
   * @param st Generator state.
   */
  HD
  inline void ResetNextStream(RngStream& st)
  {
#ifdef __CUDA_ARCH__
    MatVecModM(A1p127_dev, st.start_stream, m1);
    MatVecModM(A2p127_dev, &st.start_stream[3], m2);
#else 
    MatVecModM(A1p127, st.start_stream, m1);
    MatVecModM(A2p127, &st.start_stream[3], m2);
#endif
#pragma unroll
    for (int i = 0;  i < 6; i++)
      st.seed[i] = st.start_substream[i] = st.start_stream[i];
  }

  /**
   * @brief Applies a precomputed advancement operator.
   *
   * Advances a three-component state vector by n stream or
   * substream units using binary decomposition.
   *
   * @tparam N Number of available powers.
   *
   * @param state State vector.
   * @param jump Precomputed powers A^(2^k).
   * @param modulus Modulus.
   * @param n Number of units to advance.
   */
  template<int N>
  void ApplyAdvance(
		    UInt state[3],
		    const UInt advance[N][3][3],
		    UInt modulus,
		    uint64_t n)
  {
    for (unsigned k=0; n && k<N; ++k, n >>= 1)
      {
        if (n & 1)
	  MatVecModM(advance[k], state, modulus);
      }
  }

  /**
   * @brief Advances by n substreams.
   *
   * Uses precomputed powers stored in an AdvanceArrays
   * instance to perform the jump in O(log n).
   *
   * @tparam N Number of precomputed powers.
   *
   * @param st Generator state.
   * @param jump Precomputed substream advancement tables.
   * @param n Number of substreams to advance.
   */
  template<int N>
  void AdvanceSubstreams(
			 RngStream& st,
			 const AdvanceArrays<N>& advance,
			 uint64_t n)
  {
    ApplyAdvance<N>(st.start_substream, advance.B1, m1, n);
    ApplyAdvance<N>(st.start_substream + 3, advance.B2, m2, n);
#pragma unroll
    for (int i=0;i<6;i++)
      st.seed[i] = st.start_substream[i];
  }  
  
  /**
   * @brief Advances by n streams.
   *
   * Uses precomputed powers stored in an AdvanceArrays
   * instance to perform the jump in O(log n).
   *
   * @tparam N Number of precomputed powers.
   *
   * @param st Generator state.
   * @param jump Precomputed stream advancement tables.
   * @param n Number of streams to advance.
   */
  template<int N>
  void AdvanceStreams(
		      RngStream& st,
		      const AdvanceArrays<N>& advance,
		      uint64_t n)
  {
    ApplyAdvance<N>(st.start_stream, advance.B1, m1, n);
    ApplyAdvance<N>(st.start_stream + 3, advance.B2, m2, n);
#pragma unroll
    for (int i=0;i<6;i++)
      st.seed[i] = st.start_substream[i] = st.start_stream[i];
  }

  /**
   * @brief Advances by n substreams using the default
   * substream advancement tables.
   *
   * @param st Generator state.
   * @param n Number of substreams to advance.
   */
  void AdvanceSubstreams(RngStream& st, uint64_t n)
  {
    AdvanceSubstreams(st, advance76, n);
  }
  
  /**
   * @brief Advances by n streams using the default
   * stream advancement tables.
   *
   * @param st Generator state.
   * @param n Number of streams to advance.
   */
  void AdvanceStreams(RngStream& st, uint64_t n)
  {
    AdvanceStreams(st, advance127, n);
  }
  
} // namespace rngstream
