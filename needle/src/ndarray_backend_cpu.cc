#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  size_t ndim = shape.size();
  if (ndim == 0) {
    out->ptr[0] = a.ptr[offset];
    return;
  }
  std::vector<int32_t> idx(ndim, 0);
  for (size_t out_i = 0; out_i < out->size; ++out_i) {
    size_t in_pos = offset;
    for (size_t d = 0; d < ndim; ++d) {
      in_pos += static_cast<size_t>(idx[d]) * static_cast<size_t>(strides[d]);
    }
    out->ptr[out_i] = a.ptr[in_pos];
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      idx[d] += 1;
      if (idx[d] < shape[d]) break;
      idx[d] = 0;
    }
  }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */

  size_t ndim = shape.size();
  if (ndim == 0) {
    out->ptr[offset] = a.ptr[0];
    return;
  }
  std::vector<int32_t> idx(ndim, 0);
  size_t total = a.size;
  for (size_t out_i = 0; out_i < total; ++out_i) {
    size_t in_pos = offset;
    for (size_t d = 0; d < ndim; ++d) {
      in_pos += static_cast<size_t>(idx[d]) * static_cast<size_t>(strides[d]);
    }
    out->ptr[in_pos] = a.ptr[out_i];
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      idx[d] += 1;
      if (idx[d] < shape[d]) break;
      idx[d] = 0;
    }
  }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  size_t ndim = shape.size();
  if (ndim == 0) {
    out->ptr[offset] = val;
    return;
  }
  std::vector<int32_t> idx(ndim, 0);
  for (size_t out_i = 0; out_i < size; ++out_i) {
    size_t in_pos = offset;
    for (size_t d = 0; d < ndim; ++d) {
      in_pos += static_cast<size_t>(idx[d]) * static_cast<size_t>(strides[d]);
    }
    out->ptr[in_pos] = val;
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      idx[d] += 1;
      if (idx[d] < shape[d]) break;
      idx[d] = 0;
    }
  }
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

template <typename Op>
static inline void single_l(const AlignedArray& a, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = op(a.ptr[i]);
}

template <typename Op>
static inline void ewise_l(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = op(a.ptr[i], b.ptr[i]);
}

template <typename Op>
static inline void scalar_l(const AlignedArray& a, scalar_t val, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = op(a.ptr[i], val);
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_l(a, b, out, [](scalar_t x, scalar_t y) { return x * y; });
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_l(a, val, out, [](scalar_t x, scalar_t v) { return x * v; });
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_l(a, b, out, [](scalar_t x, scalar_t y) { return x / y; });
}

void EwisePow(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_l(a, b, out, [](scalar_t x, scalar_t y) { return std::pow(x, y); });
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_l(a, val, out, [](scalar_t x, scalar_t v) { return x / v; });
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_l(a, val, out, [](scalar_t x, scalar_t p) { return std::pow(x, p); });
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_l(a, b, out, [](scalar_t x, scalar_t y) { return x > y ? x : y; });
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_l(a, val, out, [](scalar_t x, scalar_t v) { return x > v ? x : v; });
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_l(a, b, out, [](scalar_t x, scalar_t y) { return static_cast<scalar_t>(x == y); });
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_l(a, val, out, [](scalar_t x, scalar_t v) { return static_cast<scalar_t>(x == v); });
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  ewise_l(a, b, out, [](scalar_t x, scalar_t y) { return static_cast<scalar_t>(x >= y); });
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  scalar_l(a, val, out, [](scalar_t x, scalar_t v) { return static_cast<scalar_t>(x >= v); });
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  single_l(a, out, [](scalar_t x) { return std::log(x); });
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  single_l(a, out, [](scalar_t x) { return std::exp(x); });
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  single_l(a, out, [](scalar_t x) { return std::tanh(x); });
}

void EwiseCos(const AlignedArray& a, AlignedArray* out) {
  single_l(a, out, [](scalar_t x) { return std::cos(x); });
}

void EwiseSin(const AlignedArray& a, AlignedArray* out) {
  single_l(a, out, [](scalar_t x) { return std::sin(x); });
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  const float* A = a.ptr;
  const float* B = b.ptr;
  float* C = out->ptr;

  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < p; ++j) {
      float acc = 0.0f;
      const uint32_t arow = i * n;
      const uint32_t bcol = j;
      for (uint32_t k = 0; k < n; ++k) {
        acc += A[arow + k] * B[k * p + bcol];
      }
      C[i * p + j] = acc;
    }
  }
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  for (uint32_t i = 0; i < TILE; ++i) {
    const uint32_t ai = i * TILE;
    const uint32_t oi = i * TILE;
    for (uint32_t k = 0; k < TILE; ++k) {
      const float aik = a[ai + k];
      const uint32_t bk = k * TILE;
      for (uint32_t j = 0; j < TILE; ++j) {
        out[oi + j] += aik * b[bk + j];
      }
    }
  }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  const uint32_t mt = m / TILE;
  const uint32_t nt = n / TILE;
  const uint32_t pt = p / TILE;

  const float* A = a.ptr;
  const float* B = b.ptr;
  float* C = out->ptr;

  auto idxA = [nt](uint32_t it, uint32_t kt, uint32_t ii, uint32_t jj) -> size_t {
    return (((size_t)it * nt + kt) * TILE + ii) * TILE + jj;
  };
  auto idxB = [pt](uint32_t kt, uint32_t jt, uint32_t ii, uint32_t jj) -> size_t {
    return (((size_t)kt * pt + jt) * TILE + ii) * TILE + jj;
  };
  auto idxC = [pt](uint32_t it, uint32_t jt, uint32_t ii, uint32_t jj) -> size_t {
    return (((size_t)it * pt + jt) * TILE + ii) * TILE + jj;
  };

  for (size_t i = 0; i < out->size; ++i) C[i] = 0.0f;

  for (uint32_t it = 0; it < mt; ++it) {
    for (uint32_t jt = 0; jt < pt; ++jt) {
      float* Ctile = &C[idxC(it, jt, 0, 0)];

      for (uint32_t kt = 0; kt < nt; ++kt) {
        const float* Atile = &A[idxA(it, kt, 0, 0)];
        const float* Btile = &B[idxB(kt, jt, 0, 0)];
        AlignedDot(Atile, Btile, Ctile);
      }
    }
  }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  const size_t nblocks = out->size;
  const scalar_t* in = a.ptr;
  scalar_t* o = out->ptr;

  for (size_t b = 0; b < nblocks; ++b) {
    const size_t base = b * reduce_size;
    scalar_t best = in[base];
    for (size_t i = 1; i < reduce_size; ++i) {
      const scalar_t v = in[base + i];
      if (v > best) best = v;
    }
    o[b] = best;
  }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  const size_t nblocks = out->size;
  const scalar_t* in = a.ptr;
  scalar_t* o = out->ptr;

  for (size_t b = 0; b < nblocks; ++b) {
    const size_t base = b * reduce_size;
    scalar_t acc = 0.0f;
    for (size_t i = 0; i < reduce_size; ++i) {
      acc += in[base + i];
    }
    o[b] = acc;
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("ewise_sin", EwiseSin);
  m.def("ewise_cos", EwiseCos);
  m.def("ewise_pow", EwisePow);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
