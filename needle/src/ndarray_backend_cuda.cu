#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cfloat>

#include <iostream>
#include <sstream>

#include <cublas_v2.h>

void cuda_synchronize() {
    cudaDeviceSynchronize();
}

#define CUBLAS_CHECK(stmt)                                                   \
  do {                                                                       \
    cublasStatus_t _status = (stmt);                                         \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                  \
      throw std::runtime_error("cuBLAS error at " #stmt);                    \
    }                                                                        \
  } while (0)

namespace {
cublasHandle_t g_cublas_handle = nullptr;

void ensure_cublas_handle() {
  if (!g_cublas_handle) {
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    // Optional: choose math mode, e.g. TF32, etc.
    // CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
  }
}
}

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
#define BLOCK 32
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

#define DEFINE_EWISE_KERNEL(name, expr)                                             \
__global__ void name##Kernel(const scalar_t* a, const scalar_t* b,                  \
                             scalar_t* out, size_t size) {                          \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                               \
  if (gid < size) out[gid] = (expr);                                                \
}

// #define DEFINE_EWISE_KERNEL(name, expr)                                              \
// __global__ void name##Kernel(const scalar_t* __restrict__ a,                         \
//                              const scalar_t* __restrict__ b,                         \
//                              scalar_t* __restrict__ out, size_t size) {              \
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
//   size_t stride = blockDim.x * gridDim.x;                                            \
//   for (; gid < size; gid += stride) {                                                \
//     out[gid] = (expr);                                                               \
//   }                                                                                  \
// }

#define DEFINE_SCALAR_KERNEL(name, expr)                                            \
__global__ void name##Kernel(const scalar_t* a, scalar_t val,                       \
                             scalar_t* out, size_t size) {                          \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                               \
  if (gid < size) out[gid] = (expr);                                                \
}

// #define DEFINE_SCALAR_KERNEL(name, expr)                                             \
// __global__ void name##Kernel(const scalar_t* __restrict__ a,                         \
//                              scalar_t val,                                           \
//                              scalar_t* __restrict__ out, size_t size) {              \
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
//   size_t stride = blockDim.x * gridDim.x;                                            \
//   for (; gid < size; gid += stride) {                                                \
//     out[gid] = (expr);                                                               \
//   }                                                                                  \
// }

#define DEFINE_SINGLE_KERNEL(name, expr)                                             \
__global__ void name##Kernel(const scalar_t* a, scalar_t* out, size_t size) {       \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                               \
  if (gid < size) out[gid] = (expr);                                                \
}

// #define DEFINE_SINGLE_KERNEL(name, expr)                                             \
// __global__ void name##Kernel(const scalar_t* __restrict__ a,                         \
//                              scalar_t* __restrict__ out, size_t size) {              \
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
//   size_t stride = blockDim.x * gridDim.x;                                            \
//   for (; gid < size; gid += stride) {                                                \
//     out[gid] = (expr);                                                               \
//   }                                                                                  \
// }

#define DEFINE_EWISE_STRIDED_KERNEL(name, expr)                                      \
__global__ void name##StridedKernel(                                                 \
    const scalar_t* __restrict__ a, CudaVec a_strides, size_t a_offset,              \
    const scalar_t* __restrict__ b, CudaVec b_strides, size_t b_offset,              \
    scalar_t* __restrict__ out, CudaVec out_strides, size_t out_offset,              \
    CudaVec shape, size_t total_elems) {                                             \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
  if (gid >= total_elems) return;                                                    \
  size_t a_pos   = index_from_strides(gid, shape, a_strides,   a_offset);           \
  size_t b_pos   = index_from_strides(gid, shape, b_strides,   b_offset);           \
  size_t out_pos = index_from_strides(gid, shape, out_strides, out_offset);         \
  out[out_pos] = (expr);                                                             \
}

// Strided scalar elementwise kernel (only "a" has strides)
#define DEFINE_SCALAR_STRIDED_KERNEL(name, expr)                                     \
__global__ void name##StridedKernel(                                           \
    const scalar_t* __restrict__ a, CudaVec a_strides, size_t a_offset,              \
    scalar_t val,                                                                    \
    scalar_t* __restrict__ out, CudaVec out_strides, size_t out_offset,              \
    CudaVec shape, size_t total_elems) {                                             \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
  if (gid >= total_elems) return;                                                    \
  size_t a_pos   = index_from_strides(gid, shape, a_strides,   a_offset);           \
  size_t out_pos = index_from_strides(gid, shape, out_strides, out_offset);         \
  out[out_pos] = (expr);                                                             \
}

// Strided unary kernel (log, exp, tanh, sin, cos, etc.)
#define DEFINE_UNARY_STRIDED_KERNEL(name, expr)                                      \
__global__ void name##StridedKernel(                                                 \
    const scalar_t* __restrict__ a, CudaVec a_strides, size_t a_offset,              \
    scalar_t* __restrict__ out, CudaVec out_strides, size_t out_offset,              \
    CudaVec shape, size_t total_elems) {                                             \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
  if (gid >= total_elems) return;                                                    \
  size_t a_pos   = index_from_strides(gid, shape, a_strides,   a_offset);           \
  size_t out_pos = index_from_strides(gid, shape, out_strides, out_offset);         \
  out[out_pos] = (expr);                                                             \
}

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

// CudaDims CudaOneDim(size_t size) {
//   CudaDims dim;
//   int block_size = BASE_THREAD_NUM;  // 256
//   // cap grid size to keep it reasonable and ensure enough work per thread
//   int max_blocks = 65535;           // per dimension limit
//   size_t num_blocks = (size + block_size - 1) / block_size;
//   if (num_blocks > max_blocks) num_blocks = max_blocks;
//   dim.block = dim3(block_size, 1, 1);
//   dim.grid  = dim3(num_blocks, 1, 1);
//   return dim;
// }

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}


__device__ inline size_t index_from_strides(size_t gid,
                                            CudaVec shape,
                                            CudaVec strides,
                                            size_t offset) {
  size_t rem = gid;
  size_t pos = offset;
  // Iterate from last dimension to first (row-major)
  for (int d = static_cast<int>(shape.size) - 1; d >= 0; --d) {
    int32_t dim   = shape.data[d];
    int32_t coord = static_cast<int32_t>(rem % static_cast<size_t>(dim));
    rem /= static_cast<size_t>(dim);
    pos += static_cast<size_t>(coord) * static_cast<size_t>(strides.data[d]);
  }
  return pos;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

// __global__ void FillKernel(scalar_t* __restrict__ out, scalar_t val, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   size_t stride = blockDim.x * gridDim.x;
//   for (; gid < size; gid += stride) {
//     out[gid] = val;
//   }
// }

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= size) return;
  size_t rem = gid;
  size_t in_pos = offset;
  for (int d = static_cast<int>(shape.size) - 1; d >= 0; --d) {
    int32_t dim = shape.data[d];
    int32_t coord = static_cast<int32_t>(rem % static_cast<size_t>(dim));
    rem /= static_cast<size_t>(dim);
    in_pos += static_cast<size_t>(coord) * static_cast<size_t>(strides.data[d]);
  }
  out[gid] = a[in_pos];
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* __restrict__ a,
                                   scalar_t* __restrict__ out,
                                   size_t size,
                                   CudaVec shape,
                                   CudaVec strides,
                                   size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;
  size_t rem = gid;
  size_t pos = offset;
  for (int d = static_cast<int>(shape.size) - 1; d >= 0; --d) {
    int32_t dim = shape.data[d];
    int32_t coord = static_cast<int32_t>(rem % static_cast<size_t>(dim));
    rem /= static_cast<size_t>(dim);
    pos += static_cast<size_t>(coord) * static_cast<size_t>(strides.data[d]);
  }
  out[pos] = a[gid];
}

__global__ void ScalarSetitemKernel(scalar_t val,
                                    scalar_t* __restrict__ out,
                                    size_t size,
                                    CudaVec shape,
                                    CudaVec strides,
                                    size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;
  size_t rem = gid;
  size_t pos = offset;
  for (int d = static_cast<int>(shape.size) - 1; d >= 0; --d) {
    int32_t dim = shape.data[d];
    int32_t coord = static_cast<int32_t>(rem % static_cast<size_t>(dim));
    rem /= static_cast<size_t>(dim);
    pos += static_cast<size_t>(coord) * static_cast<size_t>(strides.data[d]);
  }
  out[pos] = val;
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  size_t size = a.size;
  CudaDims dim = CudaOneDim(size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
}



void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(
      val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
}


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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


DEFINE_EWISE_KERNEL(EwiseMul,  a[gid] * b[gid])
DEFINE_SCALAR_KERNEL(ScalarMul, a[gid] * val)
DEFINE_EWISE_KERNEL(EwiseDiv,  a[gid] / b[gid])
DEFINE_EWISE_KERNEL(EwisePow,  powf(a[gid], b[gid]))
DEFINE_SCALAR_KERNEL(ScalarDiv, a[gid] / val)
DEFINE_SCALAR_KERNEL(ScalarPower, powf(a[gid], val))
DEFINE_EWISE_KERNEL(EwiseMaximum,  a[gid] > b[gid] ? a[gid] : b[gid])
DEFINE_SCALAR_KERNEL(ScalarMaximum, a[gid] > val    ? a[gid] : val)
DEFINE_EWISE_KERNEL(EwiseEq,  a[gid] == b[gid] ? 1.0f : 0.0f)
DEFINE_SCALAR_KERNEL(ScalarEq, a[gid] == val    ? 1.0f : 0.0f)
DEFINE_EWISE_KERNEL(EwiseGe,  a[gid] >= b[gid] ? 1.0f : 0.0f)
DEFINE_SCALAR_KERNEL(ScalarGe, a[gid] >= val    ? 1.0f : 0.0f)
DEFINE_SINGLE_KERNEL(EwiseLog,  logf(a[gid]))
DEFINE_SINGLE_KERNEL(EwiseExp,  expf(a[gid]))
DEFINE_SINGLE_KERNEL(EwiseTanh, tanhf(a[gid]))
DEFINE_SINGLE_KERNEL(EwiseCos, cosf(a[gid]))
DEFINE_SINGLE_KERNEL(EwiseSin, sinf(a[gid]))


// Binary ops
DEFINE_EWISE_STRIDED_KERNEL(EwiseAdd,      a[a_pos] +  b[b_pos])
DEFINE_EWISE_STRIDED_KERNEL(EwiseMul,      a[a_pos] *  b[b_pos])
DEFINE_EWISE_STRIDED_KERNEL(EwiseDiv,      a[a_pos] /  b[b_pos])
DEFINE_EWISE_STRIDED_KERNEL(EwiseMaximum,  a[a_pos] >  b[b_pos] ? a[a_pos] : b[b_pos])
DEFINE_EWISE_STRIDED_KERNEL(EwiseEq,       a[a_pos] == b[b_pos] ? 1.0f : 0.0f)
DEFINE_EWISE_STRIDED_KERNEL(EwiseGe,       a[a_pos] >= b[b_pos] ? 1.0f : 0.0f)
DEFINE_EWISE_STRIDED_KERNEL(EwisePow,      powf(a[a_pos], b[b_pos]))

// Scalar ops
DEFINE_SCALAR_STRIDED_KERNEL(ScalarAdd,      a[a_pos] + val)
DEFINE_SCALAR_STRIDED_KERNEL(ScalarMul,      a[a_pos] * val)
DEFINE_SCALAR_STRIDED_KERNEL(ScalarDiv,      a[a_pos] / val)
DEFINE_SCALAR_STRIDED_KERNEL(ScalarMaximum,  a[a_pos] > val ? a[a_pos] : val)
DEFINE_SCALAR_STRIDED_KERNEL(ScalarEq,       a[a_pos] == val ? 1.0f : 0.0f)
DEFINE_SCALAR_STRIDED_KERNEL(ScalarGe,       a[a_pos] >= val ? 1.0f : 0.0f)
DEFINE_SCALAR_STRIDED_KERNEL(ScalarPower,    powf(a[a_pos], val))

// Unary ops
DEFINE_UNARY_STRIDED_KERNEL(EwiseLog,   logf(a[a_pos]))
DEFINE_UNARY_STRIDED_KERNEL(EwiseExp,   expf(a[a_pos]))
DEFINE_UNARY_STRIDED_KERNEL(EwiseTanh,  tanhf(a[a_pos]))
DEFINE_UNARY_STRIDED_KERNEL(EwiseSin,   sinf(a[a_pos]))
DEFINE_UNARY_STRIDED_KERNEL(EwiseCos,   cosf(a[a_pos]))

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
void EwisePow(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwisePowKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
void EwiseCos(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseCosKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
void EwiseSin(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseSinKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}



inline CudaDims CudaOneDimElems(size_t num_elems) {
  return CudaOneDim(num_elems);  // you already have CudaOneDim(size)
}
void EwiseAddStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     const CudaArray& b,
                     const std::vector<int32_t>& b_strides,
                     size_t b_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseAddStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      b.ptr, VecToCuda(b_strides), b_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseMulStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     const CudaArray& b,
                     const std::vector<int32_t>& b_strides,
                     size_t b_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseMulStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      b.ptr, VecToCuda(b_strides), b_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseDivStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     const CudaArray& b,
                     const std::vector<int32_t>& b_strides,
                     size_t b_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseDivStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      b.ptr, VecToCuda(b_strides), b_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseMaximumStrided(const CudaArray& a,
                         const std::vector<int32_t>& shape,
                         const std::vector<int32_t>& a_strides,
                         size_t a_offset,
                         const CudaArray& b,
                         const std::vector<int32_t>& b_strides,
                         size_t b_offset,
                         CudaArray* out,
                         const std::vector<int32_t>& out_strides,
                         size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseMaximumStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      b.ptr, VecToCuda(b_strides), b_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseEqStrided(const CudaArray& a,
                    const std::vector<int32_t>& shape,
                    const std::vector<int32_t>& a_strides,
                    size_t a_offset,
                    const CudaArray& b,
                    const std::vector<int32_t>& b_strides,
                    size_t b_offset,
                    CudaArray* out,
                    const std::vector<int32_t>& out_strides,
                    size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseEqStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      b.ptr, VecToCuda(b_strides), b_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseGeStrided(const CudaArray& a,
                    const std::vector<int32_t>& shape,
                    const std::vector<int32_t>& a_strides,
                    size_t a_offset,
                    const CudaArray& b,
                    const std::vector<int32_t>& b_strides,
                    size_t b_offset,
                    CudaArray* out,
                    const std::vector<int32_t>& out_strides,
                    size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseGeStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      b.ptr, VecToCuda(b_strides), b_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwisePowStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     const CudaArray& b,
                     const std::vector<int32_t>& b_strides,
                     size_t b_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwisePowStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      b.ptr, VecToCuda(b_strides), b_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}
void ScalarAddStrided(const CudaArray& a,
                      const std::vector<int32_t>& shape,
                      const std::vector<int32_t>& a_strides,
                      size_t a_offset,
                      scalar_t val,
                      CudaArray* out,
                      const std::vector<int32_t>& out_strides,
                      size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  ScalarAddStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      val,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void ScalarMulStrided(const CudaArray& a,
                      const std::vector<int32_t>& shape,
                      const std::vector<int32_t>& a_strides,
                      size_t a_offset,
                      scalar_t val,
                      CudaArray* out,
                      const std::vector<int32_t>& out_strides,
                      size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  ScalarMulStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      val,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void ScalarDivStrided(const CudaArray& a,
                      const std::vector<int32_t>& shape,
                      const std::vector<int32_t>& a_strides,
                      size_t a_offset,
                      scalar_t val,
                      CudaArray* out,
                      const std::vector<int32_t>& out_strides,
                      size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  ScalarDivStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      val,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void ScalarMaximumStrided(const CudaArray& a,
                          const std::vector<int32_t>& shape,
                          const std::vector<int32_t>& a_strides,
                          size_t a_offset,
                          scalar_t val,
                          CudaArray* out,
                          const std::vector<int32_t>& out_strides,
                          size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  ScalarMaximumStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      val,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void ScalarEqStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     scalar_t val,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  ScalarEqStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      val,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void ScalarGeStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     scalar_t val,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  ScalarGeStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      val,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void ScalarPowerStrided(const CudaArray& a,
                        const std::vector<int32_t>& shape,
                        const std::vector<int32_t>& a_strides,
                        size_t a_offset,
                        scalar_t val,
                        CudaArray* out,
                        const std::vector<int32_t>& out_strides,
                        size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  ScalarPowerStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      val,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}
void EwiseLogStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseLogStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseExpStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseExpStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseTanhStrided(const CudaArray& a,
                      const std::vector<int32_t>& shape,
                      const std::vector<int32_t>& a_strides,
                      size_t a_offset,
                      CudaArray* out,
                      const std::vector<int32_t>& out_strides,
                      size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseTanhStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseSinStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseSinStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}

void EwiseCosStrided(const CudaArray& a,
                     const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& a_strides,
                     size_t a_offset,
                     CudaArray* out,
                     const std::vector<int32_t>& out_strides,
                     size_t out_offset) {
  size_t total = out->size;
  CudaDims dim = CudaOneDimElems(total);
  EwiseCosStridedKernel<<<dim.grid, dim.block>>>(
      a.ptr, VecToCuda(a_strides), a_offset,
      out->ptr, VecToCuda(out_strides), out_offset,
      VecToCuda(shape), total);
}



////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernelOptimized(const scalar_t* __restrict__ A,
                                      const scalar_t* __restrict__ B,
                                      scalar_t* __restrict__ C,
                                      uint32_t M, uint32_t N, uint32_t P) {
  __shared__ scalar_t As[BLOCK][BLOCK + 1];
  __shared__ scalar_t Bs[BLOCK][BLOCK + 1];
  const uint32_t row = blockIdx.y * BLOCK + threadIdx.y;
  const uint32_t col = blockIdx.x * BLOCK + threadIdx.x;
  scalar_t acc = 0.0f;
  const uint32_t numTiles = (N + BLOCK - 1) / BLOCK;
  for (uint32_t t = 0; t < numTiles; ++t) {
    const uint32_t aCol = t * BLOCK + threadIdx.x;
    const uint32_t bRow = t * BLOCK + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (bRow < N && col < P) ? B[bRow * P + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
      acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
    }
    __syncthreads();
  }
  if (row < M && col < P) {
    C[row * P + col] = acc;
  }
}

// void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   /**
//    * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
//    * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
//    * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
//    * over (i,j) entries in the output array.  However, to really get the full benefit of this
//    * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
//    * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
//    * the CPU backend, here you should implement a single function that works across all size
//    * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
//    * implementations, this function here will largely just set up the kernel call, and you should
//    * implement the logic in a separate MatmulKernel() call.
//    * 
//    *
//    * Args:
//    *   a: compact 2D array of size m x n
//    *   b: comapct 2D array of size n x p
//    *   out: compact 2D array of size m x p to write the output to
//    *   M: rows of a / out
//    *   N: columns of a / rows of b
//    *   P: columns of b / out
//    */
//   dim3 block(BLOCK, BLOCK, 1);
//   dim3 grid((P + BLOCK - 1) / BLOCK,
//             (M + BLOCK - 1) / BLOCK,
//             1);
//   MatmulKernelOptimized<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
// }

void MatmulFallback(const CudaArray& a, const CudaArray& b, CudaArray* out,
                    uint32_t M, uint32_t N, uint32_t P) {
  dim3 block(BLOCK, BLOCK, 1);
  dim3 grid((P + BLOCK - 1) / BLOCK,
            (M + BLOCK - 1) / BLOCK,
            1);
  MatmulKernelOptimized<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out,
            uint32_t M, uint32_t N, uint32_t P) {
  // A: M x N (row-major)
  // B: N x P (row-major)
  // C: M x P (row-major)
  {
    ensure_cublas_handle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Standard row-major trick:
    // cublasSgemm(handle, N, N,
    //             n=P, m=M, k=N,
    //             &alpha,
    //             B, lda = n = P,
    //             A, ldb = k = N,
    //             &beta,
    //             C, ldc = n = P);
    CUBLAS_CHECK(cublasSgemm(
        g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)P,              // m (rows of op(A))
        (int)M,              // n (cols of op(B))
        (int)N,              // k
        &alpha,
        b.ptr, (int)P,       // "A" in BLAS: B, lda = P
        a.ptr, (int)N,       // "B" in BLAS: A, ldb = N
        &beta,
        out->ptr, (int)P));  // C, ldc = P
  }
}

void MatmulBatchedFallback(const CudaArray& a, const CudaArray& b, CudaArray* out,
                           uint32_t batch, uint32_t M, uint32_t N, uint32_t P) {
  dim3 block(BLOCK, BLOCK, 1);
  dim3 grid((P + BLOCK - 1) / BLOCK,
            (M + BLOCK - 1) / BLOCK,
            1);

  const size_t strideA = (size_t)M * N;
  const size_t strideB = (size_t)N * P;
  const size_t strideC = (size_t)M * P;

  for (uint32_t bi = 0; bi < batch; ++bi) {
    const scalar_t* A_ptr = a.ptr + bi * strideA;
    const scalar_t* B_ptr = b.ptr + bi * strideB;
    scalar_t* C_ptr       = out->ptr + bi * strideC;
    MatmulKernelOptimized<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, P);
  }
}

void MatmulBatched(const CudaArray& a, const CudaArray& b, CudaArray* out,
                   uint32_t batch, uint32_t M, uint32_t N, uint32_t P) {
  {
    ensure_cublas_handle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    long long strideA = (long long)M * N;  // for "B" in BLAS
    long long strideB = (long long)N * P;  // for "A" in BLAS
    long long strideC = (long long)M * P;

    // Same row-major trick as single matmul, but strided batched:
    // A (BLAS) = B row-major tensor
    // B (BLAS) = A row-major tensor
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)P,                 // m
        (int)M,                 // n
        (int)N,                 // k
        &alpha,
        b.ptr, (int)P, strideB, // "A" in BLAS: B, lda = P
        a.ptr, (int)N, strideA, // "B" in BLAS: A, ldb = N
        &beta,
        out->ptr, (int)P, strideC,
        (int)batch));
  }
}

__global__ void Conv2dNHWCKernel(const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ B,
                                 scalar_t* __restrict__ Y,
                                 int N, int H, int W, int C_in,
                                 int K, int C_out,
                                 int stride, int padding,
                                 int H_out, int W_out) {
  // total number of output elements
  size_t out_size = (size_t)N * H_out * W_out * C_out;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride_threads = (size_t)blockDim.x * gridDim.x;

  for (; gid < out_size; gid += stride_threads) {
    // Decode flat index -> (n, oh, ow, co)
    int co = gid % C_out;
    size_t tmp = gid / C_out;
    int ow = tmp % W_out;
    tmp /= W_out;
    int oh = tmp % H_out;
    int n = tmp / H_out;

    scalar_t acc = 0.0f;

    // Loop over kernel + input channels
    for (int kh = 0; kh < K; ++kh) {
      int ih = oh * stride + kh - padding;
      if (ih < 0 || ih >= H) continue;

      for (int kw = 0; kw < K; ++kw) {
        int iw = ow * stride + kw - padding;
        if (iw < 0 || iw >= W) continue;

        // base index for input pixel (n, ih, iw, 0)
        size_t in_base = (((size_t)n * H + ih) * W + iw) * C_in;

        // base index for weights at (kh, kw, 0, co)
        size_t w_spatial = ((size_t)kh * K + kw) * C_in * C_out;

        for (int ci = 0; ci < C_in; ++ci) {
          scalar_t a_val = A[in_base + ci];
          scalar_t w_val = B[w_spatial + (size_t)ci * C_out + co];
          acc = fmaf(a_val, w_val, acc);
        }
      }
    }

    Y[gid] = acc;
  }
}

void Conv2d(const CudaArray& A, const CudaArray& B, CudaArray* out,
            int N, int H, int W, int C_in,
            int K, int C_out,
            int stride, int padding) {
  // infer output spatial size from input/stride/padding
  int Hp = H + 2 * padding;
  int Wp = W + 2 * padding;
  int H_out = (Hp - K) / stride + 1;
  int W_out = (Wp - K) / stride + 1;

  size_t expected_out_size = (size_t)N * H_out * W_out * C_out;
  // (you can add an assert here if you like)
  // assert(out->size == expected_out_size);

  CudaDims dim = CudaOneDim(expected_out_size);
  Conv2dNHWCKernel<<<dim.grid, dim.block>>>(
      A.ptr, B.ptr, out->ptr,
      N, H, W, C_in,
      K, C_out,
      stride, padding,
      H_out, W_out);
}

__global__ void ConvTranspose2dNHWCKernel(const scalar_t* __restrict__ A,
                                          const scalar_t* __restrict__ B,
                                          scalar_t* __restrict__ Y,
                                          int N, int H_in, int W_in, int C_out,
                                          int K, int C_in,
                                          int stride, int padding,
                                          int H_out, int W_out) {
  // Each thread computes one output element: (n, oh, ow, ci)
  size_t out_size = (size_t)N * H_out * W_out * C_in;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride_threads = (size_t)blockDim.x * gridDim.x;

  for (; gid < out_size; gid += stride_threads) {
    int ci = gid % C_in;
    size_t tmp = gid / C_in;
    int ow = tmp % W_out;
    tmp /= W_out;
    int oh = tmp % H_out;
    int n = tmp / H_out;

    scalar_t acc = 0.0f;

    // We want to reproduce exactly:
    // for ih, iw, kh, kw, co:
    //   oh = ih*stride - padding + kh
    //   ow = iw*stride - padding + kw
    //   Y[n,oh,ow,ci] += A[n,ih,iw,co] * B[kh,kw,ci,co]

    // Rewrite for gather:
    // For a given (oh, ow, kh, kw),
    //   ih = (oh - kh + padding) / stride
    //   iw = (ow - kw + padding) / stride
    //   must be integer and within [0, H_in), [0, W_in).

    for (int kh = 0; kh < K; ++kh) {
      int tmp_h = oh - kh + padding;
      if (tmp_h < 0) continue;
      if (tmp_h % stride != 0) continue;
      int ih = tmp_h / stride;
      if (ih < 0 || ih >= H_in) continue;

      for (int kw = 0; kw < K; ++kw) {
        int tmp_w = ow - kw + padding;
        if (tmp_w < 0) continue;
        if (tmp_w % stride != 0) continue;
        int iw = tmp_w / stride;
        if (iw < 0 || iw >= W_in) continue;

        // A: (N, H_in, W_in, C_out)
        size_t a_base = (((size_t)n * H_in + ih) * W_in + iw) * C_out;

        // B: (K, K, C_in, C_out)
        size_t b_spatial = ((size_t)kh * K + kw) * C_in * C_out;
        size_t b_base = b_spatial + (size_t)ci * C_out;

        for (int co = 0; co < C_out; ++co) {
          scalar_t a_val = A[a_base + co];
          scalar_t w_val = B[b_base + co];
          acc = fmaf(a_val, w_val, acc);
        }
      }
    }

    // Y: (N, H_out, W_out, C_in)
    size_t y_idx = (((size_t)n * H_out + oh) * W_out + ow) * C_in + ci;
    Y[y_idx] = acc;
  }
}

void ConvTranspose2d(const CudaArray& A, const CudaArray& B, CudaArray* out,
                     int N, int H_in, int W_in, int C_out,
                     int K, int C_in,
                     int stride, int padding) {
  // Output spatial size (same formula as naive Python version)
  int H_out = (H_in - 1) * stride - 2 * padding + K;
  int W_out = (W_in - 1) * stride - 2 * padding + K;

  size_t expected_out_size = (size_t)N * H_out * W_out * C_in;
  // Optional sanity check:
  // assert(out->size == expected_out_size);

  CudaDims dim = CudaOneDim(expected_out_size);
  ConvTranspose2dNHWCKernel<<<dim.grid, dim.block>>>(
      A.ptr, B.ptr, out->ptr,
      N, H_in, W_in, C_out,
      K, C_in,
      stride, padding,
      H_out, W_out);
}



////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

template <typename Op>
__device__ scalar_t blockReduce(scalar_t val, Op op, scalar_t init) {
  extern __shared__ scalar_t sdata[];
  int tid = threadIdx.x;
  sdata[tid] = val;
  __syncthreads();

  // reduce in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = op(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  return sdata[0];
}

struct MaxOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a > b ? a : b;
  }
};

struct SumOp {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a + b;
  }
};

__global__ void ReduceMaxKernel(const scalar_t* __restrict__ a,
                                scalar_t* __restrict__ out,
                                size_t out_size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  const size_t base = gid * reduce_size;
  scalar_t best = a[base];
  for (size_t i = 1; i < reduce_size; ++i) {
    scalar_t v = a[base + i];
    if (v > best) best = v;
  }
  out[gid] = best;
}
__global__ void ReduceSumKernel(const scalar_t* __restrict__ a,
                                scalar_t* __restrict__ out,
                                size_t out_size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  const size_t base = gid * reduce_size;
  scalar_t acc = 0.0f;
  for (size_t i = 0; i < reduce_size; ++i) {
    acc += a[base + i];
  }
  out[gid] = acc;
}

// __global__ void ReduceMaxKernel(const scalar_t* __restrict__ a,
//                                 scalar_t* __restrict__ out,
//                                 size_t out_size,
//                                 size_t reduce_size) {
//   // one block per output element
//   size_t out_id = blockIdx.x;
//   if (out_id >= out_size) return;

//   size_t base = out_id * reduce_size;
//   size_t tid  = threadIdx.x;
//   size_t stride = blockDim.x;

//   scalar_t local = -FLT_MAX;
//   // each thread accumulates a chunk
//   for (size_t i = tid; i < reduce_size; i += stride) {
//     scalar_t v = a[base + i];
//     local = v > local ? v : local;
//   }

//   scalar_t res = blockReduce(local, MaxOp(), -FLT_MAX);
//   if (tid == 0) out[out_id] = res;
// }

// __global__ void ReduceSumKernel(const scalar_t* __restrict__ a,
//                                 scalar_t* __restrict__ out,
//                                 size_t out_size,
//                                 size_t reduce_size) {
//   size_t out_id = blockIdx.x;
//   if (out_id >= out_size) return;

//   size_t base = out_id * reduce_size;
//   size_t tid  = threadIdx.x;
//   size_t stride = blockDim.x;

//   scalar_t local = 0.0f;
//   for (size_t i = tid; i < reduce_size; i += stride) {
//     local += a[base + i];
//   }

//   scalar_t res = blockReduce(local, SumOp(), 0.0f);
//   if (tid == 0) out[out_id] = res;
// }

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  const size_t out_size = out->size;
  CudaDims dim = CudaOneDim(out_size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out_size, reduce_size);
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  const size_t out_size = out->size;
  CudaDims dim = CudaOneDim(out_size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out_size, reduce_size);
}

// void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
//   const size_t out_size = out->size;
//   int block_size = BASE_THREAD_NUM;  // 256
//   int grid_size  = static_cast<int>(out_size);
//   size_t shmem   = block_size * sizeof(scalar_t);
//   ReduceMaxKernel<<<grid_size, block_size, shmem>>>(a.ptr, out->ptr, out_size, reduce_size);
// }

// void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
//   const size_t out_size = out->size;
//   int block_size = BASE_THREAD_NUM;
//   int grid_size  = static_cast<int>(out_size);
//   size_t shmem   = block_size * sizeof(scalar_t);
//   ReduceSumKernel<<<grid_size, block_size, shmem>>>(a.ptr, out->ptr, out_size, reduce_size);
// }

__global__ void SoftmaxLastDimKernel(const scalar_t* __restrict__ X,
                                     scalar_t* __restrict__ Y,
                                     size_t outer_size,   // number of rows
                                     size_t dim) {        // length of last dimension
  // One block per row (outer index)
  size_t row = blockIdx.x;
  if (row >= outer_size) return;

  extern __shared__ scalar_t sdata[];  // shared buffer: at least dim elements
  scalar_t* smax = sdata;              // [dim]
  scalar_t* sexp = sdata;              // re-use the same storage

  // Pointer to this row
  const scalar_t* x_row = X + row * dim;
  scalar_t* y_row       = Y + row * dim;

  // 1. Compute max over the row (parallel reduction)
  scalar_t thread_max = -FLT_MAX;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    scalar_t v = x_row[i];
    if (v > thread_max) thread_max = v;
  }

  // Reduce within block
  // store thread_max into shared
  smax[threadIdx.x] = thread_max;
  __syncthreads();

  // warp-ish reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (smax[threadIdx.x + s] > smax[threadIdx.x]) {
        smax[threadIdx.x] = smax[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
  scalar_t row_max = smax[0];

  // 2. Compute exp(x - max) and accumulate sum
  scalar_t thread_sum = 0.0f;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    scalar_t v = x_row[i] - row_max;
    scalar_t ev = expf(v);
    y_row[i] = ev;      // temporary store exp into output buffer
    thread_sum += ev;
  }

  // Reduce sums
  smax[threadIdx.x] = thread_sum;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smax[threadIdx.x] += smax[threadIdx.x + s];
    }
    __syncthreads();
  }
  scalar_t row_sum = smax[0];

  // 3. Normalize
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    y_row[i] = y_row[i] / row_sum;
  }
}

void SoftmaxLastDim(const CudaArray& X, CudaArray* out, size_t dim) {
  // X is (..., dim)
  size_t total = X.size;
  size_t outer = total / dim;  // number of rows

  CudaDims dim1 = CudaOneDim(outer);
  // We want one block per row, but CudaOneDim gives us a generic grid.
  // For simplicity, override grid/block here:
  const int threads = 256;
  dim3 grid((unsigned int)outer, 1, 1);
  dim3 block(threads, 1, 1);

  size_t shmem = threads * sizeof(scalar_t);
  SoftmaxLastDimKernel<<<grid, block, shmem>>>(X.ptr, out->ptr, outer, dim);
}



__global__
void flash_attn_forward_kernel(
    const float* __restrict__ Q,   // [B, H, N, D] compact
    const float* __restrict__ K,   // [B, H, N, D] compact
    const float* __restrict__ V,   // [B, H, N, D] compact
    int N,                         // sequence length
    int d,                         // head dim
    int Tc,                        // number of key tiles  = ceil(N / Bc)
    int Tr,                        // number of query tiles = ceil(N / Br)
    int Bc,                        // tile size in columns (keys)
    int Br,                        // tile size in rows (queries)
    float softmax_scale,           // use 1.f if Q is already scaled
    float* __restrict__ O          // [B, H, N, D]
) {
    int b  = blockIdx.x;           // batch index
    int h  = blockIdx.y;           // head index
    int tx = threadIdx.x;          // 0 .. Br-1, each thread = one query row within a tile

    int nh = gridDim.y;

    // Base offsets for this (b, h) pair, assuming layout [B, H, N, D]
    int qkv_base = (b * nh + h) * N * d;   // offset into Q/K/V/O

    // Shared memory layout:
    //  Kj: [Bc, d]
    //  Vj: [Bc, d]
    //  S : [Br, Bc]  (scores / probs for one (row-tile, col-tile))
    extern __shared__ float sram[];
    float* Kj = sram;                 // Bc * d
    float* Vj = Kj + Bc * d;          // Bc * d
    float* S  = Vj + Bc * d;          // Br * Bc

    // Loop over row tiles (queries)
    for (int i = 0; i < Tr; ++i) {
        int row = i * Br + tx;        // global query index
        bool valid_row = (row < N);

        // Per-row online-softmax state over *all* K-tiles
        float row_m_prev = -INFINITY; // running max of logits
        float row_l_prev = 0.f;       // running "L" (sum of scaled exps)

        // Pointer to this query row if valid
        const float* q_row = valid_row
            ? (Q + qkv_base + row * d)
            : nullptr;

        // Loop over column tiles (keys/values)
        for (int j = 0; j < Tc; ++j) {
            int col0 = j * Bc;
            int cols = min(Bc, N - col0);   // number of valid columns in this tile
            if (cols <= 0) break;

            // --- Load K_j and V_j tile into shared memory ---
            // Use all Br threads in the block to cooperatively load cols*d values
            for (int idx = tx; idx < cols * d; idx += Br) {
                int y   = idx / d;      // 0 .. cols-1 (tile column index)
                int x   = idx % d;      // feature dim
                int col = col0 + y;     // global key index

                Kj[y * d + x] = K[qkv_base + col * d + x];
                Vj[y * d + x] = V[qkv_base + col * d + x];
            }
            __syncthreads();

            if (valid_row) {
                // --- Compute logits S(row, y) = <q_row, k_y> for this tile ---
                float row_m = -INFINITY;
                float* row_S = S + tx * Bc;  // scores/probs slice for this row

                for (int y = 0; y < cols; ++y) {
                    float sum = 0.f;
                    const float* k_row = Kj + y * d;

                    // dot product q[row, :] · k[col, :]
                    for (int x = 0; x < d; ++x) {
                        float qx = q_row[x];
                        float kx = k_row[x];
                        sum += qx * kx;
                    }

                    // either Q is pre-scaled, or do it here:
                    sum *= softmax_scale;

                    row_S[y] = sum;
                    row_m = fmaxf(row_m, sum);
                }

                // --- P = exp(S - row_m), row_l = sum P ---
                float row_l = 0.f;
                for (int y = 0; y < cols; ++y) {
                    float val = __expf(row_S[y] - row_m);
                    row_S[y] = val;
                    row_l += val;
                }

                // --- Online softmax merge with previous tiles ---
                float row_m_new   = fmaxf(row_m_prev, row_m);
                float alpha_prev  = (row_m_prev == -INFINITY)
                                      ? 0.f
                                      : __expf(row_m_prev - row_m_new);
                float alpha_curr  = __expf(row_m      - row_m_new);
                float row_l_new   = alpha_prev * row_l_prev + alpha_curr * row_l;

                // --- Update O(row, :) for this row ---
                float* o_row = O + qkv_base + row * d;

                for (int x = 0; x < d; ++x) {
                    // P_tile V_tile
                    float pv = 0.f;
                    for (int y = 0; y < cols; ++y) {
                        pv += row_S[y] * Vj[y * d + x];
                    }

                    float old_o = (row_l_prev > 0.f) ? o_row[x] : 0.f;

                    float new_o = (alpha_prev * row_l_prev * old_o + alpha_curr * pv)
                                  / row_l_new;

                    o_row[x] = new_o;
                }

                // Roll forward the online-softmax state
                row_m_prev = row_m_new;
                row_l_prev = row_l_new;
            }

            __syncthreads();
        }
    }
}

__global__
void init_zero_kernel(float* __restrict__ x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        x[idx] = 0.0f;
    }
}

void FlashAttentionForward(
    const CudaArray& q,
    const CudaArray& k,
    const CudaArray& v,
    CudaArray* out,
    int B,
    int nh,
    int N,
    int d
) {
    const int Bc = 32;
    const int Br = 32;

    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;

    // Initialize O = 0
    {
        int total   = static_cast<int>(out->size);
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        init_zero_kernel<<<blocks, threads>>>(out->ptr, total);
    }

    // Shared memory size: Kj + Vj + S
    int sram_size =
        (2 * Bc * d +     // Kj, Vj
         Br * Bc) * sizeof(float);  // S

    dim3 grid_dim(B, nh);   // (batch, heads)
    dim3 block_dim(Br);     // one thread per query row in a row-tile

    flash_attn_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        q.ptr, k.ptr, v.ptr,
        N, d,
        Tc, Tr,
        Bc, Br,
        1.0f,
        out->ptr
    );
}




}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
  m.def("matmul_batched", MatmulBatched);

  m.def("conv2d", Conv2d);
  m.def("conv_transpose2d", ConvTranspose2d);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("cuda_synchronize", &cuda_synchronize, "Synchronize CUDA device");

  m.def("softmax_lastdim", &SoftmaxLastDim);
  


    // Binary strided ewise
  m.def("ewise_add_strided",     &EwiseAddStrided);
  m.def("ewise_mul_strided",     &EwiseMulStrided);
  m.def("ewise_div_strided",     &EwiseDivStrided);
  m.def("ewise_maximum_strided", &EwiseMaximumStrided);
  m.def("ewise_eq_strided",      &EwiseEqStrided);
  m.def("ewise_ge_strided",      &EwiseGeStrided);
  m.def("ewise_pow_strided",     &EwisePowStrided);

  // Scalar strided
  m.def("scalar_add_strided",    &ScalarAddStrided);
  m.def("scalar_mul_strided",    &ScalarMulStrided);
  m.def("scalar_div_strided",    &ScalarDivStrided);
  m.def("scalar_maximum_strided",&ScalarMaximumStrided);
  m.def("scalar_eq_strided",     &ScalarEqStrided);
  m.def("scalar_ge_strided",     &ScalarGeStrided);
  m.def("scalar_power_strided",  &ScalarPowerStrided);

  // Unary strided
  m.def("ewise_log_strided",     &EwiseLogStrided);
  m.def("ewise_exp_strided",     &EwiseExpStrided);
  m.def("ewise_tanh_strided",    &EwiseTanhStrided);
  m.def("ewise_sin_strided",     &EwiseSinStrided);
  m.def("ewise_cos_strided",     &EwiseCosStrided);

  m.def("flash_attention_forward",
      &FlashAttentionForward,
      "FlashAttention forward",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("out"),
      py::arg("B"),
      py::arg("nh"),
      py::arg("N"),
      py::arg("d"));

}
