/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#include "rocblas_helper.hpp"
#include "rocblas_task.hpp"

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/rocblas/onemkl_blas_rocblas.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {
namespace column_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemm_batch(Func func, cl::sycl::queue &queue, transpose transa, transpose transb,
                       int64_t m, int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> &a,
                       int64_t lda, int64_t stride_a, cl::sycl::buffer<T, 1> &b, int64_t ldb,
                       int64_t stride_b, T beta, cl::sycl::buffer<T, 1> &c, int64_t ldc,
                       int64_t stride_c, int64_t batch_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = sc.get_mem<cuDataType *>(a_acc);
            auto b_ = sc.get_mem<cuDataType *>(b_acc);
            auto c_ = sc.get_mem<cuDataType *>(c_acc);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_operation(transa),
                              get_rocblas_operation(transb), m, n, k, (cuDataType *)&alpha, a_, lda,
                              stride_a, b_, ldb, stride_b, (cuDataType *)&beta, c_, ldc, stride_c,
                              batch_size);
        });
    });
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                          \
    void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,         \
                    int64_t n, int64_t k, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,   \
                    int64_t stride_a, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t stride_b, \
                    TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stride_c,        \
                    int64_t batch_size) {                                                          \
        gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b,     \
                   ldb, stride_b, beta, c, ldc, stride_c, batch_size);                             \
    }

GEMM_STRIDED_BATCH_LAUNCHER(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

// USM APIs

template <typename Func, typename T>
inline cl::sycl::event gemm_batch(Func func, cl::sycl::queue &queue, transpose transa,
                                  transpose transb, int64_t m, int64_t n, int64_t k, T alpha,
                                  const T *a, int64_t lda, int64_t stride_a, const T *b,
                                  int64_t ldb, int64_t stride_b, T beta, T *c, int64_t ldc,
                                  int64_t stride_c, int64_t batch_size,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<const cuDataType *>(a);
            auto b_ = reinterpret_cast<const cuDataType *>(b);
            auto c_ = reinterpret_cast<cuDataType *>(c);
            rocblas_status err;
            ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_operation(transa),
                              get_rocblas_operation(transb), m, n, k, (cuDataType *)&alpha, a_, lda,
                              stride_a, b_, ldb, stride_b, (cuDataType *)&beta, c_, ldc, stride_c,
                              batch_size);
        });
    });
    return done;
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                      \
    cl::sycl::event gemm_batch(                                                                    \
        cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,          \
        int64_t k, TYPE alpha, const TYPE *a, int64_t lda, int64_t stride_a, const TYPE *b,        \
        int64_t ldb, int64_t stride_b, TYPE beta, TYPE *c, int64_t ldc, int64_t stride_c,          \
        int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {         \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stride_a, \
                          b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);     \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event gemm_batch(Func func, cl::sycl::queue &queue, transpose *transa,
                                  transpose *transb, int64_t *m, int64_t *n, int64_t *k, T *alpha,
                                  const T **a, int64_t *lda, const T **b, int64_t *ldb, T *beta,
                                  T **c, int64_t *ldc, int64_t group_count, int64_t *group_size,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], k[i], lda[i], ldb[i], ldc[i], group_size[i]);
    }
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas_host_task(cgh, queue,[=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            int64_t offset = 0;
            rocblas_status err;
            for (int64_t i = 0; i < group_count; i++) {
                auto **a_ = reinterpret_cast<const cuDataType **>(a);
                auto **b_ = reinterpret_cast<const cuDataType **>(b);
                auto **c_ = reinterpret_cast<cuDataType **>(c);
                ROCBLAS_ERROR_FUNC(func, err, handle, get_rocblas_operation(transa[i]),
                                  get_rocblas_operation(transb[i]), (int)m[i], (int)n[i], (int)k[i],
                                  (cuDataType *)&alpha[i], a_ + offset, (int)lda[i], b_ + offset,
                                  (int)ldb[i], (cuDataType *)&beta[i], c_ + offset, (int)ldc[i],
                                  (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });
    return done;
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,     \
                               int64_t *m, int64_t *n, int64_t *k, TYPE *alpha, const TYPE **a,  \
                               int64_t *lda, const TYPE **b, int64_t *ldb, TYPE *beta, TYPE **c, \
                               int64_t *ldc, int64_t group_count, int64_t *group_size,           \
                               const cl::sycl::vector_class<cl::sycl::event> &dependencies) {    \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, \
                          beta, c, ldc, group_count, group_size, dependencies);                  \
    }

GEMM_BATCH_LAUNCHER_USM(float, rocblas_sgemm_batched)
GEMM_BATCH_LAUNCHER_USM(double, rocblas_dgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_batched)

#undef GEMM_BATCH_LAUNCHER_USM

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

} // namespace column_major
namespace row_major {

// Buffer APIs

template <typename Func, typename T>
inline void gemm_batch(Func func, cl::sycl::queue &queue, transpose transa, transpose transb,
                       int64_t m, int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> &a,
                       int64_t lda, int64_t stride_a, cl::sycl::buffer<T, 1> &b, int64_t ldb,
                       int64_t stride_b, T beta, cl::sycl::buffer<T, 1> &c, int64_t ldc,
                       int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                          \
    void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,         \
                    int64_t n, int64_t k, TYPE alpha, cl::sycl::buffer<TYPE, 1> &a, int64_t lda,   \
                    int64_t stride_a, cl::sycl::buffer<TYPE, 1> &b, int64_t ldb, int64_t stride_b, \
                    TYPE beta, cl::sycl::buffer<TYPE, 1> &c, int64_t ldc, int64_t stride_c,        \
                    int64_t batch_size) {                                                          \
        gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b,     \
                   ldb, stride_b, beta, c, ldc, stride_c, batch_size);                             \
    }

GEMM_STRIDED_BATCH_LAUNCHER(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

// USM APIs

template <typename Func, typename T>
inline cl::sycl::event gemm_batch(Func func, cl::sycl::queue &queue, transpose transa,
                                  transpose transb, int64_t m, int64_t n, int64_t k, T alpha,
                                  const T *a, int64_t lda, int64_t stride_a, const T *b,
                                  int64_t ldb, int64_t stride_b, T beta, T *c, int64_t ldc,
                                  int64_t stride_c, int64_t batch_size,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                      \
    cl::sycl::event gemm_batch(                                                                    \
        cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,          \
        int64_t k, TYPE alpha, const TYPE *a, int64_t lda, int64_t stride_a, const TYPE *b,        \
        int64_t ldb, int64_t stride_b, TYPE beta, TYPE *c, int64_t ldc, int64_t stride_c,          \
        int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {         \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, stride_a, \
                          b, ldb, stride_b, beta, c, ldc, stride_c, batch_size, dependencies);     \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(float, rocblas_sgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, rocblas_dgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_strided_batched)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_strided_batched)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline cl::sycl::event gemm_batch(Func func, cl::sycl::queue &queue, transpose *transa,
                                  transpose *transb, int64_t *m, int64_t *n, int64_t *k, T *alpha,
                                  const T **a, int64_t *lda, const T **b, int64_t *ldb, T *beta,
                                  T **c, int64_t *ldc, int64_t group_count, int64_t *group_size,
                                  const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                            \
    cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,     \
                               int64_t *m, int64_t *n, int64_t *k, TYPE *alpha, const TYPE **a,  \
                               int64_t *lda, const TYPE **b, int64_t *ldb, TYPE *beta, TYPE **c, \
                               int64_t *ldc, int64_t group_count, int64_t *group_size,           \
                               const cl::sycl::vector_class<cl::sycl::event> &dependencies) {    \
        return gemm_batch(ROCBLAS_ROUTINE, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, \
                          beta, c, ldc, group_count, group_size, dependencies);                  \
    }

GEMM_BATCH_LAUNCHER_USM(float, rocblas_sgemm_batched)
GEMM_BATCH_LAUNCHER_USM(double, rocblas_dgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, rocblas_cgemm_batched)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, rocblas_zgemm_batched)

#undef GEMM_BATCH_LAUNCHER_USM

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

} // namespace row_major
} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
