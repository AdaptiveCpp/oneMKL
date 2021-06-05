#ifndef _MKL_BLAS_ROCBLAS_TASK_HPP_
#define _MKL_BLAS_ROCBLAS_TASK_HPP_
#include <rocblas.h>
#include <complex>
#include <CL/sycl.hpp>
#include "oneapi/mkl/types.hpp"
#ifndef __HIPSYCL__
#include "rocblas_scope_handle.hpp"
#include <CL/sycl/detail/pi.hpp>
#else
#include "rocblas_scope_handle_hipsycl.hpp"
namespace cl::sycl {
using interop_handler = cl::sycl::interop_handle;
}
#endif
namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {

#ifdef __HIPSYCL__
template <typename H, typename F>
static inline void host_task_internal(H &cgh, cl::sycl::queue queue, F f) {
    cgh.hipSYCL_enqueue_custom_operation([f, queue](cl::sycl::interop_handle ih){
        auto sc = RocblasScopedContextHandler(queue, ih);
        f(sc);
    });
}
#else
template <typename H, typename F>
static inline void host_task_internal(H &cgh, cl::sycl::queue queue, F f) {
    cgh.interop_task([f, queue](cl::sycl::interop_handler ih){
        auto sc = RocblasScopedContextHandler(queue, ih);
        f(sc);
    });
}
#endif
template <typename H, typename F>
static inline void onemkl_rocblas_host_task(H &cgh, cl::sycl::queue queue, F f) {
    (void)host_task_internal(cgh, queue, f);
}

} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif // _MKL_BLAS_ROCBLAS_TASK_HPP_