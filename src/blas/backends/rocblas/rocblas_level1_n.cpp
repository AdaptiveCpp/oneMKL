/***************************************************************************
*  copyright (c) codeplay software limited
*  licensed under the apache license, version 2.0 (the "license");
*  you may not use this file except in compliance with the license.
*  you may obtain a copy of the license at
*
*      http://www.apache.org/licenses/license-2.0
*
*  for your convenience, a copy of the license has been included in this
*  repository.
*
*  unless required by applicable law or agreed to in writing, software
*  distributed under the license is distributed on an "as is" basis,
*  without warranties or conditions of any kind, either express or implied.
*  see the license for the specific language governing permissions and
*  limitations under the license.
*
**************************************************************************/
#include "rocblas__helper.hpp"
#include "rocblas__task.hpp"

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/rocblas_/onemkl_blas_rocblas.hpp"


namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas_ {
namespace column_major {

// buffer apis

// level 1
template <typename func, typename t1, typename t2>
inline void asum(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t1, 1> &x,
                 const int64_t incx, cl::sycl::buffer<t2, 1> &result) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    overflow_check(n, incx);

    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);
            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<cudatatype1 *>(x_acc);
            auto res_ = sc.get_mem<cudatatype2 *>(res_acc);
            rocblas__status err;
            // asum does not support negative index
            rocblas_error_func(func, err, handle, n, x_, std::abs(incx), res_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define asum_launcher(type1, type2, rocblas_routine)                             \
    void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type1, 1> &x, \
              const int64_t incx, cl::sycl::buffer<type2, 1> &result) {         \
        asum(rocblas_routine, queue, n, x, incx, result);                        \
    }
asum_launcher(float, float, rocblas_sasum)
asum_launcher(double, double, rocblas_dasum)
asum_launcher(std::complex<float>, float, rocblas_scasum)
asum_launcher(std::complex<double>, double, rocblas_dzasum)
#undef asum_launcher

template <typename func, typename t1, typename t2>
inline void scal(func func, cl::sycl::queue &queue, int64_t n, t1 a, cl::sycl::buffer<t2, 1> &x,
                 int64_t incx) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    overflow_check(n, incx);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = sc.get_mem<cudatatype2 *>(x_acc);
            rocblas__status err;
            // scal does not support negative incx
            rocblas_error_func(func, err, handle, n, (cudatatype1 *)&a, x_, std::abs(incx));
        });
    });
}

#define scal_launcher(type1, type2, rocblas_routine)                                      \
    void scal(cl::sycl::queue &queue, int64_t n, type1 a, cl::sycl::buffer<type2, 1> &x, \
              int64_t incx) {                                                            \
        scal(rocblas_routine, queue, n, a, x, incx);                                      \
    }
scal_launcher(float, float, rocblas_sscal)
scal_launcher(double, double, rocblas_dscal)
scal_launcher(std::complex<float>, std::complex<float>, rocblas_cscal)
scal_launcher(std::complex<double>, std::complex<double>, rocblas_zscal)
scal_launcher(float, std::complex<float>, rocblas_csscal)
scal_launcher(double, std::complex<double>, rocblas_zdscal)
#undef scal_launcher

template <typename func, typename t>
inline void axpy(func func, cl::sycl::queue &queue, int64_t n, t alpha, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<cudatatype *>(x_acc);
            auto y_ = sc.get_mem<cudatatype *>(y_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, (cudatatype *)&alpha, x_, incx, y_, incy);
        });
    });
}

#define axpy_launcher(type, rocblas_routine)                                                \
    void axpy(cl::sycl::queue &queue, int64_t n, type alpha, cl::sycl::buffer<type, 1> &x, \
              int64_t incx, cl::sycl::buffer<type, 1> &y, int64_t incy) {                  \
        axpy(rocblas_routine, queue, n, alpha, x, incx, y, incy);                           \
    }

axpy_launcher(float, rocblas_saxpy)
axpy_launcher(double, rocblas_daxpy)
axpy_launcher(std::complex<float>, rocblas_caxpy)
axpy_launcher(std::complex<double>, rocblas_zaxpy)
#undef axpy_launcher

template <typename func, typename t1, typename t2>
inline void rotg(func func, cl::sycl::queue &queue, cl::sycl::buffer<t1, 1> &a,
                 cl::sycl::buffer<t1, 1> &b, cl::sycl::buffer<t2, 1> &c,
                 cl::sycl::buffer<t1, 1> &s) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc = a.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto a_ = sc.get_mem<cudatatype1 *>(a_acc);
            auto b_ = sc.get_mem<cudatatype1 *>(b_acc);
            auto c_ = sc.get_mem<cudatatype2 *>(c_acc);
            auto s_ = sc.get_mem<cudatatype1 *>(s_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, a_, b_, c_, s_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define rotg_launcher(type1, type2, rocblas_routine)                         \
    void rotg(cl::sycl::queue &queue, cl::sycl::buffer<type1, 1> &a,        \
              cl::sycl::buffer<type1, 1> &b, cl::sycl::buffer<type2, 1> &c, \
              cl::sycl::buffer<type1, 1> &s) {                              \
        rotg(rocblas_routine, queue, a, b, c, s);                            \
    }

rotg_launcher(float, float, rocblas_srotg)
rotg_launcher(double, double, rocblas_drotg)
rotg_launcher(std::complex<float>, float, rocblas_crotg)
rotg_launcher(std::complex<double>, double, rocblas_zrotg)
#undef rotg_launcher

template <typename func, typename t>
inline void rotm(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy,
                 cl::sycl::buffer<t, 1> &param) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto param_acc = param.template get_access<cl::sycl::access::mode::read>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<cudatatype *>(x_acc);
            auto y_ = sc.get_mem<cudatatype *>(y_acc);
            auto param_ = sc.get_mem<cudatatype *>(param_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy, param_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define rotm_launcher(type, rocblas_routine)                                                   \
    void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, int64_t incx,  \
              cl::sycl::buffer<type, 1> &y, int64_t incy, cl::sycl::buffer<type, 1> &param) { \
        rotm(rocblas_routine, queue, n, x, incx, y, incy, param);                              \
    }

rotm_launcher(float, rocblas_srotm)
rotm_launcher(double, rocblas_drotm)
#undef rotm_launcher

template <typename func, typename t>
inline void copy(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<cudatatype *>(x_acc);
            auto y_ = sc.get_mem<cudatatype *>(y_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });
}

#define copy_launcher(type, rocblas_routine)                                                  \
    void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, int64_t incx, \
              cl::sycl::buffer<type, 1> &y, int64_t incy) {                                  \
        copy(rocblas_routine, queue, n, x, incx, y, incy);                                    \
    }

copy_launcher(float, rocblas_scopy)
copy_launcher(double, rocblas_dcopy)
copy_launcher(std::complex<float>, rocblas_ccopy)
copy_launcher(std::complex<double>, rocblas_zcopy)
#undef copy_launcher

template <typename func, typename t>
inline void dot(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                const int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy,
                cl::sycl::buffer<t, 1> &result) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<cudatatype *>(x_acc);
            auto y_ = sc.get_mem<cudatatype *>(y_acc);
            auto res_ = sc.get_mem<cudatatype *>(res_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy, res_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define dot_launcher(ext, type, rocblas_routine)                                         \
    void dot##ext(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x,      \
                  const int64_t incx, cl::sycl::buffer<type, 1> &y, const int64_t incy, \
                  cl::sycl::buffer<type, 1> &result) {                                  \
        dot(rocblas_routine, queue, n, x, incx, y, incy, result);                        \
    }
dot_launcher(, float, rocblas_sdot)
dot_launcher(, double, rocblas_ddot)
dot_launcher(c, std::complex<float>, rocblas_cdotc)
dot_launcher(c, std::complex<double>, rocblas_zdotc)
dot_launcher(u, std::complex<float>, rocblas_cdotu)
dot_launcher(u, std::complex<double>, rocblas_zdotu)
#undef dot_launcher

template <typename func, typename t1, typename t2, typename t3>
inline void rot(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t1, 1> &x,
                const int64_t incx, cl::sycl::buffer<t1, 1> &y, int64_t incy, t2 c, t3 s) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    using cudatatype3 = typename cudaequivalenttype<t3>::type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);
            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            // rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<cudatatype1 *>(x_acc);
            auto y_ = sc.get_mem<cudatatype1 *>(y_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy, (cudatatype2 *)&c,
                              (cudatatype3 *)&s);
        });
    });
}

#define rot_launcher(type1, type2, type3, rocblas_routine)                                          \
    void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type1, 1> &x, const int64_t incx, \
             cl::sycl::buffer<type1, 1> &y, int64_t incy, type2 c, type3 s) {                      \
        rot(rocblas_routine, queue, n, x, incx, y, incy, c, s);                                     \
    }

rot_launcher(float, float, float, rocblas_srot)
rot_launcher(double, double, double, rocblas_drot)
rot_launcher(std::complex<float>, float, float, rocblas_csrot)
rot_launcher(std::complex<double>, double, double, rocblas_zdrot)
#undef rot_launcher

void sdsdot(cl::sycl::queue &queue, int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    overflow_check(n, incx, incy);
    // cublas does not support sdot so we need to mimic sdot.
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto y_acc = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto res_acc = result.get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<float *>(x_acc);
            auto y_ = sc.get_mem<float *>(y_acc);
            auto res_ = sc.get_mem<float *>(res_acc);
            rocblas__status err;
            rocblas_error_func(rocblas_sdot, err, handle, n, x_, incx, y_, incy, res_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
    // since sb is a host pointer we need to bring the result back to the host and
    // add sb to it.
    result.get_access<cl::sycl::access::mode::read_write>()[0] += sb;
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    throw unimplemented("blas", "dot", "for column_major layout");
}

template <typename func, typename t>
inline void rotmg(func func, cl::sycl::queue &queue, cl::sycl::buffer<t, 1> &d1,
                  cl::sycl::buffer<t, 1> &d2, cl::sycl::buffer<t, 1> &x1, t y1,
                  cl::sycl::buffer<t, 1> &param) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    cl::sycl::buffer<t, 1> y1_buff(&y1, cl::sycl::range<1>(1));
    queue.submit([&](cl::sycl::handler &cgh) {
        auto d1_acc = d1.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto d2_acc = d2.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto x1_acc = x1.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto y1_acc = y1_buff.template get_access<cl::sycl::access::mode::read>(cgh);
        auto param_acc = param.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto d1_ = sc.get_mem<cudatatype *>(d1_acc);
            auto d2_ = sc.get_mem<cudatatype *>(d2_acc);
            auto x1_ = sc.get_mem<cudatatype *>(x1_acc);
            auto y1_ = sc.get_mem<cudatatype *>(y1_acc);
            auto param_ = sc.get_mem<cudatatype *>(param_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, d1_, d2_, x1_, y1_, param_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define rotmg_launcher(type, rocblas_routine)                                          \
    void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<type, 1> &d1,                 \
               cl::sycl::buffer<type, 1> &d2, cl::sycl::buffer<type, 1> &x1, type y1, \
               cl::sycl::buffer<type, 1> &param) {                                    \
        rotmg(rocblas_routine, queue, d1, d2, x1, y1, param);                          \
    }

rotmg_launcher(float, rocblas_srotmg)
rotmg_launcher(double, rocblas_drotmg)
#undef rotmg_launcher

template <typename func, typename t>
inline void iamax(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                  const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx);
    // cublas does not support int64_t as return type for the data. so we need to
    // mimic iamax. we are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // this change may cause failure as the result of integer overflow
    // based on the size. alternatively either we need to write a sycl kernel
    // to elementwise copy the data between two buffer, or allow reinterpret cast
    // to convert to different type with different typesize size.
    cl::sycl::buffer<int, 1> int_res_buff{ cl::sycl::range<1>(1) };
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto int_res_acc = int_res_buff.template get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<cudatatype *>(x_acc);
            auto int_res_ = sc.get_mem<int *>(int_res_acc);
            rocblas__status err;
            // for negative incx, iamax returns 0. this behaviour is similar to that of
            // reference netlib blas.
            rocblas_error_func(func, err, handle, n, x_, incx, int_res_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
    // this requires to bring the data to host, copy it, and return it back to
    // the device
    result.template get_access<cl::sycl::access::mode::write>()[0] =
        std::max((int64_t)int_res_buff.template get_access<cl::sycl::access::mode::read>()[0] - 1,
                 int64_t{ 0 });
}

#define iamax_launcher(type, rocblas_routine)                                    \
    void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, \
               const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {      \
        iamax(rocblas_routine, queue, n, x, incx, result);                       \
    }
iamax_launcher(float, rocblas_isamax)
iamax_launcher(double, rocblas_idamax)
iamax_launcher(std::complex<float>, rocblas_icamax)
iamax_launcher(std::complex<double>, rocblas_izamax)
#undef iamax_launcher

template <typename func, typename t>
inline void swap(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<cl::sycl::access::mode::read_write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<cudatatype *>(x_acc);
            auto y_ = sc.get_mem<cudatatype *>(y_acc);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });
}

#define swap_launcher(type, rocblas_routine)                                                  \
    void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, int64_t incx, \
              cl::sycl::buffer<type, 1> &y, int64_t incy) {                                  \
        swap(rocblas_routine, queue, n, x, incx, y, incy);                                    \
    }

swap_launcher(float, rocblas_sswap)
swap_launcher(double, rocblas_dswap)
swap_launcher(std::complex<float>, rocblas_cswap)
swap_launcher(std::complex<double>, rocblas_zswap)
#undef swap_launcher

template <typename func, typename t>
inline void iamin(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                  const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx);
    // cublas does not support int64_t as return type for the data. so we need to
    // mimic iamin we are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // this change may cause failure as the result of integer overflow
    // based on the size. alternatively, either we need to write a sycl kernel
    // to elementwise copy the data between two buffer, or allow reinterpret cast
    // to convert to different type with different typesize size.
    cl::sycl::buffer<int, 1> int_res_buff{ cl::sycl::range<1>(1) };
    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto int_res_acc = int_res_buff.template get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<cudatatype *>(x_acc);
            auto int_res_ = sc.get_mem<int *>(int_res_acc);
            rocblas__status err;
            // for negative incx, iamin returns 0. this behaviour is similar to that of
            // implemented as a reference iamin.
            rocblas_error_func(func, err, handle, n, x_, incx, int_res_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
    result.template get_access<cl::sycl::access::mode::write>()[0] =
        std::max((int64_t)int_res_buff.template get_access<cl::sycl::access::mode::read>()[0] - 1,
                 int64_t{ 0 });
}

#define iamin_launcher(type, rocblas_routine)                                    \
    void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, \
               const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {      \
        iamin(rocblas_routine, queue, n, x, incx, result);                       \
    }
iamin_launcher(float, rocblas_isamin)
iamin_launcher(double, rocblas_idamin)
iamin_launcher(std::complex<float>, rocblas_icamin)
iamin_launcher(std::complex<double>, rocblas_izamin)
#undef iamin_launcher

template <typename func, typename t1, typename t2>
inline void nrm2(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t1, 1> &x,
                 const int64_t incx, cl::sycl::buffer<t2, 1> &result) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    overflow_check(n, incx);

    queue.submit([&](cl::sycl::handler &cgh) {
        auto x_acc = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<cl::sycl::access::mode::write>(cgh);
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            // by default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. when it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_setpointermode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<cudatatype1 *>(x_acc);
            auto res_ = sc.get_mem<cudatatype2 *>(res_acc);
            rocblas__status err;
            // nrm2 does not support negative index
            rocblas_error_func(func, err, handle, n, x_, std::abs(incx), res_);
            // higher level blas functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid cuda_error_illegal_adress errors
            rocblas_setpointermode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define nrm2_launcher(type1, type2, rocblas_routine)                             \
    void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type1, 1> &x, \
              const int64_t incx, cl::sycl::buffer<type2, 1> &result) {         \
        nrm2(rocblas_routine, queue, n, x, incx, result);                        \
    }
nrm2_launcher(float, float, rocblas_snrm2)
nrm2_launcher(double, double, rocblas_dnrm2)
nrm2_launcher(std::complex<float>, float, rocblas_scnrm2)
nrm2_launcher(std::complex<double>, double, rocblas_dznrm2)
#undef nrm2_launcher

// usm apis

// level 1
template <typename func, typename t1, typename t2>
inline cl::sycl::event asum(func func, cl::sycl::queue &queue, int64_t n, const t1 *x,
                            const int64_t incx, t2 *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    overflow_check(n, incx);

    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const cudatatype1 *>(x);
            auto res_ = reinterpret_cast<cudatatype2 *>(result);
            rocblas__status err;
            // asum does not support negative index
            rocblas_error_func(func, err, handle, n, x_, std::abs(incx), res_);
        });
    });
    return done;
}

#define asum_launcher_usm(type1, type2, rocblas_routine)                                         \
    cl::sycl::event asum(cl::sycl::queue &queue, int64_t n, const type1 *x, const int64_t incx, \
                         type2 *result,                                                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {         \
        return asum(rocblas_routine, queue, n, x, incx, result, dependencies);                   \
    }
asum_launcher_usm(float, float, rocblas_sasum)
asum_launcher_usm(double, double, rocblas_dasum)
asum_launcher_usm(std::complex<float>, float, rocblas_scasum)
asum_launcher_usm(std::complex<double>, double, rocblas_dzasum)
#undef asum_launcher_usm

template <typename func, typename t1, typename t2>
inline cl::sycl::event scal(func func, cl::sycl::queue &queue, int64_t n, t1 a, t2 *x, int64_t incx,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    overflow_check(n, incx);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<cudatatype2 *>(x);
            rocblas__status err;
            // scal does not support negative incx
            rocblas_error_func(func, err, handle, n, (cudatatype1 *)&a, x_, std::abs(incx));
        });
    });
    return done;
}

#define scal_launcher_usm(type1, type2, rocblas_routine)                                      \
    cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, type1 a, type2 *x, int64_t incx, \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {      \
        return scal(rocblas_routine, queue, n, a, x, incx, dependencies);                     \
    }
scal_launcher_usm(float, float, rocblas_sscal)
scal_launcher_usm(double, double, rocblas_dscal)
scal_launcher_usm(std::complex<float>, std::complex<float>, rocblas_cscal)
scal_launcher_usm(std::complex<double>, std::complex<double>, rocblas_zscal)
scal_launcher_usm(float, std::complex<float>, rocblas_csscal)
scal_launcher_usm(double, std::complex<double>, rocblas_zdscal)
#undef scal_launcher_usm

template <typename func, typename t>
inline cl::sycl::event axpy(func func, cl::sycl::queue &queue, int64_t n, t alpha, const t *x,
                            int64_t incx, t *y, int64_t incy,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const cudatatype *>(x);
            auto y_ = reinterpret_cast<cudatatype *>(y);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, (cudatatype *)&alpha, x_, incx, y_, incy);
        });
    });
    return done;
}

#define axpy_launcher_usm(type, rocblas_routine)                                         \
    cl::sycl::event axpy(cl::sycl::queue &queue, int64_t n, type alpha, const type *x,  \
                         int64_t incx, type *y, int64_t incy,                           \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) { \
        return axpy(rocblas_routine, queue, n, alpha, x, incx, y, incy, dependencies);   \
    }

axpy_launcher_usm(float, rocblas_saxpy)
axpy_launcher_usm(double, rocblas_daxpy)
axpy_launcher_usm(std::complex<float>, rocblas_caxpy)
axpy_launcher_usm(std::complex<double>, rocblas_zaxpy)
#undef axpy_launcher_usm

template <typename func, typename t1, typename t2>
inline cl::sycl::event rotg(func func, cl::sycl::queue &queue, t1 *a, t1 *b, t2 *c, t1 *s,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<cudatatype1 *>(a);
            auto b_ = reinterpret_cast<cudatatype1 *>(b);
            auto c_ = reinterpret_cast<cudatatype2 *>(c);
            auto s_ = reinterpret_cast<cudatatype1 *>(s);
            rocblas__status err;
            rocblas_error_func(func, err, handle, a_, b_, c_, s_);
        });
    });
    return done;
}

#define rotg_launcher_usm(type1, type2, rocblas_routine)                                  \
    cl::sycl::event rotg(cl::sycl::queue &queue, type1 *a, type1 *b, type2 *c, type1 *s, \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {  \
        return rotg(rocblas_routine, queue, a, b, c, s, dependencies);                    \
    }

rotg_launcher_usm(float, float, rocblas_srotg)
rotg_launcher_usm(double, double, rocblas_drotg)
rotg_launcher_usm(std::complex<float>, float, rocblas_crotg)
rotg_launcher_usm(std::complex<double>, double, rocblas_zrotg)
#undef rotg_launcher_usm

template <typename func, typename t>
inline cl::sycl::event rotm(func func, cl::sycl::queue &queue, int64_t n, t *x, int64_t incx, t *y,
                            int64_t incy, t *param,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<cudatatype *>(x);
            auto y_ = reinterpret_cast<cudatatype *>(y);
            auto param_ = reinterpret_cast<cudatatype *>(param);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy, param_);
        });
    });
    return done;
}

#define rotm_launcher_usm(type, rocblas_routine)                                             \
    cl::sycl::event rotm(cl::sycl::queue &queue, int64_t n, type *x, int64_t incx, type *y, \
                         int64_t incy, type *param,                                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {     \
        return rotm(rocblas_routine, queue, n, x, incx, y, incy, param, dependencies);       \
    }

rotm_launcher_usm(float, rocblas_srotm)
rotm_launcher_usm(double, rocblas_drotm)
#undef rotm_launcher_usm

template <typename func, typename t>
inline cl::sycl::event copy(func func, cl::sycl::queue &queue, int64_t n, const t *x, int64_t incx,
                            t *y, int64_t incy,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const cudatatype *>(x);
            auto y_ = reinterpret_cast<cudatatype *>(y);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });
    return done;
}

#define copy_launcher_usm(type, rocblas_routine)                                                   \
    cl::sycl::event copy(cl::sycl::queue &queue, int64_t n, const type *x, int64_t incx, type *y, \
                         int64_t incy,                                                            \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return copy(rocblas_routine, queue, n, x, incx, y, incy, dependencies);                    \
    }

copy_launcher_usm(float, rocblas_scopy)
copy_launcher_usm(double, rocblas_dcopy)
copy_launcher_usm(std::complex<float>, rocblas_ccopy)
copy_launcher_usm(std::complex<double>, rocblas_zcopy)
#undef copy_launcher_usm

template <typename func, typename t>
inline cl::sycl::event dot(func func, cl::sycl::queue &queue, int64_t n, const t *x,
                           const int64_t incx, const t *y, int64_t incy, t *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const cudatatype *>(x);
            auto y_ = reinterpret_cast<const cudatatype *>(y);
            auto res_ = reinterpret_cast<cudatatype *>(result);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy, res_);
        });
    });
    return done;
}

#define dot_launcher_usm(ext, type, rocblas_routine)                                                \
    cl::sycl::event dot##ext(cl::sycl::queue &queue, int64_t n, const type *x, const int64_t incx, \
                             const type *y, const int64_t incy, type *result,                      \
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {        \
        return dot(rocblas_routine, queue, n, x, incx, y, incy, result, dependencies);              \
    }
dot_launcher_usm(, float, rocblas_sdot)
dot_launcher_usm(, double, rocblas_ddot)
dot_launcher_usm(c, std::complex<float>, rocblas_cdotc)
dot_launcher_usm(c, std::complex<double>, rocblas_zdotc)
dot_launcher_usm(u, std::complex<float>, rocblas_cdotu)
dot_launcher_usm(u, std::complex<double>, rocblas_zdotu)
#undef dot_launcher_usm

template <typename func, typename t1, typename t2, typename t3>
inline cl::sycl::event rot(func func, cl::sycl::queue &queue, int64_t n, t1 *x, const int64_t incx,
                           t1 *y, int64_t incy, t2 c, t3 s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    using cudatatype3 = typename cudaequivalenttype<t3>::type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<cudatatype1 *>(x);
            auto y_ = reinterpret_cast<cudatatype1 *>(y);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy, (cudatatype2 *)&c,
                              (cudatatype3 *)&s);
        });
    });
    return done;
}

#define rot_launcher_usm(type1, type2, type3, rocblas_routine)                                      \
    cl::sycl::event rot(cl::sycl::queue &queue, int64_t n, type1 *x, const int64_t incx, type1 *y, \
                        int64_t incy, type2 c, type3 s,                                            \
                        const cl::sycl::vector_class<cl::sycl::event> &dependencies) {             \
        return rot(rocblas_routine, queue, n, x, incx, y, incy, c, s, dependencies);                \
    }

rot_launcher_usm(float, float, float, rocblas_srot)
rot_launcher_usm(double, double, double, rocblas_drot)
rot_launcher_usm(std::complex<float>, float, float, rocblas_csrot)
rot_launcher_usm(std::complex<double>, double, double, rocblas_zdrot)
#undef rot_launcher_usm

cl::sycl::event sdsdot(cl::sycl::queue &queue, int64_t n, float sb, const float *x, int64_t incx,
                       const float *y, int64_t incy, float *result,
                       const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    overflow_check(n, incx, incy);
    // cublas does not support sdot so we need to mimic sdot.
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const float *>(x);
            auto y_ = reinterpret_cast<const float *>(y);
            auto res_ = reinterpret_cast<float *>(result);
            rocblas__status err;
            rocblas_error_func(rocblas_sdot, err, handle, n, x_, incx, y_, incy, res_);
        });
    });
    done.wait();
    result[0] = result[0] + sb;
    return done;
}

cl::sycl::event dot(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx, const float *y,
                    int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "dot", "for column_major layout");
}

template <typename func, typename t>
inline cl::sycl::event rotmg(func func, cl::sycl::queue &queue, t *d1, t *d2, t *x1, t y1, t *param,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto d1_ = reinterpret_cast<cudatatype *>(d1);
            auto d2_ = reinterpret_cast<cudatatype *>(d2);
            auto x1_ = reinterpret_cast<cudatatype *>(x1);
            auto y1_ = reinterpret_cast<const cudatatype *>(&y1);
            auto param_ = reinterpret_cast<cudatatype *>(param);
            rocblas__status err;
            rocblas_error_func(func, err, handle, d1_, d2_, x1_, y1_, param_);
        });
    });
    return done;
}

#define rotmg_launcher_usm(type, rocblas_routine)                                         \
    cl::sycl::event rotmg(cl::sycl::queue &queue, type *d1, type *d2, type *x1, type y1, \
                          type *param,                                                   \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) { \
        return rotmg(rocblas_routine, queue, d1, d2, x1, y1, param, dependencies);        \
    }

rotmg_launcher_usm(float, rocblas_srotmg)
rotmg_launcher_usm(double, rocblas_drotmg)
#undef rotmg_launcher_usm

template <typename func, typename t>
inline cl::sycl::event iamax(func func, cl::sycl::queue &queue, int64_t n, const t *x,
                             const int64_t incx, int64_t *result,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx);
    // cublas does not support int64_t as return type for the data. so we need to
    // mimic iamax. we are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // this change may cause failure as the result of integer overflow
    // based on the size.
    int int_res = 0;
    int *int_res_p = &int_res;
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const cudatatype *>(x);
            auto int_res_p_ = reinterpret_cast<int *>(int_res_p);
            rocblas__status err;
            // for negative incx, iamax returns 0. this behaviour is similar to that of
            // reference iamax.
            rocblas_error_func(func, err, handle, n, x_, incx, int_res_p_);
        });
    });
    done.wait();
    result[0] = std::max((int64_t)(*int_res_p - 1), int64_t{ 0 });
    return done;
}

#define iamax_launcher_usm(type, rocblas_routine)                                                \
    cl::sycl::event iamax(cl::sycl::queue &queue, int64_t n, const type *x, const int64_t incx, \
                          int64_t *result,                                                      \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {        \
        return iamax(rocblas_routine, queue, n, x, incx, result, dependencies);                  \
    }
iamax_launcher_usm(float, rocblas_isamax)
iamax_launcher_usm(double, rocblas_idamax)
iamax_launcher_usm(std::complex<float>, rocblas_icamax)
iamax_launcher_usm(std::complex<double>, rocblas_izamax)
#undef iamax_launcher_usm

template <typename func, typename t>
inline cl::sycl::event swap(func func, cl::sycl::queue &queue, int64_t n, t *x, int64_t incx, t *y,
                            int64_t incy,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx, incy);
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<cudatatype *>(x);
            auto y_ = reinterpret_cast<cudatatype *>(y);
            rocblas__status err;
            rocblas_error_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });
    return done;
}

#define swap_launcher_usm(type, rocblas_routine)                                             \
    cl::sycl::event swap(cl::sycl::queue &queue, int64_t n, type *x, int64_t incx, type *y, \
                         int64_t incy,                                                      \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {     \
        return swap(rocblas_routine, queue, n, x, incx, y, incy, dependencies);              \
    }

swap_launcher_usm(float, rocblas_sswap)
swap_launcher_usm(double, rocblas_dswap)
swap_launcher_usm(std::complex<float>, rocblas_cswap)
swap_launcher_usm(std::complex<double>, rocblas_zswap)
#undef swap_launcher_usm

template <typename func, typename t>
inline cl::sycl::event iamin(func func, cl::sycl::queue &queue, int64_t n, const t *x,
                             const int64_t incx, int64_t *result,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype = typename cudaequivalenttype<t>::type;
    overflow_check(n, incx);
    // cublas does not support int64_t as return type for the data. so we need to
    // mimic iamin. we are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // this change may cause failure as the result of integer overflow
    // based on the size.
    int int_res = 0;
    int *int_res_p = &int_res;
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const cudatatype *>(x);
            auto int_res_p_ = reinterpret_cast<int *>(int_res_p);
            rocblas__status err;
            // for negative incx, iamin returns 0. this behaviour is similar to that of
            // implemented iamin.
            rocblas_error_func(func, err, handle, n, x_, incx, int_res_p_);
        });
    });
    done.wait();
    result[0] = std::max((int64_t)(*int_res_p - 1), int64_t{ 0 });
    return done;
}

#define iamin_launcher_usm(type, rocblas_routine)                                                \
    cl::sycl::event iamin(cl::sycl::queue &queue, int64_t n, const type *x, const int64_t incx, \
                          int64_t *result,                                                      \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {        \
        return iamin(rocblas_routine, queue, n, x, incx, result, dependencies);                  \
    }
iamin_launcher_usm(float, rocblas_isamin)
iamin_launcher_usm(double, rocblas_idamin)
iamin_launcher_usm(std::complex<float>, rocblas_icamin)
iamin_launcher_usm(std::complex<double>, rocblas_izamin)
#undef iamin_launcher_usm

template <typename func, typename t1, typename t2>
inline cl::sycl::event nrm2(func func, cl::sycl::queue &queue, int64_t n, const t1 *x,
                            const int64_t incx, t2 *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    using cudatatype1 = typename cudaequivalenttype<t1>::type;
    using cudatatype2 = typename cudaequivalenttype<t2>::type;
    overflow_check(n, incx);

    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_rocblas__host_task(cgh, queue,[=](cublasscopedcontexthandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const cudatatype1 *>(x);
            auto res_ = reinterpret_cast<cudatatype2 *>(result);
            rocblas__status err;
            // nrm2 does not support negative index
            rocblas_error_func(func, err, handle, n, x_, std::abs(incx), res_);
        });
    });
    return done;
}

#define nrm2_launcher_usm(type1, type2, rocblas_routine)                                         \
    cl::sycl::event nrm2(cl::sycl::queue &queue, int64_t n, const type1 *x, const int64_t incx, \
                         type2 *result,                                                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {         \
        return nrm2(rocblas_routine, queue, n, x, incx, result, dependencies);                   \
    }
nrm2_launcher_usm(float, float, rocblas_snrm2)
nrm2_launcher_usm(double, double, rocblas_dnrm2)
nrm2_launcher_usm(std::complex<float>, float, rocblas_scnrm2)
nrm2_launcher_usm(std::complex<double>, double, rocblas_dznrm2)
#undef nrm2_launcher_usm

} // namespace column_major
namespace row_major {

// buffer apis

// level 1
template <typename func, typename t1, typename t2>
inline void asum(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t1, 1> &x,
                 const int64_t incx, cl::sycl::buffer<t2, 1> &result) {
    throw unimplemented("blas", "asum", "for row_major layout");
}

#define asum_launcher(type1, type2, rocblas_routine)                             \
    void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type1, 1> &x, \
              const int64_t incx, cl::sycl::buffer<type2, 1> &result) {         \
        asum(rocblas_routine, queue, n, x, incx, result);                        \
    }
asum_launcher(float, float, rocblas_sasum)
asum_launcher(double, double, rocblas_dasum)
asum_launcher(std::complex<float>, float, rocblas_scasum)
asum_launcher(std::complex<double>, double, rocblas_dzasum)
#undef asum_launcher

template <typename func, typename t1, typename t2>
inline void scal(func func, cl::sycl::queue &queue, int64_t n, t1 a, cl::sycl::buffer<t2, 1> &x,
                 int64_t incx) {
    throw unimplemented("blas", "scal", "for row_major layout");
}

#define scal_launcher(type1, type2, rocblas_routine)                                      \
    void scal(cl::sycl::queue &queue, int64_t n, type1 a, cl::sycl::buffer<type2, 1> &x, \
              int64_t incx) {                                                            \
        scal(rocblas_routine, queue, n, a, x, incx);                                      \
    }
scal_launcher(float, float, rocblas_sscal)
scal_launcher(double, double, rocblas_dscal)
scal_launcher(std::complex<float>, std::complex<float>, rocblas_cscal)
scal_launcher(std::complex<double>, std::complex<double>, rocblas_zscal)
scal_launcher(float, std::complex<float>, rocblas_csscal)
scal_launcher(double, std::complex<double>, rocblas_zdscal)
#undef scal_launcher

template <typename func, typename t>
inline void axpy(func func, cl::sycl::queue &queue, int64_t n, t alpha, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy) {
    throw unimplemented("blas", "axpy", "for row_major layout");
}

#define axpy_launcher(type, rocblas_routine)                                                \
    void axpy(cl::sycl::queue &queue, int64_t n, type alpha, cl::sycl::buffer<type, 1> &x, \
              int64_t incx, cl::sycl::buffer<type, 1> &y, int64_t incy) {                  \
        axpy(rocblas_routine, queue, n, alpha, x, incx, y, incy);                           \
    }

axpy_launcher(float, rocblas_saxpy)
axpy_launcher(double, rocblas_daxpy)
axpy_launcher(std::complex<float>, rocblas_caxpy)
axpy_launcher(std::complex<double>, rocblas_zaxpy)
#undef axpy_launcher

template <typename func, typename t1, typename t2>
inline void rotg(func func, cl::sycl::queue &queue, cl::sycl::buffer<t1, 1> &a,
                 cl::sycl::buffer<t1, 1> &b, cl::sycl::buffer<t2, 1> &c,
                 cl::sycl::buffer<t1, 1> &s) {
    throw unimplemented("blas", "rotg", "for row_major layout");
}

#define rotg_launcher(type1, type2, rocblas_routine)                         \
    void rotg(cl::sycl::queue &queue, cl::sycl::buffer<type1, 1> &a,        \
              cl::sycl::buffer<type1, 1> &b, cl::sycl::buffer<type2, 1> &c, \
              cl::sycl::buffer<type1, 1> &s) {                              \
        rotg(rocblas_routine, queue, a, b, c, s);                            \
    }

rotg_launcher(float, float, rocblas_srotg)
rotg_launcher(double, double, rocblas_drotg)
rotg_launcher(std::complex<float>, float, rocblas_crotg)
rotg_launcher(std::complex<double>, double, rocblas_zrotg)
#undef rotg_launcher

template <typename func, typename t>
inline void rotm(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy,
                 cl::sycl::buffer<t, 1> &param) {
    throw unimplemented("blas", "rotm", "for row_major layout");
}

#define rotm_launcher(type, rocblas_routine)                                                   \
    void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, int64_t incx,  \
              cl::sycl::buffer<type, 1> &y, int64_t incy, cl::sycl::buffer<type, 1> &param) { \
        rotm(rocblas_routine, queue, n, x, incx, y, incy, param);                              \
    }

rotm_launcher(float, rocblas_srotm)
rotm_launcher(double, rocblas_drotm)
#undef rotm_launcher

template <typename func, typename t>
inline void copy(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy) {
    throw unimplemented("blas", "copy", "for row_major layout");
}

#define copy_launcher(type, rocblas_routine)                                                  \
    void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, int64_t incx, \
              cl::sycl::buffer<type, 1> &y, int64_t incy) {                                  \
        copy(rocblas_routine, queue, n, x, incx, y, incy);                                    \
    }

copy_launcher(float, rocblas_scopy)
copy_launcher(double, rocblas_dcopy)
copy_launcher(std::complex<float>, rocblas_ccopy)
copy_launcher(std::complex<double>, rocblas_zcopy)
#undef copy_launcher

template <typename func, typename t>
inline void dot(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                const int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy,
                cl::sycl::buffer<t, 1> &result) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

#define dot_launcher(ext, type, rocblas_routine)                                         \
    void dot##ext(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x,      \
                  const int64_t incx, cl::sycl::buffer<type, 1> &y, const int64_t incy, \
                  cl::sycl::buffer<type, 1> &result) {                                  \
        dot(rocblas_routine, queue, n, x, incx, y, incy, result);                        \
    }
dot_launcher(, float, rocblas_sdot)
dot_launcher(, double, rocblas_ddot)
dot_launcher(c, std::complex<float>, rocblas_cdotc)
dot_launcher(c, std::complex<double>, rocblas_zdotc)
dot_launcher(u, std::complex<float>, rocblas_cdotu)
dot_launcher(u, std::complex<double>, rocblas_zdotu)
#undef dot_launcher

template <typename func, typename t1, typename t2, typename t3>
inline void rot(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t1, 1> &x,
                const int64_t incx, cl::sycl::buffer<t1, 1> &y, int64_t incy, t2 c, t3 s) {
    throw unimplemented("blas", "rot", "for row_major layout");
}

#define rot_launcher(type1, type2, type3, rocblas_routine)                                          \
    void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type1, 1> &x, const int64_t incx, \
             cl::sycl::buffer<type1, 1> &y, int64_t incy, type2 c, type3 s) {                      \
        rot(rocblas_routine, queue, n, x, incx, y, incy, c, s);                                     \
    }

rot_launcher(float, float, float, rocblas_srot)
rot_launcher(double, double, double, rocblas_drot)
rot_launcher(std::complex<float>, float, float, rocblas_csrot)
rot_launcher(std::complex<double>, double, double, rocblas_zdrot)
#undef rot_launcher

void sdsdot(cl::sycl::queue &queue, int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    throw unimplemented("blas", "sdsdot", "for row_major layout");
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

template <typename func, typename t>
inline void rotmg(func func, cl::sycl::queue &queue, cl::sycl::buffer<t, 1> &d1,
                  cl::sycl::buffer<t, 1> &d2, cl::sycl::buffer<t, 1> &x1, t y1,
                  cl::sycl::buffer<t, 1> &param) {
    throw unimplemented("blas", "rotmg", "for row_major layout");
}

#define rotmg_launcher(type, rocblas_routine)                                          \
    void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<type, 1> &d1,                 \
               cl::sycl::buffer<type, 1> &d2, cl::sycl::buffer<type, 1> &x1, type y1, \
               cl::sycl::buffer<type, 1> &param) {                                    \
        rotmg(rocblas_routine, queue, d1, d2, x1, y1, param);                          \
    }

rotmg_launcher(float, rocblas_srotmg)
rotmg_launcher(double, rocblas_drotmg)
#undef rotmg_launcher

template <typename func, typename t>
inline void iamax(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                  const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    throw unimplemented("blas", "iamax", "for row_major layout");
}

#define iamax_launcher(type, rocblas_routine)                                    \
    void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, \
               const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {      \
        iamax(rocblas_routine, queue, n, x, incx, result);                       \
    }
iamax_launcher(float, rocblas_isamax)
iamax_launcher(double, rocblas_idamax)
iamax_launcher(std::complex<float>, rocblas_icamax)
iamax_launcher(std::complex<double>, rocblas_izamax)
#undef iamax_launcher

template <typename func, typename t>
inline void swap(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                 int64_t incx, cl::sycl::buffer<t, 1> &y, int64_t incy) {
    throw unimplemented("blas", "swap", "for row_major layout");
}

#define swap_launcher(type, rocblas_routine)                                                  \
    void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, int64_t incx, \
              cl::sycl::buffer<type, 1> &y, int64_t incy) {                                  \
        swap(rocblas_routine, queue, n, x, incx, y, incy);                                    \
    }

swap_launcher(float, rocblas_sswap)
swap_launcher(double, rocblas_dswap)
swap_launcher(std::complex<float>, rocblas_cswap)
swap_launcher(std::complex<double>, rocblas_zswap)
#undef swap_launcher

template <typename func, typename t>
inline void iamin(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t, 1> &x,
                  const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    throw unimplemented("blas", "iamin", "for row_major layout");
}

#define iamin_launcher(type, rocblas_routine)                                    \
    void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type, 1> &x, \
               const int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {      \
        iamin(rocblas_routine, queue, n, x, incx, result);                       \
    }
iamin_launcher(float, rocblas_isamin)
iamin_launcher(double, rocblas_idamin)
iamin_launcher(std::complex<float>, rocblas_icamin)
iamin_launcher(std::complex<double>, rocblas_izamin)
#undef iamin_launcher

template <typename func, typename t1, typename t2>
inline void nrm2(func func, cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<t1, 1> &x,
                 const int64_t incx, cl::sycl::buffer<t2, 1> &result) {
    throw unimplemented("blas", "nrm2", "for row_major layout");
}

#define nrm2_launcher(type1, type2, rocblas_routine)                             \
    void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<type1, 1> &x, \
              const int64_t incx, cl::sycl::buffer<type2, 1> &result) {         \
        nrm2(rocblas_routine, queue, n, x, incx, result);                        \
    }
nrm2_launcher(float, float, rocblas_snrm2)
nrm2_launcher(double, double, rocblas_dnrm2)
nrm2_launcher(std::complex<float>, float, rocblas_scnrm2)
nrm2_launcher(std::complex<double>, double, rocblas_dznrm2)
#undef nrm2_launcher

// usm apis

// level 1
template <typename func, typename t1, typename t2>
inline cl::sycl::event asum(func func, cl::sycl::queue &queue, int64_t n, const t1 *x,
                            const int64_t incx, t2 *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "asum", "for row_major layout");
}

#define asum_launcher_usm(type1, type2, rocblas_routine)                                         \
    cl::sycl::event asum(cl::sycl::queue &queue, int64_t n, const type1 *x, const int64_t incx, \
                         type2 *result,                                                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {         \
        return asum(rocblas_routine, queue, n, x, incx, result, dependencies);                   \
    }
asum_launcher_usm(float, float, rocblas_sasum)
asum_launcher_usm(double, double, rocblas_dasum)
asum_launcher_usm(std::complex<float>, float, rocblas_scasum)
asum_launcher_usm(std::complex<double>, double, rocblas_dzasum)
#undef asum_launcher_usm

template <typename func, typename t1, typename t2>
inline cl::sycl::event scal(func func, cl::sycl::queue &queue, int64_t n, t1 a, t2 *x, int64_t incx,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "scal", "for row_major layout");
}

#define scal_launcher_usm(type1, type2, rocblas_routine)                                      \
    cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, type1 a, type2 *x, int64_t incx, \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {      \
        return scal(rocblas_routine, queue, n, a, x, incx, dependencies);                     \
    }
scal_launcher_usm(float, float, rocblas_sscal)
scal_launcher_usm(double, double, rocblas_dscal)
scal_launcher_usm(std::complex<float>, std::complex<float>, rocblas_cscal)
scal_launcher_usm(std::complex<double>, std::complex<double>, rocblas_zscal)
scal_launcher_usm(float, std::complex<float>, rocblas_csscal)
scal_launcher_usm(double, std::complex<double>, rocblas_zdscal)
#undef scal_launcher_usm

template <typename func, typename t>
inline cl::sycl::event axpy(func func, cl::sycl::queue &queue, int64_t n, t alpha, const t *x,
                            int64_t incx, t *y, int64_t incy,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "axpy", "for row_major layout");
}

#define axpy_launcher_usm(type, rocblas_routine)                                         \
    cl::sycl::event axpy(cl::sycl::queue &queue, int64_t n, type alpha, const type *x,  \
                         int64_t incx, type *y, int64_t incy,                           \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) { \
        return axpy(rocblas_routine, queue, n, alpha, x, incx, y, incy, dependencies);   \
    }

axpy_launcher_usm(float, rocblas_saxpy)
axpy_launcher_usm(double, rocblas_daxpy)
axpy_launcher_usm(std::complex<float>, rocblas_caxpy)
axpy_launcher_usm(std::complex<double>, rocblas_zaxpy)
#undef axpy_launcher_usm

template <typename func, typename t1, typename t2>
inline cl::sycl::event rotg(func func, cl::sycl::queue &queue, t1 *a, t1 *b, t2 *c, t1 *s,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "rotg", "for row_major layout");
}

#define rotg_launcher_usm(type1, type2, rocblas_routine)                                  \
    cl::sycl::event rotg(cl::sycl::queue &queue, type1 *a, type1 *b, type2 *c, type1 *s, \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {  \
        return rotg(rocblas_routine, queue, a, b, c, s, dependencies);                    \
    }

rotg_launcher_usm(float, float, rocblas_srotg)
rotg_launcher_usm(double, double, rocblas_drotg)
rotg_launcher_usm(std::complex<float>, float, rocblas_crotg)
rotg_launcher_usm(std::complex<double>, double, rocblas_zrotg)
#undef rotg_launcher_usm

template <typename func, typename t>
inline cl::sycl::event rotm(func func, cl::sycl::queue &queue, int64_t n, t *x, int64_t incx, t *y,
                            int64_t incy, t *param,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "rotm", "for row_major layout");
}

#define rotm_launcher_usm(type, rocblas_routine)                                             \
    cl::sycl::event rotm(cl::sycl::queue &queue, int64_t n, type *x, int64_t incx, type *y, \
                         int64_t incy, type *param,                                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {     \
        return rotm(rocblas_routine, queue, n, x, incx, y, incy, param, dependencies);       \
    }

rotm_launcher_usm(float, rocblas_srotm)
rotm_launcher_usm(double, rocblas_drotm)
#undef rotm_launcher_usm

template <typename func, typename t>
inline cl::sycl::event copy(func func, cl::sycl::queue &queue, int64_t n, const t *x, int64_t incx,
                            t *y, int64_t incy,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "copy", "for row_major layout");
}

#define copy_launcher_usm(type, rocblas_routine)                                                   \
    cl::sycl::event copy(cl::sycl::queue &queue, int64_t n, const type *x, int64_t incx, type *y, \
                         int64_t incy,                                                            \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {           \
        return copy(rocblas_routine, queue, n, x, incx, y, incy, dependencies);                    \
    }

copy_launcher_usm(float, rocblas_scopy)
copy_launcher_usm(double, rocblas_dcopy)
copy_launcher_usm(std::complex<float>, rocblas_ccopy)
copy_launcher_usm(std::complex<double>, rocblas_zcopy)
#undef copy_launcher_usm

template <typename func, typename t>
inline cl::sycl::event dot(func func, cl::sycl::queue &queue, int64_t n, const t *x,
                           const int64_t incx, const t *y, int64_t incy, t *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

#define dot_launcher_usm(ext, type, rocblas_routine)                                                \
    cl::sycl::event dot##ext(cl::sycl::queue &queue, int64_t n, const type *x, const int64_t incx, \
                             const type *y, const int64_t incy, type *result,                      \
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {        \
        return dot(rocblas_routine, queue, n, x, incx, y, incy, result, dependencies);              \
    }
dot_launcher_usm(, float, rocblas_sdot)
dot_launcher_usm(, double, rocblas_ddot)
dot_launcher_usm(c, std::complex<float>, rocblas_cdotc)
dot_launcher_usm(c, std::complex<double>, rocblas_zdotc)
dot_launcher_usm(u, std::complex<float>, rocblas_cdotu)
dot_launcher_usm(u, std::complex<double>, rocblas_zdotu)
#undef dot_launcher_usm

template <typename func, typename t1, typename t2, typename t3>
inline cl::sycl::event rot(func func, cl::sycl::queue &queue, int64_t n, t1 *x, const int64_t incx,
                           t1 *y, int64_t incy, t2 c, t3 s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "rot", "for row_major layout");
}

#define rot_launcher_usm(type1, type2, type3, rocblas_routine)                                      \
    cl::sycl::event rot(cl::sycl::queue &queue, int64_t n, type1 *x, const int64_t incx, type1 *y, \
                        int64_t incy, type2 c, type3 s,                                            \
                        const cl::sycl::vector_class<cl::sycl::event> &dependencies) {             \
        return rot(rocblas_routine, queue, n, x, incx, y, incy, c, s, dependencies);                \
    }

rot_launcher_usm(float, float, float, rocblas_srot)
rot_launcher_usm(double, double, double, rocblas_drot)
rot_launcher_usm(std::complex<float>, float, float, rocblas_csrot)
rot_launcher_usm(std::complex<double>, double, double, rocblas_zdrot)
#undef rot_launcher_usm

cl::sycl::event sdsdot(cl::sycl::queue &queue, int64_t n, float sb, const float *x, int64_t incx,
                       const float *y, int64_t incy, float *result,
                       const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "sdsdot", "for row_major layout");
}

cl::sycl::event dot(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx, const float *y,
                    int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

template <typename func, typename t>
inline cl::sycl::event rotmg(func func, cl::sycl::queue &queue, t *d1, t *d2, t *x1, t y1, t *param,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "rotmg", "for row_major layout");
}

#define rotmg_launcher_usm(type, rocblas_routine)                                         \
    cl::sycl::event rotmg(cl::sycl::queue &queue, type *d1, type *d2, type *x1, type y1, \
                          type *param,                                                   \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) { \
        return rotmg(rocblas_routine, queue, d1, d2, x1, y1, param, dependencies);        \
    }

rotmg_launcher_usm(float, rocblas_srotmg)
rotmg_launcher_usm(double, rocblas_drotmg)
#undef rotmg_launcher_usm

template <typename func, typename t>
inline cl::sycl::event iamax(func func, cl::sycl::queue &queue, int64_t n, const t *x,
                             const int64_t incx, int64_t *result,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "iamax", "for row_major layout");
}

#define iamax_launcher_usm(type, rocblas_routine)                                                \
    cl::sycl::event iamax(cl::sycl::queue &queue, int64_t n, const type *x, const int64_t incx, \
                          int64_t *result,                                                      \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {        \
        return iamax(rocblas_routine, queue, n, x, incx, result, dependencies);                  \
    }
iamax_launcher_usm(float, rocblas_isamax)
iamax_launcher_usm(double, rocblas_idamax)
iamax_launcher_usm(std::complex<float>, rocblas_icamax)
iamax_launcher_usm(std::complex<double>, rocblas_izamax)
#undef iamax_launcher_usm

template <typename func, typename t>
inline cl::sycl::event swap(func func, cl::sycl::queue &queue, int64_t n, t *x, int64_t incx, t *y,
                            int64_t incy,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "swap", "for row_major layout");
}

#define swap_launcher_usm(type, rocblas_routine)                                             \
    cl::sycl::event swap(cl::sycl::queue &queue, int64_t n, type *x, int64_t incx, type *y, \
                         int64_t incy,                                                      \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {     \
        return swap(rocblas_routine, queue, n, x, incx, y, incy, dependencies);              \
    }

swap_launcher_usm(float, rocblas_sswap)
swap_launcher_usm(double, rocblas_dswap)
swap_launcher_usm(std::complex<float>, rocblas_cswap)
swap_launcher_usm(std::complex<double>, rocblas_zswap)
#undef swap_launcher_usm

template <typename func, typename t>
inline cl::sycl::event iamin(func func, cl::sycl::queue &queue, int64_t n, const t *x,
                             const int64_t incx, int64_t *result,
                             const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "iamin", "for row_major layout");
}

#define iamin_launcher_usm(type, rocblas_routine)                                                \
    cl::sycl::event iamin(cl::sycl::queue &queue, int64_t n, const type *x, const int64_t incx, \
                          int64_t *result,                                                      \
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {        \
        return iamin(rocblas_routine, queue, n, x, incx, result, dependencies);                  \
    }
iamin_launcher_usm(float, rocblas_isamin)
iamin_launcher_usm(double, rocblas_idamin)
iamin_launcher_usm(std::complex<float>, rocblas_icamin)
iamin_launcher_usm(std::complex<double>, rocblas_izamin)
#undef iamin_launcher_usm

template <typename func, typename t1, typename t2>
inline cl::sycl::event nrm2(func func, cl::sycl::queue &queue, int64_t n, const t1 *x,
                            const int64_t incx, t2 *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    throw unimplemented("blas", "nrm2", "for row_major layout");
}

#define nrm2_launcher_usm(type1, type2, rocblas_routine)                                         \
    cl::sycl::event nrm2(cl::sycl::queue &queue, int64_t n, const type1 *x, const int64_t incx, \
                         type2 *result,                                                         \
                         const cl::sycl::vector_class<cl::sycl::event> &dependencies) {         \
        return nrm2(rocblas_routine, queue, n, x, incx, result, dependencies);                   \
    }
nrm2_launcher_usm(float, float, rocblas_snrm2)
nrm2_launcher_usm(double, double, rocblas_dnrm2)
nrm2_launcher_usm(std::complex<float>, float, rocblas_scnrm2)
nrm2_launcher_usm(std::complex<double>, double, rocblas_dznrm2)
#undef nrm2_launcher_usm

} // namespace row_major
} // namespace rocblas_
} // namespace blas
} // namespace mkl
} // namespace oneapi
