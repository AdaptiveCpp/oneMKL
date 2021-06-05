#include "rocblas_scope_handle_hipsycl.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {

rocblas_handle_container::~rocblas_handle_container() noexcept(false) {
    for (auto &handle_pair : rocblas_handle_mapper_) {
        rocblas_status err;
        if (handle_pair.second != nullptr) {
            auto handle = handle_pair.second->exchange(nullptr);
            if (handle != nullptr) {
                ROCBLAS_ERROR_FUNC(rocblas_destroy_handle, err, handle);
                handle = nullptr;
            }
            delete handle_pair.second;
            handle_pair.second = nullptr;
        }
    }
    rocblas_handle_mapper_.clear();
}

thread_local rocblas_handle_container RocblasScopedContextHandler::handle_helper = rocblas_handle_container{};

RocblasScopedContextHandler::RocblasScopedContextHandler(cl::sycl::queue queue, cl::sycl::interop_handle& ih): interop_h(ih){}

rocblas_handle RocblasScopedContextHandler::get_handle(const cl::sycl::queue &queue){
    cl::sycl::device device = queue.get_device();
    int current_device = interop_h.get_native_device<cl::sycl::backend::hip>();
    hipStream_t streamId = get_stream(queue);
    rocblas_status err;
    auto it = handle_helper.rocblas_handle_mapper_.find(current_device);
    if (it != handle_helper.rocblas_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.rocblas_handle_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                hipStream_t currentStreamId;
                ROCBLAS_ERROR_FUNC(rocblas_get_stream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    ROCBLAS_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.rocblas_handle_mapper_.erase(it);
            }
        }
    }
    rocblas_handle handle;

    ROCBLAS_ERROR_FUNC(rocblas_create_handle, err, &handle);
    ROCBLAS_ERROR_FUNC(rocblas_set_stream, err, handle, streamId);

    auto insert_iter = handle_helper.rocblas_handle_mapper_.insert(
        std::make_pair(current_device, new std::atomic<rocblas_handle>(handle)));
    return handle;
}

hipStream_t RocblasScopedContextHandler::get_stream(const cl::sycl::queue &queue) {
        return interop_h.get_native_queue<cl::sycl::backend::hip>();
    }


} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi