#include <vector>
#include <complex>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> fast_hadamard_transform(py::array_t<double> x_array) {
    py::buffer_info buf1 = x_array.request();
    double *x = static_cast<double *>(buf1.ptr);
    size_t n = buf1.shape[0];
    
    auto result = py::array_t<double>(buf1.size);
    py::buffer_info buf2 = result.request();
    double *y = static_cast<double *>(buf2.ptr);

    for (size_t idx = 0; idx < n; idx++)
        y[idx] = x[idx];


    int h = 1;
    while (h < n) {
        for (size_t i = 0; i < n; i += h * 2) {
            for (size_t j = i; j < std::min(i + h, n - h); j++) {
                double y_j = y[j];
                y[j] = y_j + y[j + h];
                y[j + h] = y_j - y[j + h];
            }
        }
        h *= 2;
    }
    
    double norm = std::sqrt(n);
    for (size_t i = 0; i < n; i++) {
        y[i] /= norm;
    }
    return result;
}

py::array_t<double> fft(py::array_t<double> x_array) {
    py::buffer_info buf1 = x_array.request();
    double *x = static_cast<double *>(buf1.ptr);
    size_t n = buf1.size;
    size_t logn = std::log2(n);

    auto result = py::array_t<double>(buf1.size);
    py::buffer_info buf2 = result.request();
    double *y = static_cast<double *>(buf2.ptr);
    
    for (size_t idx = 0; idx < n; idx++)
        y[idx] = x[idx];
    
    for (size_t i = 0; i < n; i++) {
        int k = 0;
        for (size_t j = 0; j < logn; j++) {
            k |= (((i >> j) & 1) << (logn - j - 1));
        }
        if (k < i)
            std::swap(y[k], y[i]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; k++) {
                double w = std::exp(-2 * M_PI * k / len) * y[i + k + len / 2];
                double v = y[i + k];
                y[i + k] = v + w;
                y[i + k + len / 2] = v - w;
            }
        }
    }
    return result;
}

PYBIND11_MODULE(transforms, m) {
    m.def("fast_hadamard_transform", &fast_hadamard_transform, 
          "Compute fast Hadamard transform in-place");
    m.def("fft", &fft, 
          "Compute FFT in-place");
}