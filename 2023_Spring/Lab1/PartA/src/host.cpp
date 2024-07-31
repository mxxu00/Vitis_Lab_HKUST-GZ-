#include <iostream>
#include <cstring>
#include <chrono>
#include "real.h"

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl_helper.hpp"
#define DATA_SIZE (MATRIX_M * MATRIX_N * MATRIX_K)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xclbin_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    xilinx::example_utils::XilinxOclHelper xocl;
    xocl.initialize(binaryFile);
    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl = xocl.get_kernel("real_matmul");

    size_t matrix_a_size_bytes = sizeof(real_t) * MATRIX_M * MATRIX_N;
    size_t matrix_b_size_bytes = sizeof(real_t) * MATRIX_N * MATRIX_K;
    size_t matrix_c_size_bytes = sizeof(real_t) * MATRIX_M * MATRIX_K;

    std::cout << "Allocate Buffer in Global Memory\n";
    cl::Buffer a_to_device(xocl.get_context(), CL_MEM_READ_ONLY, matrix_a_size_bytes);
    cl::Buffer b_to_device(xocl.get_context(), CL_MEM_READ_ONLY, matrix_b_size_bytes);
    cl::Buffer c_from_device(xocl.get_context(), CL_MEM_WRITE_ONLY, matrix_c_size_bytes);

    real_t* a = new real_t[MATRIX_M * MATRIX_N];
    real_t* b = new real_t[MATRIX_N * MATRIX_K];
    real_t* c = new real_t[MATRIX_M * MATRIX_K];
    real_t MatC_expected[MATRIX_M][MATRIX_K];

    std::cout << "Populating buffer inputs\n";
    for (int i = 0; i < MATRIX_M; i++) {
        for (int j = 0; j < MATRIX_N; j++) {
            a[i * MATRIX_N + j] = rand() % 50;
        }
    }

    for (int i = 0; i < MATRIX_N; i++) {
        for (int j = 0; j < MATRIX_K; j++) {
            b[i * MATRIX_K + j] = rand() % 50;
        }
    }

    for (int i = 0; i < MATRIX_M; i++) {
        for (int j = 0; j < MATRIX_K; j++) {
            MatC_expected[i][j] = 0;
            for (int p = 0; p < MATRIX_N; p++) {
                MatC_expected[i][j] += a[i * MATRIX_N + p] * b[p * MATRIX_K + j];
            }
        }
    }

    std::cout << "Copy data to device\n";
    q.enqueueWriteBuffer(a_to_device, CL_TRUE, 0, matrix_a_size_bytes, a);
    q.enqueueWriteBuffer(b_to_device, CL_TRUE, 0, matrix_b_size_bytes, b);

    std::cout << "Set kernel arguments\n";
    krnl.setArg(0, a_to_device);
    krnl.setArg(1, b_to_device);
    krnl.setArg(2, c_from_device);
    // krnl.setArg(3, MATRIX_M);
    // krnl.setArg(4, MATRIX_N);
    // krnl.setArg(5, MATRIX_K);

    std::cout << "Execution of the kernel\n";
    cl::Event event;
    q.enqueueTask(krnl, NULL, &event);
    event.wait();

    std::cout << "Read back computation results\n";
    q.enqueueReadBuffer(c_from_device, CL_TRUE, 0, matrix_c_size_bytes, c);

    bool match = true;
    for (int i = 0; i < MATRIX_M; i++) {
        for (int j = 0; j < MATRIX_K; j++) {
            if (c[i * MATRIX_K + j] != MatC_expected[i][j]) {
                match = false;
                std::cerr << "Mismatch at (" << i << ", " << j << "): " << c[i * MATRIX_K + j] << " != " << MatC_expected[i][j] << std::endl;
                break;
            }
        }
    }

    if (match) {
        std::cout << "TEST PASSED\n";
    } else {
        std::cout << "TEST FAILED\n";
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
