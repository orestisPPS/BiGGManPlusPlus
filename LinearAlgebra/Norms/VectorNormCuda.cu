#include "VectorNormCuda.cuh"

namespace LinearAlgebra {

    VectorNormCuda::VectorNormCuda(double* d_vector, int size, VectorNormType normType, unsigned short lP_Order)
            : _d_vector(d_vector), _size(size), _normType(normType) {
        _value = double (0.0);
        switch (_normType) {
            case L1:
                _value = calculateL1Norm();
                break;
            case L2:
                _value = calculateL2Norm();
                break;
            case LInf:
                _value = calculateLInfNorm();
                break;
            case Lp:
                _value = calculateLpNorm(lP_Order);
                break;
            default:
                throw std::invalid_argument("Invalid norm type.");
        }
    }


    VectorNormType& VectorNormCuda::type() {
        return _normType;
    }

    // Implementation for value method
    double VectorNormCuda::value() const {
        return _value;
    }
    
    
    // L1 Norm Kernel
    __global__ void l1NormKernel(double* d_vector, double* d_partialSum, int size) {
        extern __shared__ double sharedData[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        sharedData[tid] = (i < size) ? fabs(d_vector[i]) : 0;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedData[tid] += sharedData[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) d_partialSum[blockIdx.x] = sharedData[0];
    }

    // L2 Norm Kernel
    __global__ void l2NormKernel(const double* d_vector, double* d_partialSum, int size) {
        extern __shared__ double sharedData[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        sharedData[tid] = (i < size) ? d_vector[i] * d_vector[i] : 0;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedData[tid] += sharedData[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) d_partialSum[blockIdx.x] = sharedData[0];
    }

    // LInf Norm Kernel
    __global__ void lInfNormKernel(double* d_vector, double* d_partialSum, int size) {
        extern __shared__ double sharedData[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        sharedData[tid] = (i < size) ? fabs(d_vector[i]) : 0;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedData[tid] = max(sharedData[tid], sharedData[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) d_partialSum[blockIdx.x] = sharedData[0];
    }

    // Lp Norm Kernel
    __global__ void lpNormKernel(double* d_vector, double* d_partialSum, int size, double order) {
        extern __shared__ double sharedData[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        sharedData[tid] = (i < size) ? pow(fabs(d_vector[i]), order) : 0;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedData[tid] += sharedData[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) d_partialSum[blockIdx.x] = sharedData[0];
    }

    double VectorNormCuda::calculateL1Norm() {
        int blocks = (_size + 127) / 128;
        double* d_partialSums;
        double h_partialSums[blocks];
        cudaMalloc((void**)&d_partialSums, blocks * sizeof(double));

        l1NormKernel<<<blocks, 128, 128 * sizeof(double)>>>(_d_vector, d_partialSums, _size);
        cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int i = 0; i < blocks; i++) {
            sum += h_partialSums[i];
        }

        cudaFree(d_partialSums);
        return sum;
    }

    double VectorNormCuda::calculateL2Norm() {
        int blocks = (_size + 127) / 128;
        double* d_partialSums;
        double h_partialSums[blocks];
        cudaMalloc((void**)&d_partialSums, blocks * sizeof(double));

        l2NormKernel<<<blocks, 128, 128 * sizeof(double)>>>(_d_vector, d_partialSums, _size);
        cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int i = 0; i < blocks; i++) {
            sum += h_partialSums[i];
        }

        cudaFree(d_partialSums);
        return sqrt(sum);
    }

    double VectorNormCuda::calculateLInfNorm() {
        int blocks = (_size + 127) / 128;
        double* d_partialSums;
        double h_partialSums[blocks];
        cudaMalloc((void**)&d_partialSums, blocks * sizeof(double));

        lInfNormKernel<<<blocks, 128, 128 * sizeof(double)>>>(_d_vector, d_partialSums, _size);
        cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(double), cudaMemcpyDeviceToHost);

        double maxVal = 0.0;
        for (int i = 0; i < blocks; i++) {
            maxVal = std::max(maxVal, h_partialSums[i]);
        }

        cudaFree(d_partialSums);
        return maxVal;
    }

    double VectorNormCuda::calculateLpNorm(double order) {
        int blocks = (_size + 127) / 128;
        double* d_partialSums;
        double h_partialSums[blocks];
        cudaMalloc((void**)&d_partialSums, blocks * sizeof(double));

        lpNormKernel<<<blocks, 128, 128 * sizeof(double)>>>(_d_vector, d_partialSums, _size, order);
        cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int i = 0; i < blocks; i++) {
            sum += h_partialSums[i];
        }

        cudaFree(d_partialSums);
        return pow(sum, 1.0 / order);
    }


    // ... rest of your VectorNormCuda class ...

} // LinearAlgebra
