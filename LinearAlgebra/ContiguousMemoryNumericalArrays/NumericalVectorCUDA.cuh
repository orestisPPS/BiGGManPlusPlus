//
// Created by hal9000 on 8/31/23.
//

#ifndef UNTITLED_NUMERICALVECTORCUDA_CUH
#define UNTITLED_NUMERICALVECTORCUDA_CUH

#include "../../UtilityCUDA/MemoryManagementCUDA.cuh"
#include "NumericalVector/NumericalVector.h"
using namespace LinearAlgebra;


namespace LinearAlgebra {

    template<typename T>
    class NumericalVectorCUDA {
    public:
        
        NumericalVectorCUDA(unsigned int size, T initialValue = 0, bool allocateOtherVector = true, bool allocateResultVector = true) {
            _size = size;
            _allocateOtherVector = allocateOtherVector;
            _allocateResultVector = allocateResultVector;

            // Allocate and initialize host memory
            _h_values = new T[_size];
            for (unsigned int i = 0; i < _size; i++) {
                _h_values[i] = initialValue;
            }

            _h_otherVector = _allocateOtherVector ? new T[_size] : nullptr;
            _h_resultVector = _allocateResultVector ? new T[_size] : nullptr;

            // Allocate device memory
            MemoryManagementCUDA<T>::allocateDeviceMemory(&_d_values, _size);
            if (_allocateOtherVector) MemoryManagementCUDA<T>::allocateDeviceMemory(&_d_otherVector, _size);
            if (_allocateResultVector) MemoryManagementCUDA<T>::allocateDeviceMemory(&_d_resultVector, _size);

            // Copy initialized values to device memory
            MemoryManagementCUDA<T>::copyToDevice(_d_values, _h_values, _size);
        }


        ~NumericalVectorCUDA() {
            // Free device memory
            MemoryManagementCUDA<T>::freeDeviceMemory(_d_values);
            if (_allocateOtherVector) MemoryManagementCUDA<T>::freeDeviceMemory(_d_otherVector);
            if (_allocateResultVector) MemoryManagementCUDA<T>::freeDeviceMemory(_d_resultVector);

            // Delete host memory
            delete[] _h_values;
            if (_allocateOtherVector) delete[] _h_otherVector;
            if (_allocateResultVector) delete[] _h_resultVector;
        }

        // Basic operation - vector addition (placeholder)
        void add(const NumericalVectorCUDA<T>& other) {
            // TODO: Implement CUDA kernel for vector addition
        }

    private:
        unsigned int _size;
        bool _allocateOtherVector;
        bool _allocateResultVector;
        T* _h_values;
        T* _h_otherVector;
        T* _h_resultVector;
        T* _d_values;
        T* _d_otherVector;
        T* _d_resultVector;
    };


} // LinearAlgebra

#endif //UNTITLED_NUMERICALVECTORCUDA_CUH
