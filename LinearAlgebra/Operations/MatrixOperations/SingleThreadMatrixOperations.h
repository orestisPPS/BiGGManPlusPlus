//
// Created by hal9000 on 8/16/23.
//

#ifndef UNTITLED_SINGLETHREADMATRIXOPERATIONS_H
#define UNTITLED_SINGLETHREADMATRIXOPERATIONS_H
#include "../../Array/Array.h"

namespace LinearAlgebra {

    // Base template
    template <typename ArrayType>
    struct Dereferencer {
        static const ArrayType& dereference(const ArrayType &matrix) {
            return matrix;
        }
    };

    // Specialization for raw pointers
    template <typename T>
    struct Dereferencer<T*> {
        static const Array<T>& dereference(const T* matrix) {
            return *matrix;
        }
    };

    // Specialization for std::unique_ptr
    template <typename T>
    struct Dereferencer<std::unique_ptr<T>> {
        static const Array<T>& dereference(const std::unique_ptr<T> &matrix) {
            return *matrix;
        }
    };

    // Specialization for std::shared_ptr
    template <typename T>
    struct Dereferencer<std::shared_ptr<T>> {
        static const Array<T>& dereference(const std::shared_ptr<T> &matrix) {
            return *matrix;
        }
    };


    template <typename T>
    class SingleThreadMatrixOperations {
        
    public:

        // In-place addition
        template <typename ArrayTypeA, typename ArrayTypeB>
        static void addInPlace(ArrayTypeA &A, const ArrayTypeB &B) {
            _checkDimensions(A, B);
            for (short unsigned i = 0; i < A.numberOfRows(); ++i) {
                for (short unsigned j = 0; j < A.numberOfColumns(); ++j) {
                    for (short unsigned k = 0; k < A.numberOfAisles(); ++k) {
                        A(i, j, k) += dereference(B)(i, j, k);
                    }
                }
            }
        }

    private:
        
        template <typename ArrayTypeA, typename ArrayTypeB>
        static void _checkDimensions(ArrayTypeA &A, const ArrayTypeB &B) {
            if (A.numberOfRows() != B.numberOfRows() ||
                A.numberOfColumns() != B.numberOfColumns() ||
                A.numberOfAisles() != B.numberOfAisles())
                throw std::runtime_error("Matrix dimensions do not match!");
        }
    };
    
}
#endif //UNTITLED_SINGLETHREADMATRIXOPERATIONS_H
