//
// Created by hal9000 on 8/6/23.
//

#ifndef UNTITLED_HOUSEHOLDERQR_H
#define UNTITLED_HOUSEHOLDERQR_H

#include "DecompositionQR.h"

namespace LinearAlgebra {

    class HouseHolderQR : public DecompositionQR {
    public:
        /**
        * @brief Constructs a GramSchmidtQR object to perform QR decomposition using the Gram-Schmidt process.
        * @param matrix Shared pointer to the matrix to decompose.
        * @param parallelizationMethod Enum indicating the method of parallelization to use.
        * @param storeOnMatrix Boolean indicating whether to store the results on the original matrix.
        */
        explicit HouseHolderQR(shared_ptr<Array<double>>& matrix, ParallelizationMethod parallelizationMethod = Wank, bool storeOnMatrix = false);
        
    protected:
        /**
        * @brief Performs QR decomposition using the Gram-Schmidt process in a single-threaded manner.
        */
        void _singleThreadDecomposition() override;
        
        /**
        * @brief Performs QR decomposition using the Gram-Schmidt process in a multi-threaded manner.
        */
        void _multiThreadDecomposition() override;
        
        /**
        * @brief Performs QR decomposition using the Gram-Schmidt process using CUDA.
        */
        void _CUDADecomposition() override;
        
        shared_ptr<Array<double>> getQ();
        
        shared_ptr<Array<double>> getR();
        
        shared_ptr<Array<double>> getMatrix();
        
        shared_ptr<vector<double>> getEigenvalues();
        
        shared_ptr<vector<double>> getSortedEigenvalues(bool ascending = false);
        
    private:
        
        static shared_ptr<vector<double>> _calculateHouseholdVector(const shared_ptr<vector<double>>& targetVector);
        
        shared_ptr<Array<double>> _matrixCopy;
        
        shared_ptr<vector<shared_ptr<vector<double>>>> _householderVectors;
        
        void _deepCopyMatrix();
        
        void _initializeArrays();
        
        static int sign(double x);
    };

} // LinearAlgebra

#endif //UNTITLED_HOUSEHOLDERQR_H
