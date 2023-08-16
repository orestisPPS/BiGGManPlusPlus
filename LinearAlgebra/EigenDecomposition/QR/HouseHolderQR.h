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
        explicit HouseHolderQR(bool returnQ, ParallelizationMethod parallelizationMethod = Wank, bool storeOnMatrix = false);

        const shared_ptr<Array<double>> &  getQ() override;

        const shared_ptr<Array<double>> &  getR() override;
        
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
        
        
    private:
        
        static shared_ptr<vector<double>> _calculateHouseholdVector(const shared_ptr<vector<double>>& targetVector);
        
        void _applyHouseholderProjectionOnMatrixFromLeft(unsigned column, const shared_ptr<vector<double>>& householderVector,
                                                         shared_ptr<Array<double>>& matrix);
        
        void _applyHouseholderProjectionOnMatrixFromRight(unsigned row, const shared_ptr<vector<double>>& householderVector,
                                                          shared_ptr<Array<double>>& matrix);
        
        
        static int sign(double x);
    };

} // LinearAlgebra

#endif //UNTITLED_HOUSEHOLDERQR_H
