//
// Created by hal9000 on 8/6/23.
//

#include "HouseHolderQR.h"

namespace LinearAlgebra {
    HouseHolderQR::HouseHolderQR(bool returnQ, ParallelizationMethod parallelizationMethod, bool storeOnMatrix) :
                    DecompositionQR(returnQ, parallelizationMethod, storeOnMatrix){
        _decompositionType = Householder;
    }

    void HouseHolderQR::_singleThreadDecomposition() {
        shared_ptr<Array<double>> matrix;
        if (_storeOnMatrix) {
            matrix = _matrix;
        }
        else {
            matrix = _R;
        }
        
        unsigned n = _matrix->numberOfRows();
        unsigned m = _matrix->numberOfColumns();

        for (unsigned i = 0; i < m - 1; i++) {
            // Extract the i-th column of A starting at row i to n-1;
            auto iColumnOfR = matrix->getColumnPartial(i, i, n - 1);

            auto v = _calculateHouseholdVector(iColumnOfR);

            // Normalize the Householder vector
            VectorOperations::normalize(v);

            // Apply the Householder transformation directly to the matrix to get R and Q
            _applyHouseholderProjectionOnMatrixFromLeft(i, v, matrix);
            if(_returnQ)
                _applyHouseholderProjectionOnMatrixFromRight(i, v, _Q);
            
        }
    }
    
    shared_ptr<vector<double>> HouseHolderQR::_calculateHouseholdVector(const shared_ptr<vector<double>> &targetVector) {
        
            auto householdVector = make_shared<vector<double>>(*targetVector);
            
            //Calculate the norm of the i-th column of R
            double norm = VectorNorm(targetVector, L2).value();
            //α = -sign(R[i,i]) * norm = -sign(iColumnOfR[0]) * norm
            double alpha = -sign(targetVector->at(0)) * norm;
            householdVector->at(0) = targetVector->at(0) + alpha;
            for (int i = 1; i < householdVector->size(); i++) {
                householdVector->at(i) = targetVector->at(i);
            }
            VectorOperations::normalize(householdVector);
            return householdVector;
    }
    
    void HouseHolderQR::_applyHouseholderProjectionOnMatrixFromLeft(unsigned column, const shared_ptr<vector<double>> &householderVector,
                                                                    shared_ptr<Array<double>> &matrix) {
        
        auto n = matrix->numberOfRows();
        auto m = matrix->numberOfColumns();
        // Apply the Householder transformation directly to the matrix to get R
        for (unsigned col = column; col < m; col++) {
            double proj = 0.0;

            // Compute the projection
            for (unsigned row = column; row < n; row++) {
                proj += householderVector->at(row - column) * matrix->at(row, col);
            }

            proj *= 2;  // Multiply by 2 as per the Householder formula

            // Subtract the projection from the matrix
            for (unsigned row = column; row < n; row++) {
                matrix->at(row, col) -= proj * householderVector->at(row - column);
            }
        }
    }

    void HouseHolderQR::_applyHouseholderProjectionOnMatrixFromRight(unsigned column, const shared_ptr<vector<double>> &householderVector,
                                                                     shared_ptr<Array<double>> &matrix) {

        auto n = matrix->numberOfRows();
        auto m = matrix->numberOfColumns();
        // Apply the Householder transformation directly to the matrix to get Q

        for (unsigned row = 0; row < n; row++) {
            double proj = 0.0;

            // Compute the projection
            for (unsigned col = column; col < m; col++) {
                proj += matrix->at(row, col) * householderVector->at(col - column);
            }

            proj *= 2;  // Multiply by 2 as per the Householder formula

            for (unsigned col = column; col < m; col++) {
                matrix->at(row, col) -= proj * householderVector->at(col - column);
            }
        }
    }

    
    void HouseHolderQR::_multiThreadDecomposition() {
    }

    void HouseHolderQR::_CUDADecomposition() {
    } 
    
    int HouseHolderQR::sign(double x){
        if (x >= 0.0){
            return 1;
        }
        else{
            return -1;
        }
    }

    const shared_ptr<Array<double>> &  HouseHolderQR::getQ() {
        if (_returnQ){
            return _Q;
        }
        else{
            throw runtime_error("returnQ is set to false. Q is not being stored.");
        }
    }

    const shared_ptr<Array<double>> &  HouseHolderQR::getR() {
        if (_storeOnMatrix){
            return _matrix;
        }
        else{
            return _R;
        }
    }


} // LinearAlgebra