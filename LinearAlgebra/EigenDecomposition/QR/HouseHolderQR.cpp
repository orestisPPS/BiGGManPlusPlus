//
// Created by hal9000 on 8/6/23.
//

#include "HouseHolderQR.h"

namespace LinearAlgebra {
    HouseHolderQR::HouseHolderQR(shared_ptr<Array<double>> &matrix, ParallelizationMethod parallelizationMethod,
                                 bool storeOnMatrix) : DecompositionQR(matrix, parallelizationMethod, storeOnMatrix){
        _decompositionType = GramSchmidt;
        if (!_storeOnMatrix){
            _deepCopyMatrix();
        }
    }

    void HouseHolderQR::_singleThreadDecomposition() {
        unsigned n = _matrix->numberOfRows();
        unsigned m = _matrix->numberOfColumns();

        double norm = 0.0;
        auto alpha = 0.0;
        
        for (unsigned i = 0; i < m - 1; i++) {
            // Extract the i-th column of A starting at row i to n-1;
            auto iColumnOfR = _matrix->getColumnPartial(i, i, n - 1);

            auto v = _calculateHouseholdVector(iColumnOfR);

            // Normalize the Householder vector
            VectorOperations::normalize(v);

            

            // Apply the Householder transformation directly to the matrix 
            for (unsigned col = i; col < m; col++) {
                double proj = 0.0;

                // Compute the projection
                for (unsigned row = i; row < n; row++) {
                    proj += v->at(row - i) * _matrix->at(row, col);
                }

                proj *= 2;  // Multiply by 2 as per the Householder formula

                // Subtract the projection from the matrix
                for (unsigned row = i; row < n; row++) {
                    _matrix->at(row, col) -= proj * v->at(row - i);
                }
            }
        }
        

                _matrix->print(6);

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

    shared_ptr<Array<double>> HouseHolderQR::getQ() {
        //Implement this
    }

    shared_ptr<Array<double>> HouseHolderQR::getR() {
        return _matrix;
    }
    
    shared_ptr<Array<double>> HouseHolderQR::getMatrix() {
        if (!_storeOnMatrix) {
            return _matrix;
        }
        else {
            return _matrixCopy;
        }
    }
    
    shared_ptr<vector<double>> HouseHolderQR::getEigenvalues() {
        auto eigenValues = make_shared<vector<double>>(_matrix->numberOfRows());
        for (unsigned i = 0; i < _matrix->numberOfRows(); i++) {
            eigenValues->at(i) = _matrix->at(i, i);
        }
    }
    
    shared_ptr<vector<double>> HouseHolderQR::getSortedEigenvalues(bool ascending) {
        auto eigenValues = getEigenvalues();
        sort(eigenValues->begin(), eigenValues->end());
        if (!ascending) {
            reverse(eigenValues->begin(), eigenValues->end());
        }
        return eigenValues;
    }
    
    void HouseHolderQR::_deepCopyMatrix() {
        //R = A TODO : fix this with copy constructor
        _matrixCopy = make_shared<Array<double>>(_matrix->numberOfRows(), _matrix->numberOfColumns());
        double* dataA = _matrix->getArrayPointer();
        double* dataACopy = _matrixCopy->getArrayPointer();
        for (unsigned i = 0; i < _matrix->size(); i++){
            dataACopy[i] = dataA[i];
        }
    }
    


} // LinearAlgebra