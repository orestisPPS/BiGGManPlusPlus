//
// Created by hal9000 on 8/5/23.
//

#include "DecompositionQR.h"

namespace LinearAlgebra {

    DecompositionQR::DecompositionQR(bool returnQ, ParallelizationMethod parallelizationMethod, bool storeOnMatrix) :
            _returnQ(returnQ), _parallelizationMethod(parallelizationMethod), _storeOnMatrix(storeOnMatrix) {
        _Q = nullptr;
    }

    void DecompositionQR::decompose() {
        switch (_parallelizationMethod) {
            case SingleThread:
                _singleThreadDecomposition();
                break;
            case MultiThread:
                _multiThreadDecomposition();
                break;
            case CUDA:
                _CUDADecomposition();
                break;
        }
    }
    
    void DecompositionQR::setMatrix(shared_ptr<NumericalMatrix<double>> &matrix) {
        if (_storeOnMatrix) {
            _matrix = matrix;
        }
        else {
            _matrix = matrix;
            
            if (_R == nullptr){
                _R = make_shared<NumericalMatrix<double>>(matrix->numberOfRows(), matrix->numberOfColumns());
            }
            _deepCopyMatrixIntoR();
        }
        if (_returnQ) {
            if (_Q == nullptr) {
                _Q = make_shared<NumericalMatrix<double>>(matrix->numberOfRows(), matrix->numberOfRows());
            }
            for (unsigned i = 0; i < matrix->numberOfRows(); i++) {
                for (unsigned j = 0; j < matrix->numberOfColumns(); j++){
                    _Q ->at(i, j) = 0;
                }
            }
            for (unsigned i = 0; i < matrix->numberOfRows(); i++) {
                _Q ->at(i, i) = 1;
            }
        }
    }

    const shared_ptr<NumericalMatrix<double>> &  DecompositionQR::getQ() {
    }

    const shared_ptr<NumericalMatrix<double>> &  DecompositionQR::getR() {
    }
    
    void DecompositionQR::_singleThreadDecomposition() {

    }

    void DecompositionQR::_multiThreadDecomposition() {

    }

    void DecompositionQR::_CUDADecomposition() {

    }

    void DecompositionQR::_deepCopyMatrixIntoR() {
        //R = A TODO : fix this with copy constructor
        double* dataA = _matrix->getArrayPointer();
        double* dataACopy = _R->getArrayPointer();
        for (unsigned i = 0; i < _matrix->size(); i++){
            dataACopy[i] = dataA[i];
        }
    }

    void DecompositionQR::getRQ(shared_ptr<NumericalMatrix<double>>& result) {
        switch (_parallelizationMethod) {
            case SingleThread:
                _getRQSingleThread(result);
                break;
            case MultiThread:
                _getRQMultithread(result);
                break;
            case CUDA:
                _getRQCuda(result);
        }
    }

    void DecompositionQR::_getRQSingleThread(shared_ptr<NumericalMatrix<double>>& result) {
        // Initialize result to zeros
        for (int i = 0; i < _Q->numberOfRows(); i++) {
            for (int j = 0; j < _R->numberOfColumns(); j++) {
                result->at(i, j) = 0.0;
            }
        }

        // Multiply Q and R
        for (int i = 0; i < _Q->numberOfRows(); i++) {
            for (int j = 0; j < _R->numberOfColumns(); j++) {
                // Leverage the fact that R is upper triangular
                for (int k = 0; k < _Q->numberOfColumns(); k++) { // Start from `j` instead of `0`
                    result->at(i, j) += _R->at(i, k) * _Q->at(k, j);
                }
            }
        }
    }



    void DecompositionQR::_getRQMultithread(shared_ptr<NumericalMatrix<double>>& result) {
        throw runtime_error("Not implemented");
    }

    void DecompositionQR::_getRQCuda(shared_ptr<NumericalMatrix<double>>& result) {
        throw runtime_error("Not implemented");
    }

}