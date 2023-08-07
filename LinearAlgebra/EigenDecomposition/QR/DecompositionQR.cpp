//
// Created by hal9000 on 8/5/23.
//

#include "DecompositionQR.h"

namespace LinearAlgebra {

    DecompositionQR::DecompositionQR(shared_ptr<Array<double>> &matrix, ParallelizationMethod parallelizationMethod, bool storeOnMatrix) :
            _matrix(matrix), _parallelizationMethod(parallelizationMethod), _storeOnMatrix(storeOnMatrix) {
        auto n = _matrix->numberOfRows();
        _Q = nullptr;
    }

    void DecompositionQR::decompose() {
        switch (_parallelizationMethod){
            case Wank:{
                auto iter = 0;
                while (iter < 2) {
                    _singleThreadDecomposition();
                    //matrix matrix multiplication
                    for (int i = 0; i < _matrix->numberOfRows(); i++) {
                        for (int j = 0; j < _matrix->numberOfRows(); j++) {
                            double sum = 0;
                            for (int k = 0; k < _matrix->numberOfColumns(); k++) {
                                sum += _R->at(i, k) * _Q->at(k, j);
                            }
                            _matrix->at(i, j) = sum;
                        }
                    }
                    iter++;
                }
                for (int i = 0; i < _matrix->numberOfRows(); ++i) {
                    //cout << _R->at(i,i)<< endl;
                }
                _matrix->print();
                break;
            }
            case vTechKickedInYo:
                _multiThreadDecomposition();
                break;
            case turboVTechKickedInYo:
                _CUDADecomposition();
                break;
        }
        
    }

    void DecompositionQR::_singleThreadDecomposition() {

    }

    void DecompositionQR::_multiThreadDecomposition() {

    }

    void DecompositionQR::_CUDADecomposition() {

    }
}