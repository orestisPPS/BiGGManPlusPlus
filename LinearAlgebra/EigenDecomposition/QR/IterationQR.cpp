//
// Created by hal9000 on 8/5/23.
//

#include "IterationQR.h"

namespace LinearAlgebra {
    IterationQR::IterationQR(unsigned maxIterations, double exitError, DecompositionType decompositionType, ParallelizationMethod parallelizationMethod, bool storeOnMatrix) :
    _maxIterations(maxIterations), _exitError(exitError), _decompositionType(decompositionType),
    _parallelizationMethod(parallelizationMethod), _storeOnMatrix(storeOnMatrix) {

        switch (_decompositionType) {
            case GramSchmidt:
                _matrixQRDecomposition = make_shared<GramSchmidtQR>(true, _parallelizationMethod, true);
                break;
            case Householder:
                _matrixQRDecomposition = make_shared<HouseHolderQR>(true, _parallelizationMethod, true);
                break;
        }
    }

    void IterationQR::calculateEigenValues() {
        while (_iteration < _maxIterations) {
            _matrixQRDecomposition->decompose();
            _Q = _matrixQRDecomposition->getQ();
            _R = _matrixQRDecomposition->getR();
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
            _iteration++;
            _matrixQRDecomposition->setMatrix(_matrix);
            auto eigenvalues = getEigenvalues();
        }
    }
    
    shared_ptr<Array<double>> IterationQR::getMatrix() {
        return _matrixQRDecomposition->getMatrix();
    }

    shared_ptr<vector<double>> IterationQR::getEigenvalues() {
        auto eigenValues = make_shared<vector<double>>(_matrix->numberOfRows());
        for (unsigned i = 0; i < _matrix->numberOfRows(); i++) {
            eigenValues->at(i) = _matrix->at(i, i);
        }
    }
    
    void IterationQR::setMatrix(shared_ptr<Array<double>>& matrix) {
        _matrix = matrix;
        _matrixSet = true;
        if (_storeOnMatrix) {
            _deepCopyMatrix();
        }
        _matrixQRDecomposition->setMatrix(_matrix);
    }


} // LinearAlgebra