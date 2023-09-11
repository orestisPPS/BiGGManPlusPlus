/*
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
                _matrixQRDecomposition = make_shared<GramSchmidtQR>(true, _parallelizationMethod, false);
                break;
            case Householder:
                _matrixQRDecomposition = make_shared<HouseHolderQR>(true, _parallelizationMethod, false);
                break;
        }
    }

    void IterationQR::calculateEigenvalues() {
        
        if (!_matrixSet) {
            throw runtime_error("NumericalMatrix not set");
        }
        
        while (_iteration < _maxIterations) {
            if (_iteration > 0) {
                _matrixQRDecomposition->setMatrix(_matrix);
            }
            _matrixQRDecomposition->decompose();
            _matrixQRDecomposition->getRQ(_matrix);
            //auto eigenvalues = getEigenvalues();
            _iteration++;
            cout<< "Q" << endl;
            _matrixQRDecomposition->getQ()->print(4);
            cout<< ""<< endl;
            cout<< "R" << endl;
            _matrixQRDecomposition->getR()->print(4);
            
            auto eig = getEigenvalues();
            cout<<"========== Iteration "<<_iteration<<" =========="<<endl;
            for (unsigned i = 0; i < eig->size(); i++) {
                cout<<"Eigenvalue "<<i<<" = "<<eig->at(i)<<endl;
            }
        }
*/
/*        cout<<"========== Iteration "<<_iteration<<" =========="<<endl;
        cout<<"========== Q =========="<<endl;
        _matrixQRDecomposition->getQ()->print(6);
        cout<<"========== R =========="<<endl;
        _matrixQRDecomposition->getR()->print(6);
        cout<<"========== A =========="<<endl;
        _matrix->print(6);*//*

        
*/
/*        auto eig = getEigenvalues();
        for (unsigned i = 0; i < eig->size(); i++) {
            cout<<"Eigenvalue "<<i<<" = "<<eig->at(i)<<endl;
        }*//*

        _matrixSet = false;
    }
    
    shared_ptr<NumericalVector<double>> IterationQR::getEigenvalues() {
        auto eigenValues = make_shared<NumericalVector<double>>(_matrix->numberOfRows());
        for (unsigned i = 0; i < _matrix->numberOfRows(); i++) {
            eigenValues->at(i) = _matrix->at(i, i);
        }
        return eigenValues;
    }
        
    shared_ptr<NumericalVector<double>> IterationQR::getSortedEigenvalues(bool ascending) {
        auto eigenValues = std::move(getEigenvalues());
        sort(eigenValues->begin(), eigenValues->end());
        if (!ascending) {
            reverse(eigenValues->begin(), eigenValues->end());
        }
        return eigenValues;
    }
    
    void IterationQR::setMatrix(shared_ptr<NumericalMatrix<double>>& matrix) {
        _matrix = matrix;
        _matrixSet = true;
        if (!_storeOnMatrix) {
            _deepCopyMatrix();
        }
        _matrixQRDecomposition->setMatrix(_matrix);
    }

    void IterationQR::_deepCopyMatrix() {
        //R = A TODO : fix this with copy constructor
        _matrixCopy = make_shared<NumericalMatrix<double>>(_matrix->numberOfRows(), _matrix->numberOfColumns());
        double* dataA = _matrix->getArrayPointer();
        double* dataACopy = _matrixCopy->getArrayPointer();
        for (unsigned i = 0; i < _matrix->size(); i++){
            dataACopy[i] = dataA[i];
        }
    }


} // LinearAlgebra*/
