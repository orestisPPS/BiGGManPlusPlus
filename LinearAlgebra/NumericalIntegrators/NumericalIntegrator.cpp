//
// Created by hal9000 on 10/8/23.
//

#include "NumericalIntegrator.h"

namespace LinearAlgebra {
    
    NumericalIntegrator::NumericalIntegrator() = default;
    
    void NumericalIntegrator::setTimeParameters(double initialTime, double finalTime, unsigned int totalSteps) {
        _initialTime = initialTime;
        _finalTime = finalTime;
        _timeStep = (finalTime - initialTime) / totalSteps;
        _currentTime = initialTime;
        _numberOfTimeSteps = totalSteps;
        _currentStep = 0;
        _calculateIntegrationCoefficients();
        _initializeMethodVectors();
        _timeParametersSet = true;
        results = make_unique<NumericalAnalysis::TransientAnalysisResults>(_initialTime, _numberOfTimeSteps, _timeStep,
                                                                            _RHS->size(), true, true);
    }

    void NumericalIntegrator::setSolver(shared_ptr<Solver> solver) {
        _solver = std::move(solver);
        _solverSet = true;
    }
    
    void NumericalIntegrator::assembleEffectiveMatrix() {
        
    }
    
    void NumericalIntegrator::assembleEffectiveRHS() {
        
    }
    
    void NumericalIntegrator::setMatricesAndVector(shared_ptr<NumericalMatrix<double>> & M,
                                                   shared_ptr<NumericalMatrix<double>> & C,
                                                   shared_ptr<NumericalMatrix<double>> & K,
                                                   shared_ptr<NumericalVector<double>> & RHS) {
        _M = std::move(M);
        _C = std::move(C);
        _K = std::move(K);
        _RHS_hat = make_shared<NumericalVector<double>>(RHS->size());
        _RHS_hat->deepCopy(RHS);
        _RHS = std::move(RHS);
        
        if (_K_hat == nullptr){
            _K_hat = make_shared<NumericalMatrix<double>>(_K->numberOfRows(), _K->numberOfColumns());
        }
        _initializeMethodVectors();
        _dataSet = true;
    }
    
    void NumericalIntegrator::setMatricesAndVector(shared_ptr<NumericalMatrix<double>> & C,
                                                   shared_ptr<NumericalMatrix<double>> & K,
                                                   shared_ptr<NumericalVector<double>> & RHS) {
        _M = nullptr;
        _C = std::move(C);
        _K = std::move(K);
        _RHS_hat = make_shared<NumericalVector<double>>(RHS->size());
        _RHS_hat->deepCopy(RHS);
        _RHS = std::move(RHS);
        
        if (_K_hat == nullptr){
            _K_hat = make_shared<NumericalMatrix<double>>(_K->numberOfRows(), _K->numberOfColumns());
        }
        _initializeMethodVectors();
        _dataSet = true;
    }
    
    void NumericalIntegrator::solveCurrentTimeStep(unsigned stepIndex, double currentTime, double currentTimeStep) {

    }
    
    void NumericalIntegrator::solveCurrentTimeStepWithMatrixRebuild() {

    }

    shared_ptr<LinearSystem>  NumericalIntegrator::assembleEffectiveLinearSystem() {
        _checkData();
        _checkTimeParameters();
        assembleEffectiveMatrix();
        assembleEffectiveRHS();
        _currentLinearSystem = make_shared<LinearSystem>(_K_hat, _RHS_hat);
        _effectiveLinearSystemAssembled = true;
        return _currentLinearSystem;
    }
    
    void NumericalIntegrator::_calculateFirstOrderDerivative() {
        
    }
    
    void NumericalIntegrator::_calculateSecondOrderDerivative() {
        
    }

    void NumericalIntegrator::_initializeMethodVectors() {
        _U_old = make_shared<NumericalVector<double>>(_RHS->size());
        _U_dot_old = make_shared<NumericalVector<double>>(_RHS->size());
        _U_dot_dot_old = make_shared<NumericalVector<double>>(_RHS->size());
        _U_new = make_shared<NumericalVector<double>>(_RHS->size());
        _U_dot_new = make_shared<NumericalVector<double>>(_RHS->size());
        _U_dot_dot_new = make_shared<NumericalVector<double>>(_RHS->size());
        _sum = make_shared<NumericalVector<double>>(_RHS->size());
        _matrixVectorProduct1 = make_shared<NumericalVector<double>>(_RHS->size());
        _matrixVectorProduct2 = make_shared<NumericalVector<double>>(_RHS->size());
    }
    void NumericalIntegrator::_calculateHigherOrderDerivatives() {

    }
    
    void NumericalIntegrator::_calculateIntegrationCoefficients() {

    }

    void NumericalIntegrator::_checkSolver() const {
        if (!_solverSet)
            throw runtime_error("NumericalIntegrator::_checkSolver: Solver not set");
    }
    
    void NumericalIntegrator::_checkTimeParameters() const {
        if (!_timeParametersSet)
            throw runtime_error("NumericalIntegrator::_checkTimeParameters: Time parameters not set");
    }
    
    void NumericalIntegrator::_checkData() const {
        if (!_dataSet)
            throw runtime_error("NumericalIntegrator::_checkData: Data not set");
    }





} // LinearAlgebra