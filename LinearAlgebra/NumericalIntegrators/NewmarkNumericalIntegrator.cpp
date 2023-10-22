//
// Created by hal9000 on 10/20/23.
//

#include "NewmarkNumericalIntegrator.h"

namespace LinearAlgebra {
    NewmarkNumericalIntegrator::NewmarkNumericalIntegrator(double alpha, double delta) : NumericalIntegrator(), _alpha(alpha), _delta(delta) {
    }
    
    void NewmarkNumericalIntegrator::solveCurrentTimeStep(unsigned stepIndex, double currentTime, double currentTimeStep) {
        _checkData();
        _checkTimeParameters();
        _checkSolver();
        
        if (stepIndex == 0)
            assembleEffectiveMatrix();
        
        assembleEffectiveRHS();
        _currentLinearSystem = make_shared<LinearSystem>(_K_hat, _RHS_hat);
        _solver->setLinearSystem(_currentLinearSystem);
        _solver->solve();
        _U_new->deepCopy(_currentLinearSystem->solution);
        _calculateHigherOrderDerivatives();
        //_calculateSecondOrderDerivative();
        //_calculateFirstOrderDerivative();
        
        _U_new->printHorizontallyWithIndex("U_new");
        //_U_dot_new->printHorizontallyWithIndex("U_dot_new");
        //_U_dot_dot_new->printHorizontallyWithIndex("U_dot_dot_new");
        
        _U_old->deepCopy(_U_new);
        _U_dot_old->deepCopy(_U_dot_new);
        _U_dot_dot_old->deepCopy(_U_dot_dot_new);
        
        results->addSolution(currentTimeStep, _U_new, _U_dot_new, _U_dot_dot_new);
    }
    
    void NewmarkNumericalIntegrator::solveCurrentTimeStepWithMatrixRebuild() {
        
    }
    
    void NewmarkNumericalIntegrator::_calculateIntegrationCoefficients() {
        _a0 = 1.0 / (_alpha * _timeStep * _timeStep);
        _a1 = _delta / (_alpha * _timeStep);
        _a2 = 1.0 / (_alpha * _timeStep);
        _a3 = 1.0 / (2.0 * _alpha) - 1.0;
        _a4 = _delta / _alpha - 1.0;
        _a5 = (_timeStep / 2.0) * (_delta / _alpha - 2.0);
        _a6 = _timeStep * (1.0 - _delta);
        _a7 = _delta * _timeStep;
    }
    
    void NewmarkNumericalIntegrator::assembleEffectiveMatrix() {
        _M->add(_C, _K_hat, _a0, _a1);
        _K_hat->add(_K, _K_hat);
    }
    
    void NewmarkNumericalIntegrator::assembleEffectiveRHS() {

        //RHS_hat = RHS +  M (a0 u + a2 u' + a3 u'') + C (a1 u + a4 u' + a5 u'')
        auto mCoefficientsJob = [&](unsigned start, unsigned end) -> void{
            for (unsigned i = start; i < end; ++i) {
                (*_sum)[i] = _a0 * (*_U_old)[i] + _a2 * (*_U_dot_old)[i] + _a3 * (*_U_dot_dot_old)[i];
            }
        };
        ThreadingOperations<double>::executeParallelJob(mCoefficientsJob, _sum->size(), 1);
        _M->multiplyVector(_sum, _matrixVectorProduct1);
        
        auto cCoefficientsJob = [&](unsigned start, unsigned end) -> void{
            for (unsigned i = start; i < end; ++i) {
                (*_sum)[i] = _a1 * (*_U_old)[i] + _a4 * (*_U_dot_old)[i] + _a5 * (*_U_dot_dot_old)[i];
            }
        };
        ThreadingOperations<double>::executeParallelJob(cCoefficientsJob, _sum->size(), 1);
        _C->multiplyVector(_sum, _matrixVectorProduct2);
        
        auto sumJob = [&](unsigned start, unsigned end) -> void{
            for (unsigned i = start; i < end; ++i) {
                (*_RHS_hat)[i] = (*_RHS)[i] + (*_matrixVectorProduct1)[i] + (*_matrixVectorProduct2)[i];
            }
        };
        ThreadingOperations<double>::executeParallelJob(sumJob, _sum->size(), 1);
    }

    void NewmarkNumericalIntegrator::_calculateSecondOrderDerivative() {
        
        auto secondOrderTerm = [&](unsigned start, unsigned end) -> void{
            for (unsigned i = start; i < end; ++i) {
                (*_U_dot_dot_new)[i] =
                        _a0 * (*_U_new)[i] - _a0 * (*_U_old)[i] - _a2 * (*_U_dot_old)[i] - _a3 * (*_U_dot_dot_old)[i];
            }
        };
        
        ThreadingOperations<double>::executeParallelJob(secondOrderTerm, _sum->size(), 1);
        
    }
    

    void NewmarkNumericalIntegrator::_calculateFirstOrderDerivative() {
        auto firstOrderTerm = [&](unsigned start, unsigned end) -> void{
            for (unsigned i = start; i < end; ++i) {
                (*_U_dot_new)[i] = (*_U_dot_old)[i] + _a6 * (*_U_dot_dot_old)[i] + _a7 * (*_U_dot_dot_new)[i];
            }
        };
        
        ThreadingOperations<double>::executeParallelJob(firstOrderTerm, _sum->size(), 1);
    }

    void NewmarkNumericalIntegrator::_calculateHigherOrderDerivatives() {
        auto calculateDerivativesJob = [&](unsigned start, unsigned end) -> void{
            for (unsigned i = start; i < end; ++i) {
                (*_U_dot_dot_new)[i] = _a0 * (*_U_new)[i] - _a0 * (*_U_old)[i] - _a2 * (*_U_dot_old)[i] - _a3 * (*_U_dot_dot_old)[i];
                (*_U_dot_new)[i] = (*_U_dot_old)[i] + _a6 * (*_U_dot_dot_old)[i] + _a7 * (*_U_dot_dot_new)[i];
            }
        };
        
        ThreadingOperations<double>::executeParallelJob(calculateDerivativesJob, _sum->size(), 1);
        
    }


}
// LinearAlgebra