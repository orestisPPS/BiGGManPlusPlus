//
// Created by hal9000 on 10/20/23.
//

#ifndef UNTITLED_NEWMARKNUMERICALINTEGRATOR_H
#define UNTITLED_NEWMARKNUMERICALINTEGRATOR_H
#include "NumericalIntegrator.h"
namespace LinearAlgebra {

    class NewmarkNumericalIntegrator : public NumericalIntegrator{
    public:
        NewmarkNumericalIntegrator(double alpha, double delta);
        
        void solveCurrentTimeStep(unsigned stepIndex, double currentTime, double currentTimeStep) override;
        
        void solveCurrentTimeStepWithMatrixRebuild() override;

        void assembleEffectiveMatrix() override;

        void assembleEffectiveRHS() override;
        
    protected:

        
/*        void _calculateFirstOrderDerivative() override;
        
        void _calculateSecondOrderDerivative() override;*/
        
        void _calculateHigherOrderDerivatives() override;

    private:
        
        double _alpha;
        double _delta;
        double _a0;
        double _a1;
        double _a2;
        double _a3;
        double _a4;
        double _a5;
        double _a6;
        double _a7;
        
        void _calculateIntegrationCoefficients() override;
        
    };

} // LinearAlgebra

#endif //UNTITLED_NEWMARKNUMERICALINTEGRATOR_H
