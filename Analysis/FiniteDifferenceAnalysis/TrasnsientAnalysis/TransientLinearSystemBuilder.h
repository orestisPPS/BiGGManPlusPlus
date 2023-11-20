//
// Created by hal9000 on 3/28/23.
//
#ifndef UNTITLED_TRANSIENTLINEARSYSTEMBUILDER_H
#define UNTITLED_TRANSIENTLINEARSYSTEMBUILDER_H

#include "../../../MathematicalEntities/MathematicalProblem/TransientMathematicalProblem.h"
#include "../EquilibriumLinearSystemBuilder.h"

using namespace MathematicalEntities;

using namespace NumericalAnalysis;

namespace NumericalAnalysis {

    class TransientLinearSystemBuilder : public EquilibriumLinearSystemBuilder{

    public:
        explicit TransientLinearSystemBuilder(const shared_ptr<AnalysisDegreesOfFreedom>& analysisDegreesOfFreedom, const shared_ptr<Mesh> &mesh,
                                              const shared_ptr<TransientMathematicalProblem>& mathematicalProblem, const shared_ptr<FiniteDifferenceSchemeOrder>& specs, CoordinateType = Natural);

        
        shared_ptr<NumericalMatrix<double>> M;
        
        shared_ptr<NumericalMatrix<double>> C;
    
    void assembleMatrices();
    
    void applyInitialConditions();
    

    private:
        
        shared_ptr<TransientMathematicalProblem> _transientMathematicalProblem;
        
        shared_ptr<NumericalVector<double>> _initialConditionOrder0;
        
        shared_ptr<NumericalVector<double>> _initialConditionOrder1;
        
    };

} // LinearAlgebra

#endif //UNTITLED_TRANSIENTLINEARSYSTEMBUILDER_H
