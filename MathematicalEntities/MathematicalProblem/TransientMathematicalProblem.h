//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "SteadyStateMathematicalProblem.h"

namespace MathematicalEntities {

    class TransientMathematicalProblem : public SteadyStateMathematicalProblem{
        
    public:

        TransientMathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                     shared_ptr<DomainBoundaryConditions> bcs,
                                     double initialConditionOrderZero,
                                     struct Field_DOFType *degreesOfFreedom);

        TransientMathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                     shared_ptr<DomainBoundaryConditions> bcs,
                                     double initialConditionOrderZero, double initialConditionOrderOne,
                                     struct Field_DOFType *degreesOfFreedom);
        
        
        TransientMathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                     shared_ptr<DomainBoundaryConditions> bcs,
                                     shared_ptr<map<unsigned, double>> domainInitialConditionsOrderZero,
                                     struct Field_DOFType *degreesOfFreedom);

        TransientMathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                     shared_ptr<DomainBoundaryConditions> bcs,
                                     shared_ptr<map<unsigned, double>> domainInitialConditionsOrderZero,
                                        shared_ptr<map<unsigned, double>> domainInitialConditionsOrderOne,
                                     struct Field_DOFType *degreesOfFreedom);
        
        
        double _initialConditionOrderZero;
        double _initialConditionOrderOne;
        shared_ptr<map<unsigned, double>> _domainInitialConditionsOrderZero;
        shared_ptr<map<unsigned, double>> _domainInitialConditionsOrderOne;
    };

};
