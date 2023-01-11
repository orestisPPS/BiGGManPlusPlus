//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../BoundaryConditions/BoundaryCondition.h"
#include "../PositioningInSpace/CoordinateSystem.h"

namespace MathematicalProblem {

    class TransientMathematicalProblem {
    public:
        TransientMathematicalProblem(PartialDifferentialEquation *pde,
                                      map<Position,list<BoundaryConditions::BoundaryCondition*>> *bcs,
                                      double* ic,
                                      list<DegreeOfFreedom*> *dof,
                                      CoordinateSystem coordinateSystem);

        TransientMathematicalProblem(PartialDifferentialEquation *pde,
                                     map<Position,list<BoundaryConditions::BoundaryCondition*>> *bcs,
                                     map<int*,double>* domainIC,
                                     list<DegreeOfFreedom*> *dof,
                                     CoordinateSystem coordinateSystem);
                
        PartialDifferentialEquation *pde;
        map<Position,list<BoundaryConditions::BoundaryCondition*>> *boundaryConditions;
        double* initialCondition;
        map<int*,double>* domainInitialConditions;
        list<DegreeOfFreedom*> *degreesOfFreedom;
        CoordinateSystem coordinateSystem;
    };

};
