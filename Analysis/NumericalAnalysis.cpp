//
// Created by hal9000 on 3/13/23.
//

#include "NumericalAnalysis.h"

namespace NumericalAnalysis {
    NumericalAnalysis::NumericalAnalysis(MathematicalProblem *mathematicalProblem, Mesh *mesh) :
            mathematicalProblem(mathematicalProblem), mesh(mesh) {
    }
    
    NumericalAnalysis::~NumericalAnalysis() {
        delete mathematicalProblem;
        delete mesh;
        delete degreesOfFreedom;
        mathematicalProblem = nullptr;
        mesh = nullptr;
        degreesOfFreedom = nullptr;
    }
    
    AnalysisDegreesOfFreedom* NumericalAnalysis::initiateDegreesOfFreedom() const {
        auto dofs = new AnalysisDegreesOfFreedom(mesh, mathematicalProblem->boundaryConditions,
                                                 mathematicalProblem->degreesOfFreedom);

        return dofs;
    }
} // NumericalAnalysis