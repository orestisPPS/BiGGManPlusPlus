//
// Created by hal9000 on 3/13/23.
//

#include "NumericalAnalysis.h"

namespace NumericalAnalysis {
    NumericalAnalysis::NumericalAnalysis(MathematicalProblem *mathematicalProblem, Mesh *mesh, Solver* solver) :
            mathematicalProblem(mathematicalProblem), mesh(mesh), linearSystem(nullptr),
            degreesOfFreedom(initiateDegreesOfFreedom()), solver(solver) {
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
    
    void NumericalAnalysis::solve() const {
        solver->solve();
        
        cout<<"Linear System solved..."<<endl;
    }
} // NumericalAnalysis