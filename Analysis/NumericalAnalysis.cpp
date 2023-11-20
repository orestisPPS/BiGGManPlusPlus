//
// Created by hal9000 on 3/13/23.
//

#include "NumericalAnalysis.h"

#include <utility>

namespace NumericalAnalysis {
    NumericalAnalysis::NumericalAnalysis(shared_ptr<MathematicalProblem> mathematicalProblem,
                                         shared_ptr<Mesh> mesh, shared_ptr<Solver> solver) :
                                         
                                         mathematicalProblem(std::move(mathematicalProblem)), mesh(std::move(mesh)),
                                         solver(std::move(solver)), coordinateSystem(Natural){
        degreesOfFreedom = std::move(initiateDegreesOfFreedom());
    }
    
    shared_ptr<AnalysisDegreesOfFreedom> NumericalAnalysis::initiateDegreesOfFreedom() const {
        return make_shared<AnalysisDegreesOfFreedom>(mesh, mathematicalProblem->boundaryConditions,
                                                     mathematicalProblem->degreesOfFreedom);
    }
    
    void NumericalAnalysis::setAnalysisCoordinateSystem(CoordinateType newCoordinateSystem) {
        this->coordinateSystem = newCoordinateSystem;
    }
    
    void NumericalAnalysis::solve(){
        solver->solve();
    }

    void NumericalAnalysis::applySolutionToDegreesOfFreedom() const {
    }

    NumericalVector<double> NumericalAnalysis::getSolutionAtNode(NumericalVector<double>& nodeCoordinates, double tolerance, DOFType dofType) const {
    
    }
} // NumericalAnalysis