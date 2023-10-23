//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"  

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
            shared_ptr<SteadyStateMathematicalProblem> mathematicalProblem,
            shared_ptr<Mesh> mesh,
            shared_ptr<Solver> solver,
            shared_ptr<FDSchemeSpecs> schemeSpecs, CoordinateType coordinateSystem) :
            FiniteDifferenceAnalysis(mathematicalProblem, std::move(mesh), std::move(solver), std::move(schemeSpecs), coordinateSystem),
            steadyStateMathematicalProblem(std::move(mathematicalProblem)){
        
        auto linearSystemInitializer = make_unique<EquilibriumLinearSystemBuilder>(
                degreesOfFreedom, this->mesh, steadyStateMathematicalProblem, this->schemeSpecs, this->coordinateSystem);
        linearSystemInitializer->assembleSteadyStateLinearSystem();
        this->linearSystem = make_shared<LinearSystem>(linearSystemInitializer->K, linearSystemInitializer->RHS);
        this->solver->setLinearSystem(linearSystem);
    }
    
    void SteadyStateFiniteDifferenceAnalysis::solve() {
        solver->solve();
    }

    void SteadyStateFiniteDifferenceAnalysis::applySolutionToDegreesOfFreedom() const {
        if (linearSystem->solution == nullptr) {
            throw runtime_error("Linear System has not been solved");
        }
        for (auto i = 0; i < degreesOfFreedom->freeDegreesOfFreedom->size(); i++) {
            degreesOfFreedom->freeDegreesOfFreedom->at(i)->setValue(linearSystem->solution->at(i));
        }
    }

    NumericalVector<double>
    SteadyStateFiniteDifferenceAnalysis::getSolutionAtNode(NumericalVector<double>& nodeCoordinates, double tolerance, DOFType dofType) const {
        switch (nodeCoordinates.size()) {
            case 1:
                nodeCoordinates[1] = 0.0;
                nodeCoordinates[2] = 0.0;
                break;
            case 2:
                nodeCoordinates[2] = 0.0;
                break;
            case 3:
                break;
            default:
                throw runtime_error("Target Node coordinates must have size of 1, 2, or 3");
        }
        auto coordMap = mesh->getCoordinatesToNodesMap();
        for (auto &coord : *coordMap) {
            auto iNodeCoords = coord.first;
            if (abs(iNodeCoords[0] - nodeCoordinates[0]) < tolerance &&
                abs(iNodeCoords[1] - nodeCoordinates[1]) < tolerance &&
                abs(iNodeCoords[2] - nodeCoordinates[2]) < tolerance) {
                auto node = coord.second;
                auto nodeDOFs = *node->degreesOfFreedom;
                auto i = 0;
                if (dofType != NoDOFType){
                    for (auto &dof : nodeDOFs) {
                        if (dof->type() == dofType)
                            return NumericalVector<double>({dof->value()});
                    }
                }
                else{
                    auto result = NumericalVector<double>(nodeDOFs.size());
                    for (auto &dof : nodeDOFs) {
                        result[i] = dof->value();
                        i++;
                    }
                    return result;
                }
            }
        }
        throw runtime_error("Target Node not found");
    }
    


    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis