//
// Created by hal9000 on 3/13/23.
//

#include "NumericalAnalysis.h"

#include <utility>

namespace NumericalAnalysis {
    NumericalAnalysis::NumericalAnalysis(shared_ptr<MathematicalProblem>mathematicalProblem,
                                         shared_ptr<Mesh> mesh,
                                         shared_ptr<Solver> solver,
                                         CoordinateType coordinateSystem) :
            mathematicalProblem(std::move(mathematicalProblem)), mesh(std::move(mesh)), linearSystem(nullptr),
            degreesOfFreedom(initiateDegreesOfFreedom()), solver(std::move(solver)) {
    }
    
    shared_ptr<AnalysisDegreesOfFreedom> NumericalAnalysis::initiateDegreesOfFreedom() const {
        return make_shared<AnalysisDegreesOfFreedom>(
                mesh, mathematicalProblem->boundaryConditions,mathematicalProblem->degreesOfFreedom);
    }
    
    void NumericalAnalysis::solve() const {
        solver->solve();
    }

    void NumericalAnalysis::applySolutionToDegreesOfFreedom() const {
        if (linearSystem->solution == nullptr) {
            throw runtime_error("Linear System has not been solved");
        }
        for (auto i = 0; i < degreesOfFreedom->freeDegreesOfFreedom->size(); i++) {
            degreesOfFreedom->freeDegreesOfFreedom->at(i)->setValue(linearSystem->solution->at(i));
        }
    }

    NumericalVector<double> NumericalAnalysis::getSolutionAtNode(NumericalVector<double> &nodeCoordinates, double tolerance) const {
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
                NumericalVector<double> nodeSolution = NumericalVector<double>(nodeDOFs.size());
                auto i = 0;
                for (auto &dof : nodeDOFs) {
                    nodeSolution[i] = dof->value();
                    i++;
                }
                cout<<"Node Coordinates: "<<iNodeCoords[0]<<", "<<iNodeCoords[1]<<", "<<iNodeCoords[2]<<endl;
                return nodeSolution;
            }
        }
        throw runtime_error("Target Node not found");
    }
} // NumericalAnalysis