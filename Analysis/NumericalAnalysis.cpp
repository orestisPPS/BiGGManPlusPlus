//
// Created by hal9000 on 3/13/23.
//

#include "NumericalAnalysis.h"

namespace NumericalAnalysis {
    NumericalAnalysis::NumericalAnalysis(MathematicalProblem *mathematicalProblem, Mesh *mesh, Solver* solver, CoordinateType coordinateSystem) :
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

    void NumericalAnalysis::applySolutionToDegreesOfFreedom() const {
        for (auto i = 0; i < degreesOfFreedom->freeDegreesOfFreedom->size(); i++) {
            degreesOfFreedom->freeDegreesOfFreedom->at(i)->setValue(linearSystem->solution->at(i));
        }
    }

    vector<double> NumericalAnalysis::getSolutionAtNode(vector<double> &nodeCoordinates, double tolerance) const {
        switch (nodeCoordinates.size()) {
            case 1:
                nodeCoordinates.push_back(0.0);
                nodeCoordinates.push_back(0.0);
                break;
            case 2:
                nodeCoordinates.push_back(0.0);
                break;
            case 3:
                break;
            default:
                throw runtime_error("Target Node coordinates must have size of 1, 2 or 3");
        }
        auto coordMap = mesh->getCoordinateToNodeMap();
        for (auto &coord :coordMap) {
            auto iNodeCoords = coord.first;
            if (abs(iNodeCoords[0] - nodeCoordinates[0]) < tolerance &&
                abs(iNodeCoords[1] - nodeCoordinates[1]) < tolerance &&
                abs(iNodeCoords[2] - nodeCoordinates[2]) < tolerance) {
                auto node = coord.second;
                auto nodeDOFs = node->degreesOfFreedom;
                vector<double> nodeSolution(nodeDOFs->size());
                for (auto i = 0; i < nodeDOFs->size(); i++) {
                    nodeSolution[i] = nodeDOFs->at(i)->value();
                }
                return nodeSolution;
            }
        }
    }
} // NumericalAnalysis