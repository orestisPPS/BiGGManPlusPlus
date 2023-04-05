//
// Created by hal9 000 on 3/28/23.
//

#include "LinearSystem.h"

namespace LinearAlgebra {
    
    LinearSystem::LinearSystem(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh) {
        this->_analysisDegreesOfFreedom = analysisDegreesOfFreedom;
        this->_mesh = mesh;
        this->_isoParametricNeighbourFinder = new IsoParametricNeighbourFinder(mesh);
        numberOfDOFs = new unsigned(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size());
        matrix = new Array<double>(*numberOfDOFs, *numberOfDOFs);
        RHS = new vector<double>(*numberOfDOFs);
        //matrix->print();
    }
        
    LinearSystem::~LinearSystem() {
        delete matrix;
        delete RHS;
        delete numberOfDOFs;
        delete _isoParametricNeighbourFinder;
        _mesh = nullptr;
    }
    
    void LinearSystem::createLinearSystem() {
        createMatrix();
        createRHS();
    }
    
    void LinearSystem::createMatrix() {
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);

            //I Position in the matrix (free DOF id)
            auto positionI = (*dof->id->value);
            matrix->at(positionI, positionI) = 2;

            //J Position in the matrix (neighbouring free DOF id)
            auto dofGraph = _isoParametricNeighbourFinder->getIsoParametricNodeGraph(node, 1)
                                                                      .getSpecificDOFGraph(dof->type());

            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    if (neighbourDof->id->constraintType() == Free) {
                        auto positionJ = (*neighbourDof->id->value);
                        matrix->at(positionI, positionJ) = +1;
                    }
                }
                //TODO: DEALLOCATE MAPS!
            }
            delete dofGraph;
            dofGraph = nullptr;
        }
        matrix->print();
    }
    
    void LinearSystem::createRHS() {
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);
            auto dofGraph = _isoParametricNeighbourFinder->getIsoParametricNodeGraph(node, 1)
                    .getSpecificDOFGraph(dof->type());

            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    if (neighbourDof->id->constraintType() == Fixed) {
                        auto positionI = (*dof->id->value);
                        RHS->at(positionI) += (*neighbourDof->id->value);
                    }

                }
            }
            delete dofGraph;
            dofGraph = nullptr;
        }
    }
}// LinearAlgebra