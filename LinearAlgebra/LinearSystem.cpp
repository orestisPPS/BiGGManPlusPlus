//
// Created by hal9 000 on 3/28/23.
//

#include "LinearSystem.h"
#include "../Discretization/Node/IsoParametricNeighbourFinder.h"
#include "../DegreesOfFreedom/IsoParametricDOFFinder.h"

namespace LinearAlgebra {
    
    LinearSystem::LinearSystem(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh) {
        this->_analysisDegreesOfFreedom = analysisDegreesOfFreedom;
        numberOfDOFs = new unsigned(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size());
        matrix = new Array<double>(*numberOfDOFs, *numberOfDOFs);
        RHS = new vector<double>(*numberOfDOFs);
        //matrix->print();
    }
        
    LinearSystem::~LinearSystem() {
        delete matrix;
        delete RHS;
        delete numberOfDOFs;
    }
    
    void LinearSystem::createLinearSystem(Mesh* mesh) {
        createMatrix(mesh);
        //createRHS();
    }
    
    void LinearSystem::createMatrix(Mesh* mesh) {
        auto hoodStuff = IsoParametricNeighbourFinder(mesh);
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //Node with the DOF
            auto node = mesh->nodeFromID(*dof->parentNode);

            //I Position in the matrix (free DOF id)
            auto positionI = (*dof->id->value);
            matrix->at(positionI, positionI) = 2;

            //J Position in the matrix (neighbouring free DOF id)
            auto dofGraph =
                    //hoodStuff.getIsoParametricNodeGraph(node, 1).getSpecificDOFGraph(dof->type(), Free);
                    hoodStuff.getIsoParametricNodeGraph(node, 1).getSpecificDOFGraph(dof->type());

            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    if (neighbourDof->id->constraintType() == Free) {
                        auto positionJ = (*neighbourDof->id->value);
                        matrix->at(positionI, positionJ) = +1;
                    }
                }
                //TODO: DEALLOCATE MAPS!
            }

        }
        matrix->print();
    }
}// LinearAlgebra