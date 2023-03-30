//
// Created by hal9000 on 3/28/23.
//

#include "LinearSystem.h"
#include "../Discretization/Node/NodeNeighbourFinder.h"

namespace LinearAlgebra {
    
    LinearSystem::LinearSystem(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh) {
        this->_analysisDegreesOfFreedom = analysisDegreesOfFreedom;
        numberOfDOFs = new unsigned(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size());
        //matrix = new Array<double>(*numberOfDOFs, *numberOfDOFs);
        //RHS = new vector<double>(*numberOfDOFs);
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
        
        for (auto &dof : *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto positionI = (*dof->id->value);
            matrix->at(positionI, positionI) = 2;
            cout<<matrix->at(positionI, positionI)<< endl;
            auto dofHood = NodeNeighbourFinder::getSpecificNeighbourDOF(mesh, dof->id->value, dof->type());
            for (auto &neighbour : dofHood) {
                auto positionJ = (*neighbour.second->id->value);
                matrix->at(positionI, positionJ) = 1;
            }
        }
        //matrix->print();
    }
    
} // LinearAlgebra