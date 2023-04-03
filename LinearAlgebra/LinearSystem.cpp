//
// Created by hal9000 on 3/28/23.
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

        
        for (auto &dof : *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto positionI = (*dof->id->value);
            matrix->at(positionI, positionI) = 2;
            auto hoodStuff = new IsoParametricNeighbourFinder(mesh);
/*            //auto dofHood = hoodStuff->getSpecificNeighbourDOF(*dof->parentNode, dof->type(), Free, 1);
            for (auto &neighbour : *dofHood) {
*//*                    auto positionJ = (*neighbour.second.at(0)->id->value);
                    matrix->at(positionI, positionJ) = 1;*//*
            }
            delete hoodStuff;
            delete dofHood;
            dofHood = nullptr;
        }*/
/*        delete hoodStuff;
        hoodStuff = nullptr;*/
        matrix->print();

    }}
    
} // LinearAlgebra