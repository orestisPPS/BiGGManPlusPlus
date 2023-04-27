//
// Created by hal9 000 on 3/28/23.
//

#include "AnalysisLinearSystem.h"

namespace LinearAlgebra {
    
    
    
    AnalysisLinearSystem::AnalysisLinearSystem(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh) {
        this->_analysisDegreesOfFreedom = analysisDegreesOfFreedom;
        this->_mesh = mesh;
        numberOfFreeDOFs = new unsigned(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size());
        numberOfFixedDOFs = new unsigned(_analysisDegreesOfFreedom->fixedDegreesOfFreedom->size());
        numberOfDOFs = new unsigned(_analysisDegreesOfFreedom->totalDegreesOfFreedomMap->size());
        _freeDOFMatrix = new Array<double>(*numberOfFreeDOFs, *numberOfFreeDOFs);
        _fixedDOFMatrix = new Array<double>(*numberOfFixedDOFs, *numberOfDOFs);
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
        //matrix->print();
    }
        
    AnalysisLinearSystem::~AnalysisLinearSystem() {
        delete matrix;
        delete RHS;
        delete numberOfFreeDOFs;
        delete numberOfFixedDOFs;
        delete numberOfDOFs;
        _mesh = nullptr;
    }
    
    void AnalysisLinearSystem::createLinearSystem() {
        _createMatrix();
        _createRHS();
    }
    
    void AnalysisLinearSystem::_createMatrix() {
        _createFixedDOFSubMatrix();
        _createFreeDOFSubMatrix();
    }
    
    void AnalysisLinearSystem::_createFreeDOFSubMatrix() {
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);

            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->numberOfNodesPerDirection).
                    getSpecificDOFGraph(dof->type());
            unsigned positionI = *dof->id->value;
            _freeDOFMatrix->at(positionI, positionI) = 2;

            //J Position in the matrix (neighbouring free DOF id)
            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    if (neighbourDof->id->constraintType() == Free) {
                        unsigned positionJ = *neighbourDof->id->value;
                        _freeDOFMatrix->at(positionI, positionJ) = 1;
                    }
                }
            }
            delete dofGraph;
            dofGraph = nullptr;
        }
        matrix = _freeDOFMatrix;
        _freeDOFMatrix = nullptr;
        
        cout<<"Free DOF matrix"<<endl;
        //matrix->print();
        cout << "  " << endl;

    }
    
    void AnalysisLinearSystem::_createFixedDOFSubMatrix() {
        for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);
            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->numberOfNodesPerDirection).
                            getSpecificDOFGraph(dof->type());
            unsigned positionI;
            unsigned positionJ;
            
            positionI = *dof->id->value;
            positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
            _fixedDOFMatrix->at(positionI, positionJ) = 2;
            
            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(neighbourDof);
                    _fixedDOFMatrix->at(positionI, positionJ) = 1;
                }
            }
            delete dofGraph;
            dofGraph = nullptr;
        }
        cout<<"Fixed DOF matrix"<<endl;
        //_fixedDOFMatrix->print();
    }
        
    void AnalysisLinearSystem::_createRHS() {
        //Marching through all the free DOFs
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);
            
            //Get all the neighbouring DOFs with the same type
            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->numberOfNodesPerDirection).
                            getSpecificDOFGraph(dof->type());
            //Marching through all the neighbouring DOFs
            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    //Check if the neighbouring DOF is fixed 
                    if (neighbourDof->id->constraintType() == Fixed) {
                        unsigned i = *neighbourDof->id->value;
                        unsigned j = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
                        RHS->at(*dof->id->value) -= _fixedDOFMatrix->at(i, j) * neighbourDof->value();
                    }
                }
            }            
            delete dofGraph;
            dofGraph = nullptr;
        }
        delete _fixedDOFMatrix;
        _fixedDOFMatrix = nullptr;
        
        //print vector
        for (auto &value: *RHS) {
            cout << value << endl;
        }
    }
}// LinearAlgebra