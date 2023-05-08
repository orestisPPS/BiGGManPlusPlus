//
// Created by hal9 000 on 3/28/23.
//

#include "AnalysisLinearSystemInitializer.h"

namespace LinearAlgebra {
    
    
    
    AnalysisLinearSystemInitializer::AnalysisLinearSystemInitializer(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh,
                                                                     MathematicalProblem* problem, FDSchemeSpecs* specs) {
        linearSystem = nullptr;
        this->_analysisDegreesOfFreedom = analysisDegreesOfFreedom;
        this->_mesh = mesh;
        this->_mathematicalProblem = problem;
        numberOfFreeDOFs = new unsigned(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size());
        numberOfFixedDOFs = new unsigned(_analysisDegreesOfFreedom->fixedDegreesOfFreedom->size());
        numberOfDOFs = new unsigned(_analysisDegreesOfFreedom->totalDegreesOfFreedomMap->size());
        _freeDOFMatrix = new Array<double>(*numberOfFreeDOFs, *numberOfFreeDOFs);
        _fixedDOFMatrix = new Array<double>(*numberOfFixedDOFs, *numberOfDOFs);
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
        _specs = specs;
        //matrix->print();
    }
        
    AnalysisLinearSystemInitializer::~AnalysisLinearSystemInitializer() {
        delete linearSystem;
        delete numberOfFreeDOFs;
        delete numberOfFixedDOFs;
        delete numberOfDOFs;
        _mesh = nullptr;
    }
    
    void AnalysisLinearSystemInitializer::createLinearSystem() {
        _createMatrix();
        _createRHS();
        linearSystem = new LinearSystem(_matrix, _RHS);
    }
    
    void AnalysisLinearSystemInitializer::_createMatrix() {
        _createFixedDOFSubMatrix();
        _createFreeDOFSubMatrix();
    }
    
    //Creates a matrix with consistent order across the domain. The scheme type is defined by the node neighbourhood.
    //Error Order is user defined.
    void AnalysisLinearSystemInitializer::_createFreeDOFSubMatrix() {
        
        bool applyForwardScheme, applyBackwardScheme, applyCentralScheme;
        
        //Define the directions of the simulation
        auto directions = _mesh->directions();

        //Create Scheme Builder for utility functions
        auto schemeBuilder = FiniteDifferenceSchemeBuilder(_specs);

        //Define the error order for each derivative order
        auto errorOrderDerivative1 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(1);
        auto errorOrderDerivative2 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(2);


        //Define the positions needed for the scheme at each direction as well as the number of points needed.
        auto templatePositionsAndPointsDerivative1 = schemeBuilder.templatePositionsAndPoints(
                1, errorOrderDerivative1, directions);
        auto templatePositionsAndPointsDerivative2 = schemeBuilder.templatePositionsAndPoints(
                2, errorOrderDerivative2, directions);
        
        //Find the maximum number of neighbours needed for the desired order of accuracy. Used as input node graph depth.
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        //Iterate over the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);

            //Initiate node graph with depth equal to the maximum number of neighbours needed for the desired order of accuracy
            auto graph =IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap, _mesh->nodesPerDirection);
            //TODO: check why node id is not ascending in direction 1. for free dof 1 (node 6)
            //      neighbours are free dof (1 (node 7), 2 (node 8), 10 (node 9). this maybe affects the sparsity pattern.
            auto neighbourDOF = graph.getSpecificDOFGraph(dof->type());
            auto positionsAndDepth = graph.getColinearPositionsAndPoints(directions);

            for (auto &direction : directions) {

                //Map with available positions and the number of neighbours available
                auto availablePositionsAndPointsAtDirection = positionsAndDepth->at(direction);
                
                //Map with template positions and the number of neighbours needed for different scheme types to achieve
                // the desired order of accuracy.Each position vector is a map to finite difference scheme
                // ({Left}->Backward, {Right}->Forward, {Left, Right}->Central).
                auto templatePositionsAndPointsAtDirectionDerivative1 = templatePositionsAndPointsDerivative1[direction];

                //Iterate over the template position vectors
                auto templatePositionsVector = vector<Position>();
                auto templatePoints = 0;
                for (auto &templatePositionAndPoints: templatePositionsAndPointsAtDirectionDerivative1) {
                    templatePositionsVector = get<0>(templatePositionAndPoints);
                    templatePoints = get<1>(templatePositionAndPoints);
                    
                    
                    
                    
                    
                    
                    
                    //Iterate over the available positions
                    for (auto &availablePositionAndPoints: availablePositionsAndPointsAtDirection) {
                        //Position of the available DOF
                        auto availablePosition = get<0>(availablePositionAndPoints);
                        //Number of neighbours available
                        auto availablePoints = get<1>(availablePositionAndPoints);
                        
                        //Iterate over the template positions and the number of neighbours needed for different scheme
                        //types to achieve the desired order of accuracy
                        for (auto &templatePosition: templatePositionsVector) {
                            //If the available position is equal to the template position and the number of neighbours
                            //available is equal to the number of neighbours needed for the desired order of accuracy
/*                            if (availablePosition == templatePosition && availablePoints >= templatePoints) {
                                cout<<"MITSOTAKI GAMIESAI"<<endl;
     *//*                           //Apply the scheme type
                                if (templatePosition == Position::Left) {
                                    applyBackwardScheme = true;
                                } else if (templatePosition == Position::Right) {
                                    applyForwardScheme = true;
                                } else if (templatePosition == Position::LeftRight) {
                                    applyCentralScheme = true;
                                }*//*
                            }*/
                        }
                    }
                    
                    
                    bool lolipop = false;
                    
                }
            }
                


            //Calculate Diagonal Element
            unsigned positionI = *dof->id->value;
            auto valueICoefficient = 0.0;
            for (auto &direction : directions) {

            }

/*            _freeDOFMatrix->at(positionI, positionI) = -2;

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
            dofGraph = nullptr;*/




            //Find PDE Coefficients
            auto secondOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;
        }

        _matrix = _freeDOFMatrix;
        _freeDOFMatrix = nullptr;

        cout<<"Free DOF matrix"<<endl;
        //matrix->print();
        cout << "  " << endl;

    }
    
    
    
    
    
    void AnalysisLinearSystemInitializer::_createFixedDOFSubMatrix() {
        for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);
            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->nodesPerDirection).
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
        
    void AnalysisLinearSystemInitializer::_createRHS() {
        //Marching through all the free DOFs
        _RHS = new vector<double>(*numberOfFreeDOFs, 0);
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);
            
            //Get all the neighbouring DOFs with the same type
            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->nodesPerDirection).
                            getSpecificDOFGraph(dof->type());
            //Marching through all the neighbouring DOFs
            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    //Check if the neighbouring DOF is fixed 
                    if (neighbourDof->id->constraintType() == Fixed) {
                        unsigned i = *neighbourDof->id->value;
                        unsigned j = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
                        _RHS->at(*dof->id->value) -= _fixedDOFMatrix->at(i, j) * neighbourDof->value();
                    }
                }
            }            
            delete dofGraph;
            dofGraph = nullptr;
        }
        delete _fixedDOFMatrix;
        _fixedDOFMatrix = nullptr;
        
        //print vector
        for (auto &value: *_RHS) {
            cout << value << endl;
        }
    }
}// LinearAlgebra