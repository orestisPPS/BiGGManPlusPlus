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
        short unsigned maxDerivativeOrder = 2;
        auto templatePositionsAndPointsMap = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto qualifiedPositionsAndPoints = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto availablePositionsAndPoints = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        
        
        //Create Scheme Builder for utility functions
        auto schemeBuilder = FiniteDifferenceSchemeBuilder(_specs);

        //Define the error order for each derivative order
        auto errorOrderDerivative1 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(1);
        auto errorOrderDerivative2 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(2);


        //Define the positions needed for the scheme at each direction as well as the number of points needed.
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions, templatePositionsAndPointsMap[1]);
        schemeBuilder.templatePositionsAndPoints(2, errorOrderDerivative2, directions, templatePositionsAndPointsMap[2]);

            
        
        //Find the maximum number of neighbours needed for the desired order of accuracy. Used as input node graph depth.
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        //Iterate over the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);
            
            
            //Initiate node graph with depth equal to the maximum number of neighbours needed for the desired order of accuracy

            
            //TODO: check why node id is not ascending in direction 1. for free dof 1 (node 6)
            //      neighbours are free dof (1 (node 7), 2 (node 8), 10 (node 9). this maybe affects the sparsity pattern.

            unsigned positionI = *dof->id->value;

            auto graph =IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap, _mesh->nodesPerDirection);
            //
            
            auto availablePositionsAndDepth = *graph.getColinearPositionsAndPoints(directions);
            auto graphFilter = map<Position, short>();
            
            for (auto &direction : directions) {
                // 1) Map with template positions and the number of neighbours needed for different scheme types to achieve
                //    the desired order of accuracy. Each position vector is a map to finite difference scheme
                //    ({Left}->Backward, {Right}->Forward, {Left, Right}->Central).
                // 2) Map the available positions with the number of neighbours available
                // 3) Check if the available positions are qualified for the desired order of accuracy stencil.
                _checkIfAvailableAreQualified(availablePositionsAndDepth[direction],
                                              templatePositionsAndPointsMap[1][direction],
                                              qualifiedPositionsAndPoints[1][direction]);
                _checkIfAvailableAreQualified(availablePositionsAndDepth[direction],
                                              templatePositionsAndPointsMap[2][direction],
                                              qualifiedPositionsAndPoints[2][direction]);
                
                auto firstDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[1][direction], direction, errorOrderDerivative1, 1);
                auto secondDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[2][direction], direction, errorOrderDerivative2, 2);
                
                auto filterDerivative1 = map<Position, unsigned short>();
                for (auto &tuple : qualifiedPositionsAndPoints[1][direction]) {
                    for (auto &point : tuple.first) {
                        filterDerivative1[point] = 1;
                    }
                }
                auto nodeGraphDerivative1 = graph.getNodeGraph(filterDerivative1);
                auto colinearDOFDerivative1 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative1);
                
                auto filterDerivative2 = map<Position, unsigned short>();
                for (auto &tuple : qualifiedPositionsAndPoints[2][direction]) {
                    for (auto &point : tuple.first) {
                        filterDerivative2[point] = 1;
                    }
                }
                auto nodeGraphDerivative2 = graph.getNodeGraph(filterDerivative2);
                auto colinearDOFDerivative2 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative2);
            }
                

            //Calculate Diagonal Element
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
        }
        delete _fixedDOFMatrix;
        _fixedDOFMatrix = nullptr;
        
        //print vector
        for (auto &value: *_RHS) {
            cout << value << endl;
        }
    }

    map<short unsigned, map<Direction, map<vector<Position>, short>>> AnalysisLinearSystemInitializer::
    _initiatePositionsAndPointsMap(short unsigned& maxDerivativeOrder, vector<Direction>& directions) {
        map<short unsigned, map<Direction, map<vector<Position>, short>>> positionsAndPoints;
        for (short unsigned derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++) {
            positionsAndPoints.insert(pair<short unsigned, map<Direction, map<vector<Position>, short>>>(
                    derivativeOrder, map<Direction, map<vector<Position>, short>>()));
            for (auto &direction : directions) {
                positionsAndPoints[derivativeOrder].insert(pair<Direction, map<vector<Position>, short>>(
                        direction, map<vector<Position>, short>()));
            }
        }
        return positionsAndPoints;
    }
    
    void AnalysisLinearSystemInitializer::_checkIfAvailableAreQualified(map<vector<Position>,unsigned short>& availablePositionsAndPoints,
                                                                        map<vector<Position>,short>& templatePositionsAndPoints,
                                                                        map<vector<Position>,short>& qualifiedPositionsAndPoints){
        //Check if the specifications of the template positions and points are met in the available positions and points
        for (auto &templatePositionAndPoints : templatePositionsAndPoints) {
            for (auto &availablePositionAndPoints : availablePositionsAndPoints) {
                //Check if the template positions and points are met in the available positions and points
                if (availablePositionAndPoints.first == templatePositionAndPoints.first &&
                    availablePositionAndPoints.second >= templatePositionAndPoints.second) {
                    qualifiedPositionsAndPoints.insert(pair<vector<Position>, short>(
                            templatePositionAndPoints.first, templatePositionAndPoints.second));
                }
            }
        }
    }


}// LinearAlgebra