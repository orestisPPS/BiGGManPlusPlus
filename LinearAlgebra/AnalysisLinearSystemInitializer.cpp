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
        _freeDOFMatrix = new Array<double>(*numberOfFreeDOFs, *numberOfFreeDOFs, 1, 0);
        _fixedDOFMatrix = new Array<double>(*numberOfFixedDOFs, *numberOfDOFs, 1, 0);
        _totalDOFMatrix = new Array<double>(*numberOfDOFs, *numberOfDOFs, 1, 0);
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
        _specs = specs;
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
        //linearSystem = new LinearSystem(_freeDOFMatrix, _RHS);
        linearSystem = new LinearSystem(_freeDOFMatrix, _RHS);
        linearSystem->exportToMatlabFile("firstCLaplacecplace.m", "/home/hal9000/code/BiGGMan++/Testing/", true);
    }
    
    void AnalysisLinearSystemInitializer::_createMatrix() {
        _createFixedDOFSubMatrix();
        _createFreeDOFSubMatrix();
        _createTotalDOFSubMatrix();
    }
    
    //Creates a matrix with consistent order across the domain. The scheme type is defined by the node neighbourhood.
    //Error Order is user defined.
    void AnalysisLinearSystemInitializer::_createFreeDOFSubMatrix() {

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
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);
        schemeBuilder.templatePositionsAndPoints(2, errorOrderDerivative2, directions,
                                                 templatePositionsAndPointsMap[2]);
        //Find the maximum number of neighbours needed for the desired order of accuracy. Used as input node graph depth.
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        unsigned positionI = 0;
        unsigned positionJ = 0;

        //Iterate over the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);

            //Find PDE Coefficients
            auto secondOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;

            //Jusqu'ici, tout va bien
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);

            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            //Jusqu'ici, tout va bien

            for (auto &direction: directions) {
                // 1) Map with template positions and the number of neighbours needed for different scheme types to achieve
                //    the desired order of accuracy. Each position vector is a map to finite difference scheme
                //    ({Left}->Backward, {Right}->Forward, {Left, Right}->Central).
                // 2) Map the available positions with the number of neighbours available
                // 3) Check if the available positions are qualified for the desired order of accuracy stencil.
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[1][direction],
                                              qualifiedPositionsAndPoints[1][direction]);
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[2][direction],
                                              qualifiedPositionsAndPoints[2][direction]);

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto firstDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[1][direction], direction, errorOrderDerivative1, 1);
                auto secondDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[2][direction], direction, errorOrderDerivative2, 2);

                auto directionIndex = spatialDirectionToUnsigned[direction];

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto firstDerivativeCoefficient = firstOrderCoefficients->at(directionIndex);
                auto secondDerivativeCoefficient = secondOrderCoefficients->at(directionIndex, directionIndex);

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto filterDerivative1 = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositionsAndPoints[1][direction]) {
                    for (auto &point: tuple.first) {
                        filterDerivative1.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto filterDerivative2 = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositionsAndPoints[2][direction]) {
                    for (auto &point: tuple.first) {
                        filterDerivative2.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto nodeGraphDerivative1 = graph.getNodeGraph(filterDerivative1);
                auto nodeGraphDerivative2 = graph.getNodeGraph(filterDerivative2);
                vector<DegreeOfFreedom*> colinearDOFDerivative1, colinearDOFDerivative2;

                colinearDOFDerivative1 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative1);
                colinearDOFDerivative2 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative2);
                
                //Calculate the fucking scheme
                positionI = *dof->id->value;
                for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                    if (colinearDOFDerivative1[iDof]->id->constraintType() == Free){
                        positionJ = *colinearDOFDerivative1[iDof]->id->value;
                        _freeDOFMatrix->at(positionI, positionJ) =
                                _freeDOFMatrix->at(positionI, positionJ) +
                                firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
                    }
                }
                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    if (colinearDOFDerivative2[iDof]->id->constraintType() == Free){
                        positionJ = *colinearDOFDerivative2[iDof]->id->value;
                        _freeDOFMatrix->at(positionI, positionJ) =
                                _freeDOFMatrix->at(positionI, positionJ) +
                                secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    }
                }
            }
        }
        cout << "Free DOF matrix" << endl;
            _freeDOFMatrix->print();
        cout << "  " << endl;
    }

    void AnalysisLinearSystemInitializer::_createFixedDOFSubMatrix() {

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
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);
        schemeBuilder.templatePositionsAndPoints(2, errorOrderDerivative2, directions,
                                                 templatePositionsAndPointsMap[2]);
        //Find the maximum number of neighbours needed for the desired order of accuracy. Used as input node graph depth.
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        unsigned positionI = 0;
        unsigned positionJ = 0;

        //Iterate over the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);

            //Find PDE Coefficients
            auto secondOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;

            //Jusqu'ici, tout va bien
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);

            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            //Jusqu'ici, tout va bien

            for (auto &direction: directions) {
                // 1) Map with template positions and the number of neighbours needed for different scheme types to achieve
                //    the desired order of accuracy. Each position vector is a map to finite difference scheme
                //    ({Left}->Backward, {Right}->Forward, {Left, Right}->Central).
                // 2) Map the available positions with the number of neighbours available
                // 3) Check if the available positions are qualified for the desired order of accuracy stencil.
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[1][direction],
                                              qualifiedPositionsAndPoints[1][direction]);
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[2][direction],
                                              qualifiedPositionsAndPoints[2][direction]);

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto firstDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[1][direction], direction, errorOrderDerivative1, 1);
                auto secondDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[2][direction], direction, errorOrderDerivative2, 2);

                auto directionIndex = spatialDirectionToUnsigned[direction];

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto firstDerivativeCoefficient = firstOrderCoefficients->at(directionIndex);
                auto secondDerivativeCoefficient = secondOrderCoefficients->at(directionIndex, directionIndex);

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto filterDerivative1 = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositionsAndPoints[1][direction]) {
                    for (auto &point: tuple.first) {
                        filterDerivative1.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto filterDerivative2 = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositionsAndPoints[2][direction]) {
                    for (auto &point: tuple.first) {
                        filterDerivative2.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto nodeGraphDerivative1 = graph.getNodeGraph(filterDerivative1);
                auto nodeGraphDerivative2 = graph.getNodeGraph(filterDerivative2);
                vector<DegreeOfFreedom*> colinearDOFDerivative1, colinearDOFDerivative2;

                colinearDOFDerivative1 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative1);
                colinearDOFDerivative2 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative2);

                //Calculate the fucking scheme
                positionI = *dof->id->value;
                for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative1[iDof]);
                    _fixedDOFMatrix->at(positionI, positionJ) =
                            _fixedDOFMatrix->at(positionI, positionJ) +
                            firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
                    bool lol = 0;
                }
                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative2[iDof]);
                    _fixedDOFMatrix->at(positionI, positionJ) =
                            _fixedDOFMatrix->at(positionI, positionJ) +
                            secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    bool lol = false;

                }
            }
        }
        cout << "Fixed DOF matrix" << endl;
        _fixedDOFMatrix->print();
        cout << "  " << endl;
    }

    void AnalysisLinearSystemInitializer::_createTotalDOFSubMatrix() {
        
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
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);
        schemeBuilder.templatePositionsAndPoints(2, errorOrderDerivative2, directions,
                                                 templatePositionsAndPointsMap[2]);
        //Find the maximum number of neighbours needed for the desired order of accuracy. Used as input node graph depth.
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        unsigned positionI = 0;
        unsigned positionJ = 0;

        //Iterate over the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->totalDegreesOfFreedom) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);

            //Find PDE Coefficients
            auto secondOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;

            //Jusqu'ici, tout va bien
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);

            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            //Jusqu'ici, tout va bien

            vector<DegreeOfFreedom*> colinearDOFDerivative1, colinearDOFDerivative2;
            for (auto &direction: directions) {
                // 1) Map with template positions and the number of neighbours needed for different scheme types to achieve
                //    the desired order of accuracy. Each position vector is a map to finite difference scheme
                //    ({Left}->Backward, {Right}->Forward, {Left, Right}->Central).
                // 2) Map the available positions with the number of neighbours available
                // 3) Check if the available positions are qualified for the desired order of accuracy stencil.
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[1][direction],
                                              qualifiedPositionsAndPoints[1][direction]);
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[2][direction],
                                              qualifiedPositionsAndPoints[2][direction]);

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto firstDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[1][direction], direction, errorOrderDerivative1, 1);
                auto secondDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[2][direction], direction, errorOrderDerivative2, 2);

                auto directionIndex = spatialDirectionToUnsigned[direction];

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto firstDerivativeCoefficient = firstOrderCoefficients->at(directionIndex);
                auto secondDerivativeCoefficient = secondOrderCoefficients->at(directionIndex, directionIndex);

                //Jusqu'ici, tout va bien
                //Mais l'important c'est pas la chute, c'est l'atterisage!!!
                auto filterDerivative1 = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositionsAndPoints[1][direction]) {
                    for (auto &point: tuple.first) {
                        filterDerivative1.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto filterDerivative2 = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositionsAndPoints[2][direction]) {
                    for (auto &point: tuple.first) {
                        filterDerivative2.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }

                auto nodeGraphDerivative1 = graph.getNodeGraph(filterDerivative1);
                auto nodeGraphDerivative2 = graph.getNodeGraph(filterDerivative2);


                if (dof->id->constraintType() == Free) {
                    colinearDOFDerivative1 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative1);
                    colinearDOFDerivative2 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative2);
                } else {
                    colinearDOFDerivative1 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative1);
                    colinearDOFDerivative2 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative2);
                    auto lol = 0;
                }

                //Calculate the fucking scheme
                positionI = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
                for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative1[iDof]);
                    _totalDOFMatrix->at(positionI, positionJ) =
                            _totalDOFMatrix->at(positionI, positionJ) +
                            firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
                    bool lol = 0;
                }
                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative2[iDof]);
                    _totalDOFMatrix->at(positionI, positionJ) =
                            _totalDOFMatrix->at(positionI, positionJ) +
                            secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    bool lol = false;

                }
            }
        }
        cout << "Total DOF matrix" << endl;
        _totalDOFMatrix->print();
        cout << "  " << endl;
    }

        
    void AnalysisLinearSystemInitializer::_createRHS() {
        //Marching through all the free DOFs
        _RHS = new vector<double>(*numberOfDOFs, 0);
          _RHS = new vector<double>(*numberOfFreeDOFs, 0);
      for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);
            
            //Get all the neighbouring DOFs with the same type
            auto dofGraph =
                    IsoParametricNodeGraph(node, 5, _parametricCoordToNodeMap, _mesh->nodesPerDirection).
                            getSpecificDOFGraph(dof->type());
            //Marching through all the neighbouring DOFs
            for (auto &neighbour: dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    //Check if the neighbouring DOF is fixed 
                    if (neighbourDof->id->constraintType() == Fixed) {
                        unsigned i = *dof->id->value;
                        unsigned j = *neighbourDof->id->value;
                        //unsigned j = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(neighbourDof);
                        _RHS->at(*dof->id->value) += _fixedDOFMatrix->at(i, j) * neighbourDof->value();
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