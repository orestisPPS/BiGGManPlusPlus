//
// Created by hal9 000 on 3/28/23.
//

#include "AnalysisLinearSystemInitializer.h"

namespace LinearAlgebra {



    AnalysisLinearSystemInitializer::
    AnalysisLinearSystemInitializer(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh, MathematicalProblem* problem,
                                    FDSchemeSpecs* specs, CoordinateType coordinateType) :
                                    linearSystem(nullptr),
                                    _analysisDegreesOfFreedom(analysisDegreesOfFreedom),
                                    _mesh(mesh),
                                    _mathematicalProblem(problem),
                                    _specs(specs),
                                    _coordinateType(coordinateType) {
        _rhsVector = new vector<double>(*_analysisDegreesOfFreedom->numberOfDOFs, 0);
        _totalDOFMatrix = new Array<double>(*_analysisDegreesOfFreedom->numberOfDOFs, *_analysisDegreesOfFreedom->numberOfDOFs, 1, 0);
        _freeFreeMatrix = new Array<double>(*_analysisDegreesOfFreedom->numberOfFreeDOFs, *_analysisDegreesOfFreedom->numberOfFreeDOFs, 1, 0);
        _fixedFreeMatrix = new Array<double>(*_analysisDegreesOfFreedom->numberOfFreeDOFs, *_analysisDegreesOfFreedom->numberOfFixedDOFs, 1, 0);
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
    }

    AnalysisLinearSystemInitializer::~AnalysisLinearSystemInitializer() {
        delete linearSystem;
        _mesh = nullptr;
    }

    void AnalysisLinearSystemInitializer::createLinearSystem() {
        assembleMatrices();
        _createRHS();
        linearSystem = new LinearSystem(_freeFreeMatrix, _rhsVector);
        //linearSystem->exportToMatlabFile("firstCLaplacecplace.m", "/home/hal9000/code/BiGGMan++/Testing/", true);
    }

    void AnalysisLinearSystemInitializer::assembleMatrices() {
        _createTotalDOFMatrix();
        _createFreeFreeDOFSubMatrix();
        _createFreeFixedDOFSubMatrix();
    }

    //Creates a matrix with consistent order across the domain. The scheme type is defined by the node neighbourhood.
    //Error Order is user defined.
#include <chrono>

    void AnalysisLinearSystemInitializer::_createFreeFreeDOFSubMatrix() {
        auto start = std::chrono::steady_clock::now(); // Start the timer

        auto &totalDOF = *_analysisDegreesOfFreedom->numberOfFreeDOFs;
        for (unsigned i = 0; i < totalDOF; i++) {
            for (unsigned j = 0; j < totalDOF; j++) {
                _freeFreeMatrix->at(i, j) = _totalDOFMatrix->at(i, j);
            }
        }

        auto end = std::chrono::steady_clock::now(); // Stop the timer
        
        cout << "Free - Free DOF Sub-Matrix Assembled in : " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
        //_freeFreeMatrix->print();
    }

    void AnalysisLinearSystemInitializer::_createFreeFixedDOFSubMatrix() {
        auto start = std::chrono::steady_clock::now(); // Start the timer
        unsigned & minIndex = *_analysisDegreesOfFreedom->numberOfFreeDOFs;
        for (unsigned i = 0; i < *_analysisDegreesOfFreedom->numberOfFreeDOFs; i++) {
            for (unsigned j = 0; j < *_analysisDegreesOfFreedom->numberOfFixedDOFs; j++) {
                _fixedFreeMatrix->at(i, j) = _totalDOFMatrix->at(i, j + minIndex);
            }
        }
        delete _totalDOFMatrix;
        _totalDOFMatrix = nullptr;
        
        auto end = std::chrono::steady_clock::now(); // Stop the timer
        cout << "Fixed - Free Sub - Matrix Assembled in : " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
        //_fixedFreeMatrix->print();
        cout << "  " << endl;
    }

    void AnalysisLinearSystemInitializer::_createTotalDOFMatrix() {
        auto start = std::chrono::steady_clock::now(); // Start the timer

        auto directions = _mesh->directions();
        short unsigned maxDerivativeOrder = 2;
        auto templatePositionsAndPointsMap = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto qualifiedPositionsAndPoints = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto schemeBuilder = FiniteDifferenceSchemeBuilder(_specs);

        auto errorOrderDerivative1 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(1);
        auto errorOrderDerivative2 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(2);

        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);
        schemeBuilder.templatePositionsAndPoints(2, errorOrderDerivative2, directions,
                                                 templatePositionsAndPointsMap[2]);

        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();
        vector<DegreeOfFreedom*>  colinearDOFDerivative1, colinearDOFDerivative2;
        for (auto &dof: *_analysisDegreesOfFreedom->totalDegreesOfFreedom) {

            auto node = _mesh->nodeFromID(*dof->parentNode);
            auto secondOrderCoefficients = _mathematicalProblem->pde->properties->getLocalProperties(
                    *node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients = _mathematicalProblem->pde->properties->getLocalProperties(
                    *node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficient = _mathematicalProblem->pde->properties->getLocalProperties(
                    *node->id.global).zerothOrderCoefficient;

            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);

            for (auto &direction: directions) {
                auto directionIndex = spatialDirectionToUnsigned[direction];
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[1][direction],
                                              qualifiedPositionsAndPoints[1][direction]);
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction),
                                              templatePositionsAndPointsMap[2][direction],
                                              qualifiedPositionsAndPoints[2][direction]);

                //Scheme Weights
                auto firstDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[1][direction], direction, errorOrderDerivative1, 1);
                auto secondDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositionsAndPoints[2][direction], direction, errorOrderDerivative2, 2);

                //PDE Coefficients for the current node and direction
                auto firstDerivativeCoefficient = firstOrderCoefficients->at(directionIndex);
                auto secondDerivativeCoefficient = secondOrderCoefficients->at(directionIndex, directionIndex);

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

                auto colinearCoordinates = map<Direction, vector<vector<double>>>();
                if (dof->id->constraintType() == Free) {
                    colinearDOFDerivative1 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative1);
                    colinearDOFDerivative2 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative2);
                    colinearCoordinates = graph.getSameColinearNodalCoordinatesOnBoundary(_coordinateType,
                                                                                          nodeGraphDerivative2);
                } else {
                    colinearCoordinates = graph.getSameColinearNodalCoordinatesOnBoundary(_coordinateType,
                                                                                          nodeGraphDerivative2);
                    colinearDOFDerivative1 = graph.getColinearDOFOnBoundary(dof->type(), direction,
                                                                            nodeGraphDerivative1);
                    colinearDOFDerivative2 = graph.getColinearDOFOnBoundary(dof->type(), direction,
                                                                            nodeGraphDerivative2);
                }
                
                //TODO : calculate step for derivative 1
                //Step that is calculated based on the average absolute difference of the colinear coordinates
                auto step = VectorOperations::averageAbsoluteDifference(colinearCoordinates[direction][directionIndex]);

                auto stepDerivative2 = 1 / (pow(step, errorOrderDerivative2));

                auto positionI = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
                for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                    auto positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(
                            colinearDOFDerivative1[iDof]);
                    _totalDOFMatrix->at(positionI, positionJ) +=
                            firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient / step;
                }
                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    auto positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(
                            colinearDOFDerivative2[iDof]);
                    _totalDOFMatrix->at(positionI, positionJ) +=
                            secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient * stepDerivative2;
                }
                colinearDOFDerivative1.clear();
                colinearDOFDerivative2.clear();
            }
        }
        auto end = std::chrono::steady_clock::now(); // Stop the timer
        cout << "Total DOF Matrix Assembled in "
             << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
        //_totalDOFMatrix->print();
    }



    void AnalysisLinearSystemInitializer::_createRHS() {
        //Apply the boundary values into the RHS
        auto fixedValues = vector<double >(*_analysisDegreesOfFreedom->numberOfFixedDOFs, 0);
        for (auto & fixedDOF : *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            fixedValues[*fixedDOF->id->value] = fixedDOF->value();
        }
        
        auto dirichletContribution = _fixedFreeMatrix->multiplyWithVector(fixedValues);
        for ( auto i = 0; i < dirichletContribution.size(); i++) {
            _rhsVector->at(i) -= dirichletContribution[i];
            
        }
        delete _fixedFreeMatrix;
        _fixedFreeMatrix = nullptr;


        //print vector
/*        for (auto &value: *_rhsVector) {
            cout << value << endl;
        }*/
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


/*
//
// Created by hal9 000 on 3/28/23.
//

#include "AnalysisLinearSystemInitializer.h"

namespace LinearAlgebra {
    
    
    
    AnalysisLinearSystemInitializer::AnalysisLinearSystemInitializer( AnalysisDegreesOfFreedom* analysisDegreesOfFreedom,
                                                                      Mesh* mesh, MathematicalProblem* problem, FDSchemeSpecs* specs) :
            _analysisDegreesOfFreedom(analysisDegreesOfFreedom), _mesh(mesh), _mathematicalProblem(problem), _specs(specs) {
        
        linearSystem = nullptr;
        _freeFreeMatrix = new Array<double>(*analysisDegreesOfFreedom->numberOfFreeDOFs, *analysisDegreesOfFreedom->numberOfFreeDOFs, 1, 0);
        _rhsVector = new vector<double>(*analysisDegreesOfFreedom->numberOfFreeDOFs, 0);
        _fixedDOFCoefficients = new map<DegreeOfFreedom*, double>();
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
    }
        
    AnalysisLinearSystemInitializer::~AnalysisLinearSystemInitializer() {
        delete linearSystem;
        _mesh = nullptr;
    }
    
    void AnalysisLinearSystemInitializer::createLinearSystem() {
        _createMatrix();
        _createRHS();
        //linearSystem = new LinearSystem(_freeFreeMatrix, _rhsVector);
        linearSystem = new LinearSystem(_freeFreeMatrix, _rhsVector);
        //linearSystem->exportToMatlabFile("firstCLaplacecplace.m", "/home/hal9000/code/BiGGMan++/Testing/", true);
    }
    
    void AnalysisLinearSystemInitializer::assembleMatrices() {
        _calculateFixedDOFCoefficients();
        _createFreeDOFSubMatrixAndRHS();
    }
    
    //Creates a matrix with consistent order across the domain. The scheme type is defined by the node neighbourhood.
    //Error Order is user defined.
    void AnalysisLinearSystemInitializer::_createFreeDOFSubMatrixAndRHS() {

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
                
                if (firstDerivativeCoefficient != 0){
                    for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                        if (colinearDOFDerivative1[iDof]->id->constraintType() == Free){
                            positionJ = *colinearDOFDerivative1[iDof]->id->value;
                            _freeFreeMatrix->at(positionI, positionJ) =
                                    _freeFreeMatrix->at(positionI, positionJ) +
                                    firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
                        }
                    }
                }

                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    if (colinearDOFDerivative2[iDof]->id->constraintType() == Free){
                        positionJ = *colinearDOFDerivative2[iDof]->id->value;
                        _freeFreeMatrix->at(positionI, positionJ) =
                                _freeFreeMatrix->at(positionI, positionJ) +
                                secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    }
                    //else
                    //1) do the same process for the fixed dof: find the scheme for the bounded dof and subtract from rhs
                    //   the value of the bounded dof multiplied by the scheme weight of the corresponding dof
                }
*/
/*                cout<<"row : "<<positionI<<"  "<<endl;
                //_freeFreeMatrix->printRow(positionI);
                cout<<"  "<<endl;*//*

            }
        }
        cout << "Free DOF matrix" << endl;
            //_freeFreeMatrix->print();
        cout << "  " << endl;
    }

*/
/*    void AnalysisLinearSystemInitializer::_createTotalDOFMatrix() {

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
        for (auto &dofTuple: *_analysisDegreesOfFreedom->totalDegreesOfFreedomMap) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto dof = dofTuple.second;
            auto node = _mesh->nodeFromID(*dof->parentNode);

            auto secondOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;

            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);

            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);

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
                cout<<*dof->parentNode<<endl;
            }
            qualifiedPositionsAndPoints.clear();
            availablePositionsAndDepth.clear();
        }
        cout << "Total DOF matrix" << endl;
        _totalDOFMatrix->print();
        cout << "  " << endl;
    }*//*


    void AnalysisLinearSystemInitializer::_calculateFixedDOFCoefficients() {


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

        
        unsigned examinedDofIndex = 0;
        unsigned iDOFIndex = 0;
        vector<double> iFixedDofRow(_analysisDegreesOfFreedom->totalDegreesOfFreedomMap->size(), 0);

        //Iterate over the free degrees of freedom
        for (auto &dofTuple: *_analysisDegreesOfFreedom->totalDegreesOfFreedomMap) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto dof = dofTuple.second;
            examinedDofIndex = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
            _fixedDOFCoefficients->insert(pair<DegreeOfFreedom*, double>( dof, 0.0));


            auto node = _mesh->nodeFromID(*dof->parentNode);

            auto secondOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;

            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);

            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);

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
                
                vector<DegreeOfFreedom *> colinearDOFDerivative1, colinearDOFDerivative2;

                colinearDOFDerivative1 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative1);
                colinearDOFDerivative2 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative2);
                
                //Calculate the fucking scheme
*/
/*                for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                    iDOFIndex = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative1[iDof]);
                    iFixedDofRow[iDOFIndex] += firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
*//*
*/
/*                    if (colinearDOFDerivative1[iDof] == dof)
                        examinedDofIndex = iDOFIndex;*//*
*/
/*
                }*//*

                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    iDOFIndex = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative2[iDof]);
                    iFixedDofRow[iDOFIndex] += secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    if (iDOFIndex == examinedDofIndex){
                        _fixedDOFCoefficients->at(dof) = _fixedDOFCoefficients->at(dof) +
                                                             secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    }
                }
                auto lol = 0;

            }
            cout << iFixedDofRow[examinedDofIndex] << " " <<_fixedDOFCoefficients->at(dof) << endl;
            iDOFIndex = 0;
            examinedDofIndex = 0;
            qualifiedPositionsAndPoints.clear();
            availablePositionsAndPoints.clear();
            for (auto &i : iFixedDofRow)
                i = 0;
        }
        cout << "Fixed DOF matrix" << endl;
        //_fixedFreeMatrix->print();
        cout << "  " << endl;
    }
    
    void AnalysisLinearSystemInitializer::_createRHS() {
        // Initialize the RHS with zeros
        _rhsVector = new vector<double>(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size(), 0);
        
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);
            
            //Get all the neighbouring DOFs with the same type
            auto dofGraph = IsoParametricNodeGraph(node, 2, _parametricCoordToNodeMap, _mesh->nodesPerDirection).
                                       getSpecificDOFGraph(dof->type());
            //Marching through all the neighbouring DOFs
            for (auto &neighbour: dofGraph) {

                for (auto &neighbourDof: neighbour.second) {
                    //Check if the neighbouring DOF is fixed 
                    if (neighbourDof->id->constraintType() == Fixed) {
                        _rhsVector->at(*dof->id->value) -= _fixedDOFCoefficients->at(neighbourDof) * neighbourDof->value();
                    }
                }
            }            
        }

        
        //print vector
        for (auto &value: *_rhsVector) {
            //cout << value << endl;
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

*/
/*    void AnalysisLinearSystemInitializer::_checkIfAvailableAreQualified(
            map<vector<Position>, unsigned short>& availablePositionsAndPoints,
            map<vector<Position>, short>& templatePositionsAndPoints,
            map<vector<Position>, short>& qualifiedPositionsAndPoints)
    {
        // Check if the specifications of the template positions and points are met in the available positions and points
        for (auto& templatePositionAndPoints : templatePositionsAndPoints) {
            auto it = availablePositionsAndPoints.find(templatePositionAndPoints.first);
            if (it != availablePositionsAndPoints.end() && it->second >= templatePositionAndPoints.second) {
                qualifiedPositionsAndPoints.emplace(templatePositionAndPoints.first, templatePositionAndPoints.second);
            }
        }
    }*//*

    void AnalysisLinearSystemInitializer::_checkIfAvailableAreQualified(
            map<vector<Position>, unsigned short>& availablePositionsAndPoints,
            map<vector<Position>, short>& templatePositionsAndPoints,
            map<vector<Position>, short>& qualifiedPositionsAndPoints)
    {
        for (const auto& templatePositionAndPoints : templatePositionsAndPoints) {
            const auto& templatePosition = templatePositionAndPoints.first;
            short requiredPoints = templatePositionAndPoints.second;

            auto availableIt = availablePositionsAndPoints.find(templatePosition);
            if (availableIt != availablePositionsAndPoints.end() && availableIt->second >= requiredPoints) {
                qualifiedPositionsAndPoints[templatePosition] = requiredPoints;
            }
        }
    }







}// LinearAlgebra*/
