/*
//
// Created by hal9000 on 6/2/23.
//

#include "AnalysisLinearSystemInitializer2.h"


namespace LinearAlgebra {



    AnalysisLinearSystemInitializer2::AnalysisLinearSystemInitializer2(shared_ptr<AnalysisDegreesOfFreedom> analysisDegreesOfFreedom, shared_ptr<Mesh> mesh,
                                                                     shared_ptr<MathematicalProblem> problem, shared_ptr<FDSchemeSpecs> specs) {
        linearSystem = nullptr;
        this->_analysisDegreesOfFreedom = analysisDegreesOfFreedom;
        this->_mesh = mesh;
        this->_mathematicalProblem = problem;
        numberOfFreeDOFs = new unsigned(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size());
        numberOfFixedDOFs = new unsigned(_analysisDegreesOfFreedom->fixedDegreesOfFreedom->size());
        numberOfDOFs = new unsigned(_analysisDegreesOfFreedom->totalDegreesOfFreedomMap->size());
        _matrix = new Array<double>(*numberOfFreeDOFs, *numberOfFreeDOFs, 1, 0);
        _fixedFreeMatrix = new Array<double>(*numberOfFixedDOFs, *numberOfDOFs, 1, 0);
        _freeFreeFreeFixedSubMatrix = new Array<double>(*numberOfDOFs, *numberOfDOFs, 1, 0);
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
        _specs = specs;
    }

    AnalysisLinearSystemInitializer2::~AnalysisLinearSystemInitializer2() {
        delete linearSystem;
        delete numberOfFreeDOFs;
        delete numberOfFixedDOFs;
        delete numberOfDOFs;
        _mesh = nullptr;
    }

    void AnalysisLinearSystemInitializer2::createLinearSystem() {
        _createMatrix();
        _createRHS();
        //linearSystem = new LinearSystem(_matrix, _rhsVector);
        linearSystem = new LinearSystem(_matrix, _rhsVector);
        //linearSystem->exportToMatlabFile("firstCLaplacecplace.m", "/home/hal9000/code/BiGGMan++/Testing/", true);
    }

    void AnalysisLinearSystemInitializer2::_assembleMatrices() {
        //_createFreeFixedDOFSubMatrix();
        _createFreeFreeDOFSubMatrix();
        _createFreeFreeFreeFixedSubMatrix();
    }
    
*/
/*    void AnalysisLinearSystemInitializer2::_createPermutationMatrix() {
        for (auto &freeDOF : *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto i = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(freeDOF);

            auto j = freeDOF->index;
        }
*//*
*/
/*        unsigned int& totalDOF = *_analysisDegreesOfFreedom->numberOfDOFs;
        for (auto i = 0; i < totalDOF; i++) {
            for (auto j = 0; j < totalDOF; j++) {
                if (i == j) {
                    _matrix->setElement(i, j, 1);
                }
            }
        }

        }*//*
*/
/*
    }*//*


    //Creates a matrix with consistent order across the domain. The scheme type is defined by the node neighbourhood.
    //Error Order is user defined.
    void AnalysisLinearSystemInitializer2::_createFreeFreeDOFSubMatrix() {

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
        auto errorOrderDerivative2 = _specs->getErrorOrderOfSchemeTypeForDerivative(2);


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
                _getQualifiedFromAvailable(availablePositionsAndDepth.at(direction),
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
                            _matrix->at(positionI, positionJ) =
                                    _matrix->at(positionI, positionJ) +
                                    firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
                        }
                    }
                }

                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    if (colinearDOFDerivative2[iDof]->id->constraintType() == Free){
                        positionJ = *colinearDOFDerivative2[iDof]->id->value;
                        _matrix->at(positionI, positionJ) =
                                _matrix->at(positionI, positionJ) +
                                secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    }
                }
*/
/*                cout<<"row : "<<positionI<<"  "<<endl;
                //_matrix->printRow(positionI);
                cout<<"  "<<endl;*//*

            }
        }
        cout << "Free DOF matrix" << endl;
        _matrix->print();
        cout << "  " << endl;
    }

    void AnalysisLinearSystemInitializer2::_createFreeFixedDOFSubMatrix() {

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

            positionI = *dof->id->value;

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
                //Jusqu'ici, tout va bien
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
                for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative1[iDof]);
                    _fixedFreeMatrix->at(positionI, positionJ) = _fixedFreeMatrix->at(positionI, positionJ) +
                                                                firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
                    bool lol = 0;
                }
                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative2[iDof]);
                    _fixedFreeMatrix->at(positionI, positionJ) = _fixedFreeMatrix->at(positionI, positionJ) +
                                                                secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    bool lol = false;

                }
            }
            qualifiedPositionsAndPoints.clear();
            availablePositionsAndDepth.clear();
        }
        cout << "Fixed DOF matrix" << endl;
        _fixedFreeMatrix->print();
        cout << "  " << endl;
    }

    void AnalysisLinearSystemInitializer2::_createFreeFreeFreeFixedSubMatrix() {
        auto directions = _mesh->directions();
        short unsigned maxDerivativeOrder = 2;
        auto templatePositionsAndPointsMap = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto qualifiedPositionsAndPoints = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto schemeBuilder = FiniteDifferenceSchemeBuilder(_specs);

        auto errorOrderDerivative1 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(1);
        auto errorOrderDerivative2 = _specs->getErrorOrderOfSchemeTypeForDerivative(2);

        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions, templatePositionsAndPointsMap[1]);
        schemeBuilder.templatePositionsAndPoints(2, errorOrderDerivative2, directions, templatePositionsAndPointsMap[2]);

        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        for (auto &dof : *_analysisDegreesOfFreedom->totalDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);
            auto secondOrderCoefficients = _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients = _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficient = _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;

            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap, _mesh->nodesPerDirection);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);

            for (auto &direction : directions) {
                _checkIfAvailableAreQualified(availablePositionsAndDepth.at(direction), templatePositionsAndPointsMap[1][direction], qualifiedPositionsAndPoints[1][direction]);
                _getQualifiedFromAvailable(availablePositionsAndDepth.at(direction), templatePositionsAndPointsMap[2][direction], qualifiedPositionsAndPoints[2][direction]);

                auto firstDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(qualifiedPositionsAndPoints[1][direction], direction, errorOrderDerivative1, 1);
                auto secondDerivativeSchemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(qualifiedPositionsAndPoints[2][direction], direction, errorOrderDerivative2, 2);

                auto directionIndex = spatialDirectionToUnsigned[direction];
                auto firstDerivativeCoefficient = firstOrderCoefficients->at(directionIndex);
                auto secondDerivativeCoefficient = secondOrderCoefficients->at(directionIndex, directionIndex);

                auto filterDerivative1 = map<Position, unsigned short>();
                for (auto &tuple : qualifiedPositionsAndPoints[1][direction]) {
                    for (auto &point : tuple.first) {
                        filterDerivative1.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto filterDerivative2 = map<Position, unsigned short>();
                for (auto &tuple : qualifiedPositionsAndPoints[2][direction]) {
                    for (auto &point : tuple.first) {
                        filterDerivative2.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }

                auto nodeGraphDerivative1 = graph.getNodeGraph(filterDerivative1);
                auto nodeGraphDerivative2 = graph.getNodeGraph(filterDerivative2);

                vector<DegreeOfFreedom*> colinearDOFDerivative1, colinearDOFDerivative2;

                if (dof->id->constraintType() == Free) {
                    colinearDOFDerivative1 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative1);
                    colinearDOFDerivative2 = graph.getColinearDOF(dof->type(), direction, nodeGraphDerivative2);
                } else {
                    colinearDOFDerivative1 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative1);
                    colinearDOFDerivative2 = graph.getColinearDOFOnBoundary(dof->type(), direction, nodeGraphDerivative2);
                }

                unsigned positionI = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
                for (auto iDof = 0; iDof < colinearDOFDerivative1.size(); iDof++) {
                    unsigned positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative1[iDof]);
                    _freeFreeFreeFixedSubMatrix->at(positionI, positionJ) += firstDerivativeSchemeWeights[iDof] * firstDerivativeCoefficient;
                }
                for (auto iDof = 0; iDof < colinearDOFDerivative2.size(); iDof++) {
                    unsigned positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOFDerivative2[iDof]);
                    _freeFreeFreeFixedSubMatrix->at(positionI, positionJ) += secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient;
                    if (positionI == positionJ)
                        cout<< "Diagonal element: " << _freeFreeFreeFixedSubMatrix->at(positionI, positionJ) <<
                            " "<< secondDerivativeSchemeWeights[iDof] * secondDerivativeCoefficient<<  endl;
                }

                colinearDOFDerivative1.clear();
                colinearDOFDerivative2.clear();
            }

            qualifiedPositionsAndPoints.clear();
            availablePositionsAndDepth.clear();
        }
        cout << "Total DOF matrix" << endl;
        _freeFreeFreeFixedSubMatrix->print();
    }


    void AnalysisLinearSystemInitializer2::_createRHS() {
        //Marching through all the free DOFs
        _rhsVector = new vector<double>(*numberOfDOFs, 0);
        _rhsVector = new vector<double>(*numberOfFreeDOFs, 0);
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);

            //Get all the neighbouring DOFs with the same type
            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->nodesPerDirection).
                            getSpecificDOFGraph(dof->type());
            //Marching through all the neighbouring DOFs
            for (auto &neighbour: dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    //Check if the neighbouring DOF is fixed 
                    if (neighbourDof->id->constraintType() == Fixed) {
                        //unsigned i = *dof->id->value;
                        unsigned i = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
                        //unsigned i = *neighbourDof->id->value;
                        unsigned j = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(neighbourDof);
                        cout<<_freeFreeFreeFixedSubMatrix->at(i, j)<<endl;
                        //_rhsVector->at(*dof->id->value) -= _fixedFreeMatrix->at(i, j) * neighbourDof->value();
                        _rhsVector->at(*dof->id->value) -= _freeFreeFreeFixedSubMatrix->at(i, j) * neighbourDof->value();
                        //_rhsVector->at(*dof->id->value) -= 1.0 * neighbourDof->value();
                    }
                }
            }
        }
        delete _fixedFreeMatrix;
        _fixedFreeMatrix = nullptr;
        delete _freeFreeFreeFixedSubMatrix;
        _freeFreeFreeFixedSubMatrix = nullptr;

        //print vector
*/
/*        for (auto &value: *_rhsVector) {
            cout << value << endl;
        }*//*

    }

    map<short unsigned, map<Direction, map<vector<Position>, short>>> AnalysisLinearSystemInitializer2::
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

    void AnalysisLinearSystemInitializer2::_getQualifiedFromAvailable(map<vector<Position>,unsigned short>& availablePositionsAndPoints,
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
*/
