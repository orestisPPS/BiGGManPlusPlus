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
        _rhsVector = new vector<double>(*_analysisDegreesOfFreedom->numberOfFreeDOFs, 0);
        _freeFreeFreeFixedSubMatrix = new Array<double>(*_analysisDegreesOfFreedom->numberOfFreeDOFs, *_analysisDegreesOfFreedom->numberOfDOFs, 1, 0);
        _freeFreeMatrix = new Array<double>(*_analysisDegreesOfFreedom->numberOfFreeDOFs, *_analysisDegreesOfFreedom->numberOfFreeDOFs, 1, 0);
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
        _createFreeFreeFreeFixedSubMatrix();
        _createFreeFreeDOFSubMatrix();
    }

    //Creates a matrix with consistent order across the domain. The scheme type is defined by the node neighbourhood.
    //Error Order is user defined.

    void AnalysisLinearSystemInitializer::_createFreeFreeDOFSubMatrix() {
        auto start = std::chrono::steady_clock::now(); // Start the timer

        auto &totalDOF = *_analysisDegreesOfFreedom->numberOfFreeDOFs;
        for (unsigned i = 0; i < totalDOF; i++) {
            for (unsigned j = 0; j < totalDOF; j++) {
                _freeFreeMatrix->at(i, j) = _freeFreeFreeFixedSubMatrix->at(i, j);
            }
        }

        auto end = std::chrono::steady_clock::now(); // Stop the timer
        
        cout << "Free - Free DOF Sub-Matrix Assembled in : " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
        _freeFreeMatrix->print();
    }
    
    void AnalysisLinearSystemInitializer::_createFreeFreeFreeFixedSubMatrix() {
        auto start = std::chrono::steady_clock::now(); // Start the timer

        auto directions = _mesh->directions();
        short unsigned maxDerivativeOrder = 2;
        auto templatePositionsAndPointsMap = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto schemeBuilder = FiniteDifferenceSchemeBuilder(_specs);

        auto errorOrderDerivative1 = _specs->getErrorOrderOfSchemeTypeForDerivative(1);
        auto errorOrderDerivative2 = _specs->getErrorOrderOfSchemeTypeForDerivative(2);

        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);
        schemeBuilder.templatePositionsAndPoints(2, errorOrderDerivative2, directions,
                                                 templatePositionsAndPointsMap[2]);

        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();
        
        //Iterate over all the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //Define the position of the dof in the Matrix
            auto i = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
            
            //Define the node where the dof belongs
            auto node = _mesh->nodeFromID(dof->parentNode());
            
            //Define the PDE coefficients for the current node and for every derivative order.
            auto secondOrderCoefficients = _mathematicalProblem->pde->properties->getLocalProperties(
                    *node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients = _mathematicalProblem->pde->properties->getLocalProperties(
                    *node->id.global).firstOrderCoefficients;
            
            //Find the node neighbours with a span equal to the maximum number of points needed for the scheme to be consistent
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            //Derivative order 0
            auto zerothOrderCoefficient = *_mathematicalProblem->pde->properties->getLocalProperties(
                    *node->id.global).zerothOrderCoefficient;
            _freeFreeFreeFixedSubMatrix->at(i, i) =
                    _freeFreeFreeFixedSubMatrix->at(i, i) + zerothOrderCoefficient;
            
            
            //March through all the non-zero derivative orders 
            for (auto derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++){
                
                //Decompose scheme into directional components
                for (auto &direction: directions) {

                    auto directionIndex = spatialDirectionToUnsigned[direction];
                    
                    //Check if the available positions are qualified for the current derivative order
                    auto qualifiedPositions = _getQualifiedFromAvailable(availablePositionsAndDepth[direction],templatePositionsAndPointsMap[derivativeOrder][direction]);
                    auto schemeWeights = schemeBuilder.getSchemeWeightsFromQualifiedPositions(
                            qualifiedPositions, direction,_specs->getErrorOrderOfSchemeTypeForDerivative(derivativeOrder), derivativeOrder);

                    auto graphFilter = map<Position, unsigned short>();
                    for (auto &tuple: qualifiedPositions) {
                        for (auto &point: tuple.first) {
                            graphFilter.insert(pair<Position, unsigned short>(point, tuple.second));
                        }
                    }
                    auto filteredNodeGraph = graph.getNodeGraph(graphFilter);

                    auto colinearCoordinates = map<Direction, vector<vector<double>>>();
                    auto colinearDOF = vector<DegreeOfFreedom *>();
                    
                    if (dof->constraintType() == Free) {
                        colinearCoordinates = graph.getSameColinearNodalCoordinates(_coordinateType, filteredNodeGraph);
                        colinearDOF = graph.getColinearDOF(dof->type(), direction, filteredNodeGraph);
                    }
                    else if (dof->constraintType() == Fixed) {
                        colinearCoordinates = graph.getSameColinearNodalCoordinatesOnBoundary(_coordinateType,filteredNodeGraph);
                        colinearDOF = graph.getColinearDOFOnBoundary(dof->type(), direction, filteredNodeGraph);
                    }

                    auto step = VectorOperations::averageAbsoluteDifference(colinearCoordinates[direction][directionIndex]);

                    auto pdeCoefficient = 1.0;
                    for (int iDof = 0; iDof < colinearDOF.size(); ++iDof) {
                        unsigned j = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOF[iDof]);
                        _freeFreeFreeFixedSubMatrix->at(i, j) = _freeFreeFreeFixedSubMatrix->at(i, j) +
                                                                schemeWeights[iDof] * pdeCoefficient / step;
                    }
                }
            }
        }
        auto end = std::chrono::steady_clock::now(); // Stop the timer
        cout << "Total DOF Matrix Assembled in "
             << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
        _freeFreeFreeFixedSubMatrix->print();
    }



    void AnalysisLinearSystemInitializer::_createRHS() {
        for(unsigned iFreeDOF = 0; iFreeDOF < *_analysisDegreesOfFreedom->numberOfFreeDOFs; iFreeDOF++){
            for (unsigned jNeighbour = 0; jNeighbour < *_analysisDegreesOfFreedom->numberOfDOFs; jNeighbour++) {
                auto neighbourDOF = _analysisDegreesOfFreedom->totalDegreesOfFreedomMap->at(jNeighbour);
                if (neighbourDOF->constraintType() == Fixed){
                    _rhsVector->at(iFreeDOF) = _rhsVector->at(iFreeDOF) - _freeFreeFreeFixedSubMatrix->at(iFreeDOF, jNeighbour) * neighbourDOF->value();
                }
            }
        }
        cout << "RHS Vector Assembled" << endl;
        delete _freeFreeFreeFixedSubMatrix;
        

/*        //print vector
        for (auto &value: *_rhsVector) {
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

    map<vector<Position>,short> AnalysisLinearSystemInitializer::
    _getQualifiedFromAvailable(map<vector<Position>,unsigned short>& availablePositionsAndPoints,
                               map<vector<Position>,short>& templatePositionsAndPoints){
        map<vector<Position>,short> qualifiedPositionsAndPoints = map<vector<Position>,short>();
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
        return qualifiedPositionsAndPoints;
    }

}// LinearAlgebra
