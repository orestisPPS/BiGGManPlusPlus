//
// Created by hal9 000 on 3/28/23.
//

#ifndef UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H
#include "AnalysisLinearSystemInitializer.h"

#include <memory>
#include <utility>

namespace LinearAlgebra {
    
    AnalysisLinearSystemInitializer::
    AnalysisLinearSystemInitializer(shared_ptr<AnalysisDegreesOfFreedom> analysisDegreesOfFreedom,
                                    const shared_ptr<Mesh> &mesh,
                                    shared_ptr<MathematicalProblem> problem,
                                    shared_ptr<FDSchemeSpecs> specs,
                                    CoordinateType coordinateType) :
                                    linearSystem(nullptr),
                                    _analysisDegreesOfFreedom(std::move(analysisDegreesOfFreedom)),
                                    _mesh(mesh),
                                    _mathematicalProblem(std::move(problem)),
                                    _specs(std::move(specs)),
                                    _coordinateType(coordinateType) {
        _rhsVector = make_shared<NumericalVector<double>>(*_analysisDegreesOfFreedom->numberOfFreeDOF, 0);
        _matrix = make_shared<NumericalMatrix<double>>(*_analysisDegreesOfFreedom->numberOfFreeDOF, *_analysisDegreesOfFreedom->numberOfFreeDOF);
        _parametricCoordToNodeMap = _mesh->getCoordinatesToNodesMap(Parametric);
    }
    

    void AnalysisLinearSystemInitializer::createLinearSystem() {
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
        _matrix->dataStorage->initializeElementAssignment();
        for (auto &dof: *_analysisDegreesOfFreedom->internalDegreesOfFreedom) {

            //Define the node where the dof belongs
            auto node = _mesh->nodeFromID(dof->parentNode());
            auto thisDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);

            //Find the node neighbours with a span equal to the maximum number of points needed for the scheme to be consistent
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection, false);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);

            //Derivative order 0
            auto zeroOrderCoefficient = _getPDECoefficient(0, node);
            //Define the position of the dof in the NumericalMatrix
            _matrix->setElement(thisDOFPosition, thisDOFPosition, zeroOrderCoefficient);

            //add source term
            _rhsVector->at(thisDOFPosition) += *_mathematicalProblem->pde->properties->getLocalProperties(
                    dof->parentNode()).sourceTerm;

            //March through all the non-zero derivative orders 
            for (auto derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++) {

                //Decompose scheme into directional components
                for (auto &directionI: directions) {
                    unsigned indexDirectionI = spatialDirectionToUnsigned[directionI];
                    double iThDerivativePDECoefficient = _getPDECoefficient(derivativeOrder, node, directionI);
                    if (iThDerivativePDECoefficient != 0) {
                        
                        //Check if the available positions are qualified for the current derivative order
                        auto qualifiedPositions = _getQualifiedFromAvailable(
                                availablePositionsAndDepth[directionI], templatePositionsAndPointsMap[derivativeOrder][directionI]);
                        //auto scheme = FiniteDifferenceSchemeBuilder::getSchemeWeightsFromQualifiedPositions(
                                //qualifiedPositions, directionI, errorOrderDerivative1, 1);

                        auto graphFilter = map<Position, unsigned short>();
                        for (auto &tuple: qualifiedPositions) {
                            for (auto &point: tuple.first) {
                                graphFilter.insert(pair<Position, unsigned short>(point, tuple.second));
                            }
                        }
                        auto filteredNodeGraph = graph.getNodeGraph(graphFilter);
                        auto colinearCoordinates = graph.getSameColinearNodalCoordinates(_coordinateType, filteredNodeGraph);
                        auto colinearDOF = graph.getColinearDOFOnBoundary(dof->type(), directionI, filteredNodeGraph);
                        
                        auto taylorPoint = (*node->coordinates.getPositionVector(Parametric))[indexDirectionI];
                        auto weights = calculateWeightsOfDerivativeOrder(*colinearCoordinates[directionI]->getVectorSharedPtr(),
                                                                         derivativeOrder, taylorPoint);

                            //NumericalVector<double> &schemeWeights = scheme.weights;
                            for (int iDof = 0; iDof < colinearDOF.size(); ++iDof) {
                                auto neighbourDOF = colinearDOF[iDof];
                                //auto weight = schemeWeights[iDof] * iThDerivativePDECoefficient / denominator;
                                auto weight2 = weights[iDof] * iThDerivativePDECoefficient;
                                if (neighbourDOF->constraintType() == Free) {
                                    auto neighbourDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(
                                            colinearDOF[iDof]);
  /*                                  double ijElement = _matrix->getElement(thisDOFPosition, neighbourDOFPosition);
                                    if (ijElement != 0)
                                        ijElement += weight2;
                                    else
                                        _matrix->setElement(thisDOFPosition, neighbourDOFPosition, weight2);*/
                                    double ijElement = _matrix->getElement(thisDOFPosition, neighbourDOFPosition);
                                    ijElement += weight2;
                                    _matrix->setElement(thisDOFPosition, neighbourDOFPosition, ijElement);

                                }
                                else if (neighbourDOF->constraintType() == Fixed) {
                                    auto dirichletContribution = neighbourDOF->value() * weight2;
                                    _rhsVector->at(thisDOFPosition) -= dirichletContribution;
                                }
                            }
                        }
                    }
            }
        }
        //addNeumannBoundaryConditions();
        
        this->linearSystem = make_shared<LinearSystem>(std::move(_matrix), std::move(_rhsVector));


        auto end = std::chrono::steady_clock::now(); // Stop the timer
        cout << "Linear System Assembled in "
             << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
    }

    //TODO : Fix as above
    void AnalysisLinearSystemInitializer::addNeumannBoundaryConditions() {
        /*auto start = std::chrono::steady_clock::now(); // Start the timer
        auto schemeSpecs = make_shared<FDSchemeSpecs>(2, _mesh->directions());
        auto directions = _mesh->directions();
        short unsigned maxDerivativeOrder = 1;

        auto schemeBuilder = FiniteDifferenceSchemeBuilder(schemeSpecs);
        auto errorOrderDerivative1 = 2;
        map<short unsigned, map<Direction, map<vector<Position>, short>>> templatePositionsAndPointsMap = schemeBuilder.initiatePositionsAndPointsMap(
                maxDerivativeOrder, directions);
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();
        auto boundaryNodeToPositionMap = _mesh->getBoundaryNodeToPositionMap();

        for (auto &dof: *_analysisDegreesOfFreedom->fluxDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(dof.first->parentNode());
            auto normalVector = _mesh->getNormalUnitVectorOfBoundaryNode(boundaryNodeToPositionMap->at(node), node);
            auto thisDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof.first);
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap, _mesh->nodesPerDirection,
                                                false);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            auto nodeMetrics = make_shared<Metrics>(node, _mesh->dimensions());
            //Loop through all the directions to find g_i = d(x_j)/d(x_i), g^i = d(x_i)/d(x_j)
            for (auto &directionI: directions) {
                
                auto i = spatialDirectionToUnsigned[directionI];
                //Check if the available positions are qualified for the current derivative order
                auto qualifiedPositions = schemeBuilder.getQualifiedFromAvailable(
                        availablePositionsAndDepth[directionI], templatePositionsAndPointsMap[1][directionI]);
                auto scheme = FiniteDifferenceSchemeBuilder::getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositions, directionI, errorOrderDerivative1, 1);

                auto graphFilter = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositions) {
                    for (auto &point: tuple.first) {
                        graphFilter.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto filteredNodeGraph = graph.getNodeGraph(graphFilter);
                auto colinearCoordinatesNumericalVector = graph.getSameColinearNodalCoordinatesOnBoundary(
                        _coordinateType, filteredNodeGraph)[directionI][i];
                auto colinearDOF = graph.getColinearDOFOnBoundary(dof.first->type(), directionI, filteredNodeGraph);
                double schemeAroundPoint = (*colinearCoordinatesNumericalVector)[0];
                auto weights2 = calculateWeightsOfDerivativeOrder(*colinearCoordinatesNumericalVector, 2, schemeAroundPoint);
   
                auto step = colinearCoordinatesNumericalVector->averageAbsoluteDeviationFromMean();

                //Calculate the denominator (h^p)
                double denominator = scheme.denominatorCoefficient * pow(step, scheme.power);
                NumericalVector<double> &schemeWeights = scheme.weights;
                for (int iDof = 0; iDof < colinearDOF.size(); ++iDof) {
                    auto neighbourDOF = colinearDOF[iDof];
                    auto weight = schemeWeights[iDof] / denominator;
                    auto weight2 = weights2[iDof];

                    if (neighbourDOF->constraintType() == Free) {
                        auto neighbourDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(
                                colinearDOF[iDof]);
                        //_matrix->at(thisDOFPosition, neighbourDOFPosition) += weight2 * normalVector[i];
                        double &ijElement = _matrix->getElement(thisDOFPosition, neighbourDOFPosition);
                        if (ijElement != 0)
                            ijElement += weight2 * normalVector[i];
                        else
                            _matrix->setElement(thisDOFPosition, neighbourDOFPosition, weight2 * normalVector[i]);
                    } else if (neighbourDOF->constraintType() == Fixed) {
                        auto dirichletContribution = neighbourDOF->value() * weight2 * normalVector[i];
                        _rhsVector->at(thisDOFPosition) -= dirichletContribution;
                    }
                }
            }
            _rhsVector->at(thisDOFPosition) += dof.second;
        }*/
    }

    map<short unsigned, map<Direction, map<vector<Position>, short>>> AnalysisLinearSystemInitializer::
    _initiatePositionsAndPointsMap(short unsigned &maxDerivativeOrder, vector<Direction> &directions) {
        map<short unsigned, map<Direction, map<vector<Position>, short>>> positionsAndPoints;
        for (short unsigned derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++) {
            positionsAndPoints.insert(pair<short unsigned, map<Direction, map<vector<Position>, short>>>(
                    derivativeOrder, map<Direction, map<vector<Position>, short>>()));
            for (auto &direction: directions) {
                positionsAndPoints[derivativeOrder].insert(pair<Direction, map<vector<Position>, short>>(
                        direction, map<vector<Position>, short>()));
            }
        }
        return positionsAndPoints;
    }

    map<vector<Position>, short> AnalysisLinearSystemInitializer::
    _getQualifiedFromAvailable(map<vector<Position>, unsigned short> &availablePositionsAndPoints,
                               map<vector<Position>, short> &templatePositionsAndPoints) {
        map<vector<Position>, short> qualifiedPositionsAndPoints = map<vector<Position>, short>();
        //Check if the specifications of the template positions and points are met in the available positions and points
        for (auto &templatePositionAndPoints: templatePositionsAndPoints) {
            for (auto &availablePositionAndPoints: availablePositionsAndPoints) {
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

    double AnalysisLinearSystemInitializer::_getPDECoefficient(unsigned short derivativeOrder, Node *parentNode,
                                                               Direction direction) {
        auto directionIndex = spatialDirectionToUnsigned[direction];
        auto properties = _mathematicalProblem->pde->properties->getLocalProperties(*parentNode->id.global);
        switch (derivativeOrder) {
            case 0:
                return *properties.zerothOrderCoefficient;
            case 1:
                return properties.firstOrderCoefficients->at(directionIndex);
            case 2:
                return properties.secondOrderCoefficients->getElement(directionIndex, directionIndex);
            default:
                throw runtime_error("Derivative order should be 0, 1 or 2");
        }
    }

}; // LinearAlgebra

#endif //UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H