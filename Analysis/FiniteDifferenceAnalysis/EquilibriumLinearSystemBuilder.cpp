//
// Created by hal9 000 on 3/28/23.
//

#ifndef UNTITLED_EQUILIBRIUMLINEARSYSTEMBUILDER_H
#include "EquilibriumLinearSystemBuilder.h"

#include <memory>
#include <utility>

namespace NumericalAnalysis {
    
    EquilibriumLinearSystemBuilder::
    EquilibriumLinearSystemBuilder(const shared_ptr<AnalysisDegreesOfFreedom>& analysisDegreesOfFreedom, const shared_ptr<Mesh> &mesh,
                                   const shared_ptr<SteadyStateMathematicalProblem>& mathematicalProblem, const shared_ptr<FDSchemeSpecs>& specs,
                                   CoordinateType coordinateSystem) :
            linearSystem(nullptr), _analysisDegreesOfFreedom(analysisDegreesOfFreedom), _mesh(mesh),
            _steadyStateMathematicalProblem(mathematicalProblem), _specs(specs) {
        
        _coordinateType = coordinateSystem;
        RHS = make_shared<NumericalVector<double>>(*_analysisDegreesOfFreedom->numberOfFreeDOF, 0);
        K = make_shared<NumericalMatrix<double>>(*_analysisDegreesOfFreedom->numberOfFreeDOF, *_analysisDegreesOfFreedom->numberOfFreeDOF);
        _parametricCoordToNodeMap = _mesh->getCoordinatesToNodesMap(Parametric);
    }
    

    void EquilibriumLinearSystemBuilder::assembleSteadyStateLinearSystem() {
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
        for (auto &dof: *_analysisDegreesOfFreedom->internalDegreesOfFreedom) {

            //Define the node where the dof belongs
            auto node = _mesh->nodeFromID(dof->parentNode());
            auto thisDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);

            //Find the node neighbours with a span equal to the maximum number of points needed for the scheme to be consistent
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection, false);
                                                        
            //This gives correct results
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);

            //Derivative order 0
            auto zeroOrderCoefficient = _steadyStateMathematicalProblem->pde->spatialDerivativesCoefficients()->getIndependentVariableTermCoefficient(node->id.global);
            if (zeroOrderCoefficient != 0)
                K->setElement(thisDOFPosition, thisDOFPosition, zeroOrderCoefficient);
            //Define the position of the dof in the NumericalMatrix

            //add source term
            RHS->at(thisDOFPosition) += _steadyStateMathematicalProblem->pde->spatialDerivativesCoefficients()->getIndependentVariableTermCoefficient(node->id.global);

            //March through all the non-zero derivative orders 
            for (auto derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++) {

                //Decompose scheme into directional components
                for (auto &directionI: directions) {
                    unsigned indexDirectionI = spatialDirectionToUnsigned[directionI];
                    
                    double iThDerivativePDECoefficient = _steadyStateMathematicalProblem->pde->spatialDerivativesCoefficients()->getDependentVariableTermCoefficient(derivativeOrder, node->id.global, directionI);
                    if (iThDerivativePDECoefficient != 0) {
                        
                        //Check if the available positions are qualified for the current derivative order
                        /*auto qualifiedPositions = _getQualifiedFromAvailable(
                                availablePositionsAndDepth[directionI], templatePositionsAndPointsMap[derivativeOrder][directionI]);*/
                        auto qualifiedPositions = _getQualifiedFromAvailable(
                                availablePositionsAndDepth[directionI], templatePositionsAndPointsMap[derivativeOrder][directionI]);
                        //auto scheme = FiniteDifferenceSchemeBuilder::getSchemeWeightsFromQualifiedPositions(
                                //qualifiedPositions, directionI, errorOrderDerivative1, 1);

                        auto graphFilter = map<Position, unsigned short>();
                        for (auto &position: get<0>(qualifiedPositions)) {
                            graphFilter.insert(pair<Position, unsigned short>(position, get<1>(qualifiedPositions)));
                        }
                        auto filteredNodeGraph = graph.getNodeGraph(graphFilter);
                        auto colinearCoordinates = graph.getSameColinearNodalCoordinates(_coordinateType, filteredNodeGraph);
                        auto colinearDOF = graph.getColinearDOF(dof->type(), directionI, filteredNodeGraph);
                        
                        auto taylorPoint = (*node->coordinates.getPositionVector(_coordinateType))[indexDirectionI];
                        auto weights = calculateWeightsOfDerivativeOrder(*colinearCoordinates[directionI]->getVectorSharedPtr(),
                                                                         derivativeOrder, taylorPoint);

                        //NumericalVector<double> &schemeWeights = scheme.weights;
                        for (int iDof = 0; iDof < colinearDOF.size(); ++iDof) {
                            auto neighbourDOF = colinearDOF[iDof];
                            //auto weight = schemeWeights[iDof] * iThDerivativePDECoefficient / denominator;
                            auto weight2 = weights[iDof] * iThDerivativePDECoefficient;
                            if (neighbourDOF->constraintType() == Free) {
                                auto neighbourDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOF[iDof]);
                                double ijElement = K->getElement(thisDOFPosition, neighbourDOFPosition);
                                ijElement += weight2;
                                K->setElement(thisDOFPosition, neighbourDOFPosition, ijElement);
                            }
                            else if (neighbourDOF->constraintType() == Fixed) {
                                auto dirichletContribution = neighbourDOF->value() * weight2;
                                RHS->at(thisDOFPosition) -=  dirichletContribution;
                            }
                        }
                    }
                }
            }
        }
        _addNeumannBoundaryConditions();
        //K->dataStorage->finalizeElementAssignment();

        /*auto diagonal = NumericalVector<double>(K->numberOfRows());
        for (unsigned i = 0; i < K->numberOfRows(); ++i) {
            diagonal[i] = K->getElement(i, i);
        }
        diagonal.printVerticallyWithIndex("diagonal");*/
    }

    //TODO : Fix as above
    void EquilibriumLinearSystemBuilder:: _addNeumannBoundaryConditions(){
        auto start = std::chrono::steady_clock::now(); // Start the timer

        auto directions = _mesh->directions();
        short unsigned maxDerivativeOrder = 2;
        auto templatePositionsAndPointsMap = _initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        auto schemeBuilder = FiniteDifferenceSchemeBuilder(_specs);

        auto errorOrderDerivative1 = _specs->getErrorOrderOfSchemeTypeForDerivative(1);

        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);


        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        //Iterate over all the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->fluxDegreesOfFreedom) {
            
            //Define the node where the dof belongs
            auto node = _mesh->nodeFromID(dof.first->parentNode());
            auto thisDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof.first);
            auto boundaryNodeToPositionMap = _mesh->getBoundaryNodeToPositionMap();
            auto normalVector = _mesh->getNormalUnitVectorOfBoundaryNode(boundaryNodeToPositionMap->at(node), node);
            //Find the node neighbours with a span equal to the maximum number of points needed for the scheme to be consistent
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection, false);

            //This gives correct results
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            
            //March through all the non-zero derivative orders 
            for (auto derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++) {

                //Decompose scheme into directional components
                for (auto &directionI: directions) {
                    unsigned indexDirectionI = spatialDirectionToUnsigned[directionI];

                    //Check if the available positions are qualified for the current derivative order
                    /*auto qualifiedPositions = _getQualifiedFromAvailable(
                            availablePositionsAndDepth[directionI], templatePositionsAndPointsMap[derivativeOrder][directionI]);*/
                    auto qualifiedPositions = _getQualifiedFromAvailable(
                            availablePositionsAndDepth[directionI],
                            templatePositionsAndPointsMap[derivativeOrder][directionI]);
                    //auto scheme = FiniteDifferenceSchemeBuilder::getSchemeWeightsFromQualifiedPositions(
                    //qualifiedPositions, directionI, errorOrderDerivative1, 1);

                    auto graphFilter = map<Position, unsigned short>();
                    for (auto &position: get<0>(qualifiedPositions)) {
                        graphFilter.insert(pair<Position, unsigned short>(position, get<1>(qualifiedPositions)));
                    }
                    auto filteredNodeGraph = graph.getNodeGraph(graphFilter);
                    auto colinearCoordinates = graph.getSameColinearNodalCoordinatesOnBoundary(_coordinateType,
                                                                                     filteredNodeGraph);
                    auto colinearDOF = graph.getColinearDOFOnBoundary(dof.first->type(), directionI, filteredNodeGraph);

                    auto taylorPoint = (*node->coordinates.getPositionVector(_coordinateType))[indexDirectionI];
                    auto weights = calculateWeightsOfDerivativeOrder(
                            *colinearCoordinates[directionI]->getVectorSharedPtr(),
                            derivativeOrder, taylorPoint);

                    //NumericalVector<double> &schemeWeights = scheme.weights;
                    for (int iDof = 0; iDof < colinearDOF.size(); ++iDof) {
                        auto neighbourDOF = colinearDOF[iDof];
                        auto weight2 = weights[iDof];

                        if (neighbourDOF->constraintType() == Free) {
                            auto neighbourDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(
                                    neighbourDOF);
                            double ijElement = K->getElement(thisDOFPosition, neighbourDOFPosition);
                            ijElement += weight2 * normalVector[indexDirectionI];
                            K->setElement(thisDOFPosition, neighbourDOFPosition, ijElement);
                        } else if (neighbourDOF->constraintType() == Fixed) {
                            auto dirichletContribution =
                                    neighbourDOF->value() * weight2 * normalVector[indexDirectionI];
                            (*RHS)[thisDOFPosition] -= dirichletContribution;
                        }
                    }
                }
                (*RHS)[thisDOFPosition] += dof.second;
            }
           
        }
        auto end = std::chrono::steady_clock::now(); // Stop the timer
        cout << "Linear System Assembled in "
             << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
        //K->dataStorage->finalizeElementAssignment();

    }
   

    map<short unsigned, map<Direction, map<vector<Position>, short>>> EquilibriumLinearSystemBuilder::
    _initiatePositionsAndPointsMap(short unsigned &maxDerivativeOrder, vector<Direction> &directions) {
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

    tuple<vector<Position>,short> EquilibriumLinearSystemBuilder::
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
        auto result = tuple<vector<Position>,short>();
        for (auto &qualifiedPositionAndPoints : qualifiedPositionsAndPoints) {
            auto qualifiedPositions = qualifiedPositionAndPoints.first;
            if (qualifiedPositions.size() > 1){
                return {qualifiedPositions, qualifiedPositionAndPoints.second};
            }
        }
        for (auto &qualifiedPositionAndPoints : qualifiedPositionsAndPoints) {
            auto qualifiedPositions = qualifiedPositionAndPoints.first;
            if (qualifiedPositions.size() == 1 && qualifiedPositionsAndPoints.size() == 1){
                return {qualifiedPositions, qualifiedPositionAndPoints.second};
            }
            else throw runtime_error("Qualified positions and points not found");
        }
    }
    

}; // LinearAlgebra

#endif //UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H