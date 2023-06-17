//
// Created by hal9 000 on 3/28/23.
//

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
        _rhsVector = make_shared<vector<double>>(*_analysisDegreesOfFreedom->numberOfFreeDOF, 0);
        _matrix = make_shared<Array<double>>(*_analysisDegreesOfFreedom->numberOfFreeDOF, *_analysisDegreesOfFreedom->numberOfFreeDOF, 1);
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
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
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {

            //Define the node where the dof belongs
            auto node = _mesh->nodeFromID(dof->parentNode());
            auto thisDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
            
            //Find the node neighbours with a span equal to the maximum number of points needed for the scheme to be consistent
            auto graph = IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap,
                                                _mesh->nodesPerDirection);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);

            //Derivative order 0
            auto zeroOrderCoefficient = _getPDECoefficient(0, node);
            //Define the position of the dof in the Matrix
            _matrix->at(thisDOFPosition, thisDOFPosition) = zeroOrderCoefficient;

            //add source term
            _rhsVector->at(thisDOFPosition) += *_mathematicalProblem->pde->properties->getLocalProperties(dof->parentNode()).sourceTerm;

            //March through all the non-zero derivative orders 
            for (auto derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++){

                //Decompose scheme into directional components
                for (auto &direction: directions) {
                    double iThDerivativePDECoefficient = _getPDECoefficient(derivativeOrder, node, direction);
                    if (iThDerivativePDECoefficient != 0){
                        auto directionIndex = spatialDirectionToUnsigned[direction];

                        //Check if the available positions are qualified for the current derivative order
                        auto qualifiedPositions = _getQualifiedFromAvailable(availablePositionsAndDepth[direction],templatePositionsAndPointsMap[derivativeOrder][direction]);
                        auto scheme = FiniteDifferenceSchemeBuilder::getSchemeWeightsFromQualifiedPositions(
                                qualifiedPositions, direction,_specs->getErrorOrderOfSchemeTypeForDerivative(derivativeOrder), derivativeOrder);

                        auto graphFilter = map<Position, unsigned short>();
                        for (auto &tuple: qualifiedPositions) {
                            for (auto &point: tuple.first) {
                                graphFilter.insert(pair<Position, unsigned short>(point, tuple.second));
                            }
                        }
                        auto filteredNodeGraph = graph.getNodeGraph(graphFilter);

                        auto colinearCoordinates = graph.getSameColinearNodalCoordinates(_coordinateType, filteredNodeGraph);
                        auto colinearDOF = graph.getColinearDOF(dof->type(), direction, filteredNodeGraph);

                        auto step = VectorOperations::averageAbsoluteDifference(colinearCoordinates[direction][directionIndex]);
                        //Calculate the denominator (h^p)
                        double denominator = scheme.denominatorCoefficient * pow(step, scheme.power);
                        vector<double> &schemeWeights = scheme.weights;
                        for (int iDof = 0; iDof < colinearDOF.size(); ++iDof) {
                            auto neighbourDOF = colinearDOF[iDof];
                            auto weight = schemeWeights[iDof] * iThDerivativePDECoefficient / denominator;

                            if (neighbourDOF->constraintType() == Free) {
                                auto neighbourDOFPosition = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(colinearDOF[iDof]);
                                _matrix->at(thisDOFPosition, neighbourDOFPosition) =
                                        //_matrix->at(thisDOFPosition, neighbourDOFPosition) + schemeWeights[iDof] * iThDerivativePDECoefficient;
                                        _matrix->at(thisDOFPosition, neighbourDOFPosition) + weight;
                            }
                            else if(neighbourDOF->constraintType() == Fixed){
                                auto dirichletContribution = neighbourDOF->value() * weight;
                                _rhsVector->at(thisDOFPosition) -= dirichletContribution;
                            }
                        }
                    }
                }
            }
        }
        this->linearSystem = make_shared<LinearSystem>(std::move(_matrix), std::move(_rhsVector) );
        
        this->linearSystem->matrix->print();
        
        

        auto end = std::chrono::steady_clock::now(); // Stop the timer
        cout << "Linear System Assembled in "
             << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
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

    double AnalysisLinearSystemInitializer::_getPDECoefficient(unsigned short derivativeOrder, Node *parentNode,
                                                               Direction direction) {
        auto directionIndex = spatialDirectionToUnsigned[direction];
        switch (derivativeOrder){
            case 0:
               return *_mathematicalProblem->pde->properties->getLocalProperties(*parentNode->id.global).zerothOrderCoefficient;
            case 1:
                return _mathematicalProblem->pde->properties->getLocalProperties(*parentNode->id.global).firstOrderCoefficients->at(directionIndex);
            case 2:
                return _mathematicalProblem->pde->properties->getLocalProperties(*parentNode->id.global).secondOrderCoefficients->at(directionIndex, directionIndex);
            default:
                throw runtime_error("Derivative order should be 0, 1 or 2");
        }
    }

}// LinearAlgebra
