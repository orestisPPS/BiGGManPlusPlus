//
// Created by hal9000 on 3/28/23.
//

#ifndef UNTITLED_EQUILIBRIUMLINEARSYSTEMBUILDER_H
#define UNTITLED_EQUILIBRIUMLINEARSYSTEMBUILDER_H
#include "../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../Discretization/Node/IsoparametricNodeGraph.h"
#include "../../Utility/Exporters/Exporters.h"
#include "../../LinearAlgebra/LinearSystem.h"
#include "../AnalysisDOFs/AnalysisDegreesOfFreedom.h"
#include "../../Discretization/Mesh/Mesh.h"
#include "../../MathematicalEntities/MathematicalProblem/MathematicalProblem.h"
#include "../../LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeBuilder.h"
using namespace MathematicalEntities;
using namespace NumericalAnalysis;

namespace NumericalAnalysis {

    class EquilibriumLinearSystemBuilder{

    public:
        explicit EquilibriumLinearSystemBuilder(const shared_ptr<AnalysisDegreesOfFreedom>& analysisDegreesOfFreedom, const shared_ptr<Mesh> &mesh,
                                                const shared_ptr<SteadyStateMathematicalProblem>& mathematicalProblem, const shared_ptr<FiniteDifferenceSchemeOrder>& specs,
                                                CoordinateType = Natural);
        
        shared_ptr<LinearSystem> linearSystem;

        shared_ptr<NumericalMatrix<double>> K;

        shared_ptr<NumericalVector<double>> RHS;
        
        void assembleSteadyStateLinearSystem();
        
        void updateRHS();

    protected:
        
        shared_ptr<FiniteDifferenceSchemeOrder> _specs;
        
        CoordinateType _coordinateType;

        shared_ptr<Mesh> _mesh;

        shared_ptr<AnalysisDegreesOfFreedom> _analysisDegreesOfFreedom;
        
        shared_ptr<map<vector<double>, Node*>> _parametricCoordToNodeMap;

        static map<short unsigned, map<Direction, map<vector<Position>, short>>>
        _initiatePositionsAndPointsMap(short unsigned& maxDerivativeOrder, vector<Direction>& directions);

        static tuple<vector<Position>,short> _getQualifiedFromAvailable(map<vector<Position>,unsigned short>& availablePositionsAndPoints,
                                                                      map<vector<Position>,short>& templatePositionsAndPoints);
        
        void _addNeumannBoundaryConditions();
        
    private:
        shared_ptr<SteadyStateMathematicalProblem> _steadyStateMathematicalProblem;

    };

} // LinearAlgebra

#endif //UNTITLED_EQUILIBRIUMLINEARSYSTEMBUILDER_H
