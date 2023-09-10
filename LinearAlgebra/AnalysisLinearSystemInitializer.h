//
// Created by hal9000 on 3/28/23.
//

#ifndef UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H
#define UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H
#include "ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../Discretization/Node/IsoparametricNodeGraph.h"
#include "../Utility/Exporters/Exporters.h"
#include "LinearSystem.h"
#include "../Analysis/AnalysisDOFs/AnalysisDegreesOfFreedom.h"
#include "../Discretization/Mesh/Mesh.h"
#include "../MathematicalProblem/MathematicalProblem.h"
#include "../LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeBuilder.h"
using namespace MathematicalProblems;

using namespace NumericalAnalysis;

namespace LinearAlgebra {

    class AnalysisLinearSystemInitializer{

    public:
        explicit AnalysisLinearSystemInitializer(shared_ptr<AnalysisDegreesOfFreedom> analysisDegreesOfFreedom, const shared_ptr<Mesh> &mesh,
                                                 shared_ptr<MathematicalProblem> mathematicalProblem, shared_ptr<FDSchemeSpecs> specs, CoordinateType = Natural);
        
        shared_ptr<LinearSystem> linearSystem;
        
        void createLinearSystem();
        
        void addNeumannBoundaryConditions();

        void updateRHS();

    private:

        shared_ptr<MathematicalProblem> _mathematicalProblem;

        shared_ptr<FDSchemeSpecs> _specs;
        
        CoordinateType _coordinateType;
        
        shared_ptr<NumericalVector<double>> _rhsVector;
        
        shared_ptr<NumericalMatrix<double>> _matrix;

        shared_ptr<Mesh> _mesh;

        shared_ptr<AnalysisDegreesOfFreedom> _analysisDegreesOfFreedom;
        
        shared_ptr<map<NumericalVector<double>, Node*>> _parametricCoordToNodeMap;

        static map<short unsigned, map<Direction, map<vector<Position>, short>>>
        _initiatePositionsAndPointsMap(short unsigned& maxDerivativeOrder, vector<Direction>& directions);

        static map<vector<Position>,short> _getQualifiedFromAvailable(map<vector<Position>,unsigned short>& availablePositionsAndPoints,
                                                                      map<vector<Position>,short>& templatePositionsAndPoints);
        
        double _getPDECoefficient(unsigned short derivativeOrder, Node* parentNode, Direction direction = None);
    };

} // LinearAlgebra

#endif //UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H
