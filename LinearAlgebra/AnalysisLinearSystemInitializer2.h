//
// Created by hal9000 on 6/2/23.
//

#ifndef UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H
#define UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H

#include "Array/Array.h"
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

    class AnalysisLinearSystemInitializer2{

    public:
        explicit AnalysisLinearSystemInitializer2(shared_ptr<AnalysisDegreesOfFreedom> analysisDegreesOfFreedom, shared_ptr<Mesh> mesh,
                                                 shared_ptr<MathematicalProblem> mathematicalProblem, shared_ptr<FDSchemeSpecs> specs);

        ~AnalysisLinearSystemInitializer2();

        shared_ptr<LinearSystem> linearSystem;
        
        void createLinearSystem();

        void updateRHS();

    private:

        shared_ptr<MathematicalProblem> _mathematicalProblem;

        shared_ptr<FDSchemeSpecs> _specs;

        shared_ptr<Array<double>>_matrix;
        
        shared_ptr<Array<double>>_permutationMatrix;

        shared_ptr<Array<double>>_RHS;

        shared_ptr<Mesh> _mesh;

        shared_ptr<AnalysisDegreesOfFreedom> _analysisDegreesOfFreedom;

        shared_ptr<Array<double>> _freeDOFMatrix;

        // Fixed DOF x Total DOF
        shared_ptr<Array<double>> _fixedFreeDOFMatrix;

        shared_ptr<Array<double>> _totalDOFMatrix;
        
        void _createMatrix();
        
        void _createPermutationMatrix();
        
        void _createFixedDOFSubMatrix();

        void _createFreeDOFSubMatrix();

        void _createTotalDOFSubMatrix();

        shared_ptr<map<vector<double>, Node*>> _parametricCoordToNodeMap;

        static map<short unsigned, map<Direction, map<vector<Position>, short>>>
        _initiatePositionsAndPointsMap(short unsigned& maxDerivativeOrder, vector<Direction>& directions);

        static void _checkIfAvailableAreQualified(map<vector<Position>, unsigned short>& availablePositionsAndPoints,
                                                  map<vector<Position>, short>& templatePositionsAndPoints,
                                                  map<vector<Position>, short>& qualifiedPositionsAndPoints);

        tuple<FDSchemeType, short>
        _getSchemeTypeAndOrderFromQualified(map<vector<Position>, short>& qualifiedPositionsAndPoints);

        void _createRHS();
    };

} // LinearAlgebra

#endif //UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER2_H
