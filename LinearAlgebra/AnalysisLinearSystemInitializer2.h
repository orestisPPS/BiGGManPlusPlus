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
        explicit AnalysisLinearSystemInitializer2(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh,
                                                 MathematicalProblem* mathematicalProblem, FDSchemeSpecs* specs);

        ~AnalysisLinearSystemInitializer2();

        LinearSystem* linearSystem;
        
        void createLinearSystem();

        void updateRHS();

    private:

        MathematicalProblem* _mathematicalProblem;

        FDSchemeSpecs* _specs;

        Array<double> *_matrix;
        
        Array<double> *_permutationMatrix;

        vector<double> *_RHS;

        Mesh* _mesh;

        AnalysisDegreesOfFreedom* _analysisDegreesOfFreedom;

        Array<double>* _freeDOFMatrix;

        // Fixed DOF x Total DOF
        Array<double>* _fixedFreeDOFMatrix;

        Array<double>* _totalDOFMatrix;
        
        void _createMatrix();
        
        void _createPermutationMatrix();
        
        void _createFixedDOFSubMatrix();

        void _createFreeDOFSubMatrix();

        void _createTotalDOFSubMatrix();

        map<vector<double>, Node*>* _parametricCoordToNodeMap;

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
