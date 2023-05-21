//
// Created by hal9000 on 3/28/23.
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

    class AnalysisLinearSystemInitializer{
        
        public:
            explicit AnalysisLinearSystemInitializer(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh,
                                                     MathematicalProblem* mathematicalProblem, FDSchemeSpecs* specs);
            
            ~AnalysisLinearSystemInitializer();
            
            LinearSystem* linearSystem;
            
            unsigned* numberOfFreeDOFs;
            
            unsigned* numberOfFixedDOFs;
            
            unsigned* numberOfDOFs;
            
            void createLinearSystem();
            
            void updateRHS();
            
    private:
        
        MathematicalProblem* _mathematicalProblem;
        
        FDSchemeSpecs* _specs;
        
        Array<double> *_matrix;
        
        vector<double> *_RHS;
        
        Mesh* _mesh;
        
        AnalysisDegreesOfFreedom* _analysisDegreesOfFreedom;
        
        Array<double>* _freeDOFMatrix;

        // Fixed DOF x Total DOF
        Array<double>* _fixedDOFMatrix;

        Array<double>* _totalDOFMatrix;

        void _createMatrix();
        void _createFixedDOFSubMatrix();
        void _createFreeDOFSubMatrix();

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

#endif //UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H

