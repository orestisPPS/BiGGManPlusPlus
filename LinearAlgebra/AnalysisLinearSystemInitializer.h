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

using namespace NumericalAnalysis;

namespace LinearAlgebra {

    class AnalysisLinearSystemInitializer{
        
        public:
            explicit AnalysisLinearSystemInitializer(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh);
            
            ~AnalysisLinearSystemInitializer();
            
            LinearSystem* linearSystem;
            
            unsigned* numberOfFreeDOFs;
            
            unsigned* numberOfFixedDOFs;
            
            unsigned* numberOfDOFs;
            
            void createLinearSystem();
            
            void updateRHS();
    private:
        
        Array<double> *_matrix;
        
        vector<double> *_RHS;
        
        Mesh* _mesh;
        
        AnalysisDegreesOfFreedom* _analysisDegreesOfFreedom;
        
        Array<double>* _freeDOFMatrix;

        // Fixed DOF x Total DOF
        Array<double>* _fixedDOFMatrix;

        void _createMatrix();
        void _createFixedDOFSubMatrix();
        void _createFreeDOFSubMatrix();

        map<vector<double>, Node*>* _parametricCoordToNodeMap;
        
        void _createRHS();
    };

} // LinearAlgebra

#endif //UNTITLED_ANALYSISLINEARSYSTEMINITIALIZER_H

