//
// Created by hal9000 on 3/28/23.
//

#ifndef UNTITLED_ANALYSISLINEARSYSTEM_H
#define UNTITLED_ANALYSISLINEARSYSTEM_H

#include "../Analysis/AnalysisDOFs/AnalysisDegreesOfFreedom.h"
#include "Array/Array.h"
#include "../Discretization/Node/IsoparametricNodeGraph.h"
#include "../Utility/Exporters/Exporters.h"
#include "LinearSystem.h"

using namespace NumericalAnalysis;

namespace LinearAlgebra {

    class AnalysisLinearSystem : public LinearSystem{
        
        public:
            explicit AnalysisLinearSystem(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh);
            
            ~AnalysisLinearSystem();
            
            unsigned* numberOfFreeDOFs;
            
            unsigned* numberOfFixedDOFs;
            
            unsigned* numberOfDOFs;
            
            void createLinearSystem();
            
            void updateRHS();
    private:
        
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

#endif //UNTITLED_ANALYSISLINEARSYSTEM_H

