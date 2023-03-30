//
// Created by hal9000 on 3/28/23.
//

#ifndef UNTITLED_LINEARSYSTEM_H
#define UNTITLED_LINEARSYSTEM_H

#include "../Analysis/AnalysisDOFs/AnalysisDegreesOfFreedom.h"
#include "Array.h"

using namespace NumericalAnalysis;

namespace LinearAlgebra {

    class LinearSystem {
        public:
            explicit LinearSystem(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh);
            
            ~LinearSystem();
            
            Array<double>* matrix;
            
            vector<double>* RHS;
            
            unsigned* numberOfDOFs;
            
            void createLinearSystem(Mesh* mesh);
            
            void updateRHS();
    private:
        AnalysisDegreesOfFreedom* _analysisDegreesOfFreedom;
        
        void createMatrix(Mesh* mesh);
        
        void createRHS();
    };

} // LinearAlgebra

#endif //UNTITLED_LINEARSYSTEM_H

