//
// Created by hal9000 on 3/28/23.
//

#ifndef UNTITLED_LINEARSYSTEM_H
#define UNTITLED_LINEARSYSTEM_H

#include "../Analysis/AnalysisDOFs/AnalysisDegreesOfFreedom.h"

using namespace NumericalAnalysis;

namespace LinearAlgebra {

    class LinearSystem {
        public:
            LinearSystem(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom);
            void createMatrix();
            void createRHS();
            void updateRHS();
            ~LinearSystem();
    };

} // LinearAlgebra

#endif //UNTITLED_LINEARSYSTEM_H
