//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_ANALYSIS_H
#define UNTITLED_ANALYSIS_H

#include "../LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h"
#include "../MathematicalProblem/MathematicalProblem.h"
#include "../Discretization/Mesh/Mesh.h"
#include "AnalysisDOFs/AnalysisDegreesOfFreedom.h"

using namespace LinearAlgebra;
using namespace MathematicalProblems;
using namespace Discretization;
using namespace MathematicalProblems;

namespace NumericalAnalysis {

    class NumericalAnalysis {
        public:
        NumericalAnalysis(MathematicalProblem* mathematicalProblem, Mesh *mesh);
        
        ~NumericalAnalysis();
        
        MathematicalProblem* mathematicalProblem;
        
        Mesh* mesh;
                
        AnalysisDegreesOfFreedom* degreesOfFreedom;
        
        //virtual void createLinearSystem();
        
        //virtual void solve();
        
    protected:

        AnalysisDegreesOfFreedom* initiateDegreesOfFreedom() const;
    };

} // NumericalAnalysis

#endif //UNTITLED_ANALYSIS_H

