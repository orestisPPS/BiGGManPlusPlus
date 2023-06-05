//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_ANALYSIS_H
#define UNTITLED_ANALYSIS_H

#include "../LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h"
#include "../MathematicalProblem/MathematicalProblem.h"
#include "../LinearAlgebra/AnalysisLinearSystemInitializer.h"
#include "AnalysisDOFs/AnalysisDegreesOfFreedom.h"
#include "../LinearAlgebra/Solvers/Solver.h"
#include "../Utility/Exporters/Exporters.h"


using namespace LinearAlgebra;
using namespace MathematicalProblems;
using namespace Discretization;
using namespace MathematicalProblems;

namespace NumericalAnalysis {

    class NumericalAnalysis {
        public:
        NumericalAnalysis(MathematicalProblem* mathematicalProblem, Mesh *mesh, Solver* solver, CoordinateType coordinateSystem = Natural);
        
        ~NumericalAnalysis();
        
        MathematicalProblem* mathematicalProblem;
        
        Mesh* mesh;
                
        AnalysisDegreesOfFreedom* degreesOfFreedom;
        
        LinearSystem* linearSystem;
        
        Solver *solver;

        void solve() const;
        
        void applySolutionToDegreesOfFreedom() const;
        
        vector<double> getSolutionAtNode(vector<double>& nodeCoordinates, double tolerance = 1E-4) const;
        
    protected:

        AnalysisDegreesOfFreedom* initiateDegreesOfFreedom() const;
    };

} // NumericalAnalysis

#endif //UNTITLED_ANALYSIS_H

