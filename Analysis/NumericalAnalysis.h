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
        NumericalAnalysis(shared_ptr<MathematicalProblem> mathematicalProblem, shared_ptr<Mesh> mesh,
                          shared_ptr<Solver> solver, CoordinateType coordinateSystem = Natural);
        
        
        shared_ptr<MathematicalProblem> mathematicalProblem;
        
        shared_ptr<Mesh> mesh;
                
        shared_ptr<AnalysisDegreesOfFreedom> degreesOfFreedom;
        
        shared_ptr<LinearSystem> linearSystem;
        
        shared_ptr<Solver> solver;

        void solve() const;
        
        void applySolutionToDegreesOfFreedom() const;
        
        NumericalVector<double> getSolutionAtNode(NumericalVector<double>& nodeCoordinates, double tolerance = 1E-4) const;
        
    protected:

        shared_ptr<AnalysisDegreesOfFreedom> initiateDegreesOfFreedom() const;
    };

} // NumericalAnalysis

#endif //UNTITLED_ANALYSIS_H

