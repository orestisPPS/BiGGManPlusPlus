//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_ANALYSIS_H
#define UNTITLED_ANALYSIS_H

#include "../MathematicalEntities/MathematicalProblem/MathematicalProblem.h"
#include "FiniteDifferenceAnalysis/EquilibriumLinearSystemBuilder.h"
#include "../LinearAlgebra/Solvers/Solver.h"
#include "../Utility/Exporters/Exporters.h"


using namespace LinearAlgebra;
using namespace MathematicalEntities;
using namespace Discretization;
using namespace MathematicalEntities;

namespace NumericalAnalysis {

    class  NumericalAnalysis {
        public:
        NumericalAnalysis(shared_ptr<MathematicalProblem> mathematicalProblem, shared_ptr<Mesh> mesh, shared_ptr<Solver> solver);
        
        void setAnalysisCoordinateSystem(CoordinateType coordinateSystem);
        
        
        shared_ptr<MathematicalProblem> mathematicalProblem;
        
        shared_ptr<Mesh> mesh;
        
        CoordinateType coordinateSystem;
                
        shared_ptr<AnalysisDegreesOfFreedom> degreesOfFreedom;
        
        shared_ptr<LinearSystem> linearSystem;
        
        shared_ptr<Solver> solver;

        virtual void solve();
        
        virtual void applySolutionToDegreesOfFreedom() const;
        
        virtual NumericalVector<double> getSolutionAtNode(NumericalVector<double>& nodeCoordinates, double tolerance = 1E-4, DOFType dofType = NoDOFType) const; 
        
        
    protected:
        unique_ptr<EquilibriumLinearSystemBuilder> _linearSystemInitializer;

        shared_ptr<AnalysisDegreesOfFreedom> initiateDegreesOfFreedom() const;
    };

} // NumericalAnalysis

#endif //UNTITLED_ANALYSIS_H

