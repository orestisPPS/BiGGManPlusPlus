//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"
#include "../../LinearAlgebra/AnalysisLinearSystemInitializer.h"
#include "../../LinearAlgebra/Solvers/Direct/SolverLUP.h"

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
        SteadyStateMathematicalProblem *mathematicalProblem, Mesh *mesh, Solver* solver,  FDSchemeSpecs *schemeSpecs, CoordinateType coordinateSystem) :
        FiniteDifferenceAnalysis(mathematicalProblem, mesh, solver, schemeSpecs){
        
        auto linearSystemInitializer =
                new AnalysisLinearSystemInitializer(degreesOfFreedom, mesh, mathematicalProblem, schemeSpecs, coordinateSystem);
        linearSystemInitializer->createLinearSystem();
        this->linearSystem = linearSystemInitializer->linearSystem;
        solver->setLinearSystem(linearSystem);
        cout<<"Matrix: "<<endl;
        this->linearSystem->matrix->print();
        cout<<"RHS: "<<endl;
        for (double i : *this->linearSystem->RHS) {
            cout<<i<<endl;
        }
        cout<<"SYSTEM BEFORE SOLVER..."<<endl;
        

        //Utility::Exporters::saveNodesToParaviewFile(mesh, filePath, filenameParaview);
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis