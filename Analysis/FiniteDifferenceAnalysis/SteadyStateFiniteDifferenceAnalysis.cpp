//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"
#include "../../LinearAlgebra/AnalysisLinearSystemInitializer.h"
#include "../../LinearAlgebra/Solvers/Direct/SolverLUP.h"

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
        SteadyStateMathematicalProblem *mathematicalProblem, Mesh *mesh, Solver* solver,  FDSchemeSpecs *schemeSpecs) :
        FiniteDifferenceAnalysis(mathematicalProblem, mesh, solver, schemeSpecs){
        
        auto linearSystemInitializer =
                new AnalysisLinearSystemInitializer(degreesOfFreedom, mesh, mathematicalProblem, schemeSpecs);
        linearSystemInitializer->createLinearSystem();
        linearSystem = linearSystemInitializer->linearSystem;
        solver->setLinearSystem(linearSystem);
        
/*        auto fileNameMatlab = "linearSystem.m";
        auto filenameParaview = "mesh.vtk";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        //Utility::Exporters::exportLinearSystemToMatlabFile(linearSystem->matrix, linearSystem->RHS, filePath, fileNameMatlab);
        //Utility::Exporters::saveNodesToParaviewFile(mesh, filePath, filenameParaview);*/
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis