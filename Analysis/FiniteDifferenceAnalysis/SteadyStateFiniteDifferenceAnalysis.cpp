//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"
#include "../../LinearAlgebra/AnalysisLinearSystemInitializer.h"

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
        SteadyStateMathematicalProblem *mathematicalProblem, Mesh *mesh, FDSchemeSpecs *schemeSpecs) :
        FiniteDifferenceAnalysis(mathematicalProblem, mesh, schemeSpecs){
        degreesOfFreedom = initiateDegreesOfFreedom();
        auto linearSystem = new AnalysisLinearSystemInitializer(degreesOfFreedom, mesh);
        linearSystem->createLinearSystem();
        
        auto fileNameMatlab = "linearSystem.m";
        auto filenameParaview = "mesh.vtk";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        //Utility::Exporters::exportLinearSystemToMatlabFile(linearSystem->matrix, linearSystem->RHS, filePath, fileNameMatlab);
        //Utility::Exporters::saveNodesToParaviewFile(mesh, filePath, filenameParaview);
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis