//
// Created by hal9000 on 7/30/23.
//

#ifndef UNTITLED_STEADYSTATEDIRICHLET3D_H
#define UNTITLED_STEADYSTATEDIRICHLET3D_H
#include "../../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../../StructuredMeshGeneration/MeshFactory.h"
#include "../../StructuredMeshGeneration/MeshSpecs.h"
#include "../../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../../LinearAlgebra/Solvers/Direct/SolverLUP.h"
#include "../../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"
#include "../../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"


namespace Tests {

    class SteadyStateDirichlet3D {
        
    public:
        SteadyStateDirichlet3D();
        


    };

} // Tests

#endif //UNTITLED_STEADYSTATEDIRICHLET3D_H
