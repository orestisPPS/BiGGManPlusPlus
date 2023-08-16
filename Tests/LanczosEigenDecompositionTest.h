//
// Created by hal9000 on 8/3/23.
//

#ifndef UNTITLED_LANCZOSEIGENDECOMPOSITIONTEST_H
#define UNTITLED_LANCZOSEIGENDECOMPOSITIONTEST_H

#include "../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../StructuredMeshGeneration/MeshFactory.h"
#include "../StructuredMeshGeneration/MeshSpecs.h"
#include "../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../LinearAlgebra/Solvers/Direct/SolverLUP.h"
#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"
#include "../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"
#include "../LinearAlgebra/EigenDecomposition/LanczosEigenDecomposition.h"
#include "../LinearAlgebra/EigenDecomposition/PowerMethod.h"
#include "../LinearAlgebra/EigenDecomposition/QR/DecompositionQR.h"

namespace Tests {

    class LanczosEigenDecompositionTest {
    public:

        LanczosEigenDecompositionTest();

    };

} // Tests

#endif //UNTITLED_LANCZOSEIGENDECOMPOSITIONTEST_H
