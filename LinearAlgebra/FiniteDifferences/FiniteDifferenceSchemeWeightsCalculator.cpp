//
// Created by hal9000 on 4/23/23.
//

#include "FiniteDifferenceSchemeWeightsCalculator.h"
#include "../LinearSystem.h"
#include "../Solvers/Direct/SolverLUP.h"

namespace LinearAlgebra {
    FiniteDifferenceSchemeWeightsCalculator::
    FiniteDifferenceSchemeWeightsCalculator(){
        
    }
    
    vector<double> FiniteDifferenceSchemeWeightsCalculator::calculateWeights(unsigned derivativeOrder, vector<double>& positions) {
        auto numberOfPoints = static_cast<unsigned >(positions.size());
        
        auto A = new Array<double>(numberOfPoints, numberOfPoints);
        auto b = new vector<double>(numberOfPoints, 0);
        
        // March through all rows
        for (auto row = 0; row < numberOfPoints; row++) {
            // March through all columns
            for (auto column = 0; column < numberOfPoints; column++) {
                A->at(row, column) =  pow(positions[column], row);
            }
        }
        //A->print();
        b->at(derivativeOrder) = 1.0;
        
        auto linearSystem = new LinearSystem(A, b);
        auto solver = new SolverLUP(1e-10, true);
        solver->setLinearSystem(linearSystem);
        solver->solve();
/*        auto fileNameMatlab = "linearSystem.m";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        Utility::Exporters::exportLinearSystemToMatlabFile(A, b, filePath, fileNameMatlab, false);*/
        auto solution = *linearSystem->solution;
        delete solver;
        delete linearSystem;
        return solution;
    }
} // LinearAlgebra