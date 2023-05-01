//
// Created by hal9000 on 4/23/23.
//

#include "FiniteDifferenceSchemeWeightsCalculator.h"


namespace LinearAlgebra {

    
    //Least squares polynomial fitting using the Vandermonde matrix
    vector<double> FiniteDifferenceSchemeWeightsCalculator::calculateVandermondeCoefficients(unsigned derivativeOrder, vector<double>& positions) {
        auto n = static_cast<unsigned >(positions.size());
        
        //Vandermonde matrix
        auto V = new Array<double>(n, n);
        // March through all rows
        for (auto row = 0; row < n; row++) {
            // March through all columns
            for (auto column = 0; column < n; column++) {
                V->at(row, column) = pow(positions[column], row);
                if (abs(V->at(row, column)) == 0 && row == column) {
                    V->at(row, column) = 1E-18;
                }
            }
        }
        V->print();
        
        //Right hand side vector
        auto b = new vector<double>(n, 0);
        b->at(derivativeOrder) = 1;

        auto linearSystem = new LinearSystem(V, b);
        auto solver = new SolverLUP(1E-9, true);
        solver->setLinearSystem(linearSystem);
        solver->solve();

        auto weights = vector<double>(*linearSystem->solution);
        delete linearSystem;
        delete solver;
        
/*        //Calculate the weights using the undetermined coefficient method
        vector<double> weights(n, 0);

        for (int i = 0; i < n; ++i) {
            auto weight = b->at(i);
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    weight /= (positions[i] - positions[j]);
                }
            }
            weights[i] = weight;
        }
        return weights;*/
        
        

        return weights;
        
    }
    
}// LinearAlgebra