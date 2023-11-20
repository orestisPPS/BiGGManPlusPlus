//
// Created by hal9000 on 4/25/23.
//

#ifndef UNTITLED_LINEARSYSTEM_H
#define UNTITLED_LINEARSYSTEM_H

#include "ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"

namespace LinearAlgebra {

    class LinearSystem {

    public:

        LinearSystem(shared_ptr<NumericalMatrix<double>> matrix, shared_ptr<NumericalVector<double>> rhs);
        
        shared_ptr<NumericalMatrix<double>> matrix;

        shared_ptr<NumericalVector<double>> rhs;

        shared_ptr<NumericalVector<double>> solution;

        void exportToMatlabFile(const string &fileName, const string &filePath, bool printSolution) const;
    };

} // LinearAlgebra

#endif //UNTITLED_LINEARSYSTEM_H
