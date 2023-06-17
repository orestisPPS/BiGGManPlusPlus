//
// Created by hal9000 on 4/25/23.
//

#ifndef UNTITLED_LINEARSYSTEM_H
#define UNTITLED_LINEARSYSTEM_H

#include "Array/Array.h"

namespace LinearAlgebra {

    class LinearSystem {
        
    public:
        
        LinearSystem(shared_ptr<Array<double>> matrix, shared_ptr<vector<double>> rhs);
        

        shared_ptr<Array<double>> matrix;

        shared_ptr<vector<double>> rhs;

        shared_ptr<vector<double>> solution;
        
        void exportToMatlabFile(const string& fileName, const string& filePath, bool printSolution ) const;
    };

} // LinearAlgebra

#endif //UNTITLED_LINEARSYSTEM_H
