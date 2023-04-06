//
// Created by hal9000 on 4/6/23.
//

#ifndef UNTITLED_EXPORTERS_H
#define UNTITLED_EXPORTERS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../../LinearAlgebra/Array.h"

using namespace std;
using namespace LinearAlgebra;

namespace Utility {

    class Exporters {
        
    public:


        static void exportLinearSystemToMatlabFile(Array<double>* matrix, vector<double>* vector, const string& filePath, const string& fileName);

    };

} // Utility

#endif //UNTITLED_EXPORTERS_H
