//
// Created by hal9000 on 9/1/23.
//

#ifndef UNTITLED_NUMERICALARRAY_H
#define UNTITLED_NUMERICALARRAY_H
#include <vector>
#include <stdexcept>
#include <memory>
#include <thread>
#include <type_traits>
#include <valarray>
#include <random>
#include "../ParallelizationMethods.h"
using namespace LinearAlgebra;
using namespace std;

namespace LinearAlgebra{
    template<typename T>
    class NumericalArray {
    
    public:
        explicit NumericalArray(T initialValue = 0, ParallelizationMethod parallelizationMethod = SingleThread) : _parallelizationMethod(parallelizationMethod) {
            _values = nullptr;
            _parallelizationMethod = parallelizationMethod;
        }

    protected: 
        shared_ptr<vector<T>> _values; ///< The underlying data.
        
        ParallelizationMethod _parallelizationMethod; ///< Parallelization method used for array operations.
    } 
    

    
} // LinearAlgebra
#endif //UNTITLED_NUMERICALARRAY_H
