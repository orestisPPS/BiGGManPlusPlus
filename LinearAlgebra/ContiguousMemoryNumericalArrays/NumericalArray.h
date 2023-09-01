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
#include "../../ThreadingOperations/ThreadingOperations.h"

using namespace LinearAlgebra;
using namespace std;

namespace LinearAlgebra{
    template<typename T>
    class NumericalArray {
    
    public:
        explicit NumericalArray(T initialValue = 0, ParallelizationMethod parallelizationMethod = SingleThread) :
                                _parallelizationMethod(parallelizationMethod) {
            _values = nullptr;
            _parallelizationMethod = parallelizationMethod;
            _threading = ThreadingOperations<T>(_parallelizationMethod);
        }

        /**
        * \brief Scales all elements of this numerical array.
        * Given vector v = [v1, v2, ..., vn] and scalar factor a, the updated vector is:
        * v = [a*v1, a*v2, ..., a*vn]. The scaling is performed in parallel across multiple threads.
        * 
        * \param scalar The scaling factor.
        */
        void scale(T scalar){
            auto scaleJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end; ++i) {
                    (*_values)[i] *= scalar;
                }
            };
            _threading.executeParallelJob(scaleJob);
        }

    protected: 
        shared_ptr<vector<T>> _values; ///< The underlying data.
        
        ParallelizationMethod _parallelizationMethod; ///< Parallelization method used for array operations.
        
        ThreadingOperations<T> _threading; ///< Threading operations object.
        
    
        
    } 
    

    
} // LinearAlgebra
#endif //UNTITLED_NUMERICALARRAY_H
