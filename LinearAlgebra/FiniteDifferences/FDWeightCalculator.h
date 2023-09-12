//
// Created by hal9000 on 5/1/23.
//

#ifndef UNTITLED_FDWEIGHTCALCULATOR_H
#define UNTITLED_FDWEIGHTCALCULATOR_H

#include <algorithm>
#include <stdexcept>
#include <vector>
#include "../ContiguousMemoryNumericalArrays/NumericalVector/NumericalVector.h"

namespace LinearAlgebra {

    // Recommended functions:
    // calculate_weights (or generate_weights as a convenient wrapper)

    template <typename Real_t>
    void calculate_weights(const Real_t * const __restrict__ grid, const unsigned len_g,
                           const unsigned max_deriv, Real_t * const __restrict__ weights, const Real_t around=0) {
        // Parameters
        // ----------
        // grid[len_g]: array with grid point locations
        // len_g: length of grid
        // weights[len_g, max_deriv+1]: weights of order 0 to max_deriv
        //     (output argument, contiguous memory, column major order, no need to clear before call)
        // max_deriv: highest derivative.
        // around: location where approximations are to be accurate
        //
        // References
        // ----------
        // Generation of Finite Difference Formulas on Arbitrarily
        // Spaced Grids, Bengt Fornberg,
        // Mathematics of compuation, 51, 184, 1988, 699-706
        if (len_g < max_deriv + 1){
            throw std::logic_error("size of grid insufficient");
        }
        Real_t c1, c4, c5;
        c1 = 1;
        c4 = grid[0] - around;
        weights[0] = 1;
        for (unsigned i=1; i < len_g*(max_deriv+1); ++i)
            weights[i] = 0;  // clear weights
        for (unsigned i=1; i < len_g; ++i){
            const unsigned mn = min(i, max_deriv);
            Real_t c2 = 1;
            c5 = c4;
            c4 = grid[i] - around;
            for (unsigned j=0; j<i; ++j){
                const Real_t c3 = grid[i] - grid[j];
                const Real_t c3_r = 1/c3;
                c2 = c2*c3;
                if (j == i-1){
                    const Real_t c2_r = 1/c2;
                    for (unsigned k=mn; k>=1; --k){
                        const Real_t tmp1 = weights[i - 1 + (k-1)*len_g];
                        const Real_t tmp2 = weights[i - 1 + k*len_g];
                        weights[i + k*len_g] = c1*(k*tmp1 - c5*tmp2)*c2_r;
                    }
                    weights[i] = -c1*c5*weights[i-1]*c2_r;
                }
                for (unsigned k=mn; k>=1; --k){
                    const Real_t tmp1 = weights[j + k*len_g];
                    const Real_t tmp2 = weights[j + (k-1)*len_g];
                    weights[j + k*len_g] = (c4*tmp1 - k*tmp2)*c3_r;
                }
                weights[j] = c4*weights[j]*c3_r;
            }
            c1 = c2;
        }
    }

    // Pre-processor macro __cplusplus == 201103L in ISO C++11 compliant compilers. (e.g. GCC >= 4.7.0)
    #if __cplusplus > 199711L
        template<typename Real_t, template<typename, typename...> class Cont, typename... Args>
        Cont<Real_t, Args...> calculateWeights(const Cont<Real_t, Args...>& grid, int maxOrder=-1, const Real_t around=0){
            // Cont<Real_t, Args...> must have contiguous memory storage (e.g. std::vector)
            const unsigned maxorder_ = (maxOrder < 0) ? (grid.size()+1)/2 : maxOrder;
            Cont<Real_t, Args...> coeffs(grid.size()*(maxorder_+1));
            if (grid.size() < maxorder_ + 1){
                throw std::logic_error("size of grid insufficient");
            }
            calculate_weights<Real_t>(&grid[0], grid.size(), maxorder_, &coeffs[0], around);
            return coeffs;
        }
    #endif

    #if __cplusplus > 199711L
        template<typename Real_t, template<typename, typename...> class Cont, typename... Args>
        NumericalVector<Real_t> calculateWeightsOfDerivativeOrder(const Cont<Real_t, Args...>& grid, int order, const Real_t around= 0){
            // Cont<Real_t, Args...> must have contiguous memory storage (e.g. std::vector)
            const unsigned order_ = (order < 0) ? (grid.size()+1)/2 : order;
            Cont<Real_t, Args...> weights(grid.size()*(order_+1));
            if (grid.size() < order_ + 1){
                throw std::logic_error("size of grid insufficient");
            }
            calculate_weights<Real_t>(&grid[0], grid.size(), order_, &weights[0], around);
            auto weightsOfOrder = NumericalVector<Real_t>(grid.size());
            for (unsigned i=0; i<grid.size(); ++i){
                weightsOfOrder[i] = weights[i + order_*grid.size()];
            }
            return weightsOfOrder;
        }
    #endif
    
    #if __cplusplus > 199711L
        template<typename Real_t, template<typename, typename...> class Cont, typename... Args>
        Cont<Cont<Real_t, Args...>> calculateDerivatives(const Cont<Real_t, Args...>& grid, const Cont<Real_t, Args...>& values,
                                                        int maxOrder = -1, const Real_t around=0){
            // Cont<Real_t, Args...> must have contiguous memory storage (e.g. std::vector)
            const unsigned maxOrder_ = (maxOrder < 0) ? (grid.size()+1)/2 : maxOrder;
            Cont<Real_t, Args...> weights(grid.size()*(maxOrder_+1));
            Cont<Cont<Real_t, Args...>> result(2, Cont<Real_t, Args...>(grid.size()));
            if (grid.size() != values.size()){
                throw std::logic_error("size of grid and values mismatch");
            }
            if (grid.size() < maxOrder_ + 1){
                throw std::logic_error("size of grid insufficient");
            }
            calculate_weights<Real_t>(&grid[0], grid.size(), maxOrder_, &weights[0], around);
            for (unsigned j=1; j <= maxOrder_; ++j){
                for (unsigned i=0; i<grid.size(); ++i){
                    result[j-1][i] += weights[i + j*grid.size()] * values[i];
                }
            }
            return result;
        }
    #endif
    

} // LinearAlgebra

#endif //UNTITLED_FDWEIGHTCALCULATOR_H

