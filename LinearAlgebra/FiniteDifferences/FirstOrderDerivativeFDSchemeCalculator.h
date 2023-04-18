//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FIRSTORDERFDSCHEME_H
#define UNTITLED_FIRSTORDERDERIVATIVEFDSCHEME_H

#include <map>
#include <vector>
#include <tuple>
#include <stdexcept>
#include "FDScheme.h"

using namespace std;

namespace LinearAlgebra {
    //A class containing all the first order finite difference schemes up to Fifth order accuracy
    class FirstOrderDerivativeFDSchemeCalculator : public FDScheme {
    public:

        FirstOrderDerivativeFDSchemeCalculator();
        
        static map<int, double> getWeights(FiniteDifferenceSchemeType schemeType, unsigned errorOrder);
        
        // Multiplies the weights with the function values at the points needed for the scheme
        //TODO: create scheme component class
        static map<int, double> getSchemeValues(FiniteDifferenceSchemeType schemeType, unsigned errorOrder,
                                         map<int, double>& functionValues, double stepSize);

        // TODO: create scheme component class
        static tuple<FiniteDifferenceSchemeType, int, map<int, double>>
        getSchemeFromGivenPoints(map<int, double>& functionValues, double stepSize);
    private:
        // First Derivative Forward Difference Scheme 1
        // Numerical Scheme: (u_i+1 - u_i) / h
        // Cut-Off Error: O(h)
        static map<int, double> _forward1();

        // First Derivative Forward Difference Scheme 2
        // Numerical Scheme: (-3u_i + 4u_i+1 - u_i+2) / (2h)
        // Cut-Off Error: O(h^2)
        static map<int, double> _forward2();

        // First Derivative Forward Difference Scheme 3
        // Numerical Scheme: (-11u_i + 18u_i+1 - 9u_i+2 + 2u_i+3) / (6h)
        // Cut-Off Error: O(h^3)
        static map<int, double> _forward3();

        // First Derivative Forward Difference Scheme 4
        // Numerical Scheme: (-25u_i + 48u_i+1 - 36u_i+2 + 16u_i+3 - 3u_i+4) / (12h)
        // Cut-Off Error: O(h^4)
        static map<int, double> _forward4();

        // First Derivative Forward Difference Scheme 5
        // Numerical Scheme: (-137u_i + 300u_i+1 - 300u_i+2 + 200u_i+3 - 75u_i+4 + 12u_i+5) / (60h)
        // Cut-Off Error: O(h^5)
        static map<int, double> _forward5();

        // First Derivative Backward Difference Scheme 1
        // Numerical Scheme: (u_i - u_i-1) / h
        // Cut-Off Error: O(h)
        static map<int, double> _backward1();
        
        // First Derivative Backward Difference Scheme 2
        // Numerical Scheme: (3u_i - 4u_i-1 + u_i-2) / (2h)
        // Cut-Off Error: O(h^2)
        static map<int, double> _backward2();
        
        // First Derivative Backward Difference Scheme 3
        // Numerical Scheme: (11u_i - 18u_i-1 + 9u_i-2 - 2u_i-3) / (6h)
        // Cut-Off Error: O(h^3)
        static map<int, double> _backward3();

        // First Derivative Backward Difference Scheme 4
        // Numerical Scheme: (25u_i - 48u_i-1 + 36u_i-2 - 16u_i-3 + 3u_i-4) / (12h)
        // Cut-Off Error: O(h^4)
        static map<int, double> _backward4();

        // First Derivative Backward Difference Scheme 5
        // Numerical Scheme: (137u_i - 300u_i-1 + 300u_i-2 - 200u_i-3 + 75u_i-4 - 12u_i-5) / (60h)
        // Cut-Off Error: O(h^5)
        static map<int, double> _backward5();

        // First Derivative Central Difference Scheme 2
        // Numerical Scheme: (u_i+1 - u_i-1) / (2h)
        // Cut-Off Error: O(h^2)
        static map<int, double> _central2();

        // First Derivative Central Difference Scheme 4
        // Numerical Scheme: (-u_i+2 + 8u_i+1 - 8u_i-1 + u_i-2) / (12h)
        // Cut-Off Error: O(h^4)
        static map<int, double> _central4();
        
        // First Derivative Central Difference Scheme 6
        // Numerical Scheme: (-u_i+3 + 9u_i+2 - 45u_i+1 + 45u_i-1 - 9u_i-2 + u_i-3) / (60h)
        // Cut-Off Error: O(h^6)
        static map<int, double> _central6();
        
        
        static map<tuple<FiniteDifferenceSchemeType, unsigned>, map<int, double>> _schemeTypeAndOrderToSchemeWeights();
        

        
    };

} // LinearAlgebra

#endif //UNTITLED_FIRSTORDERFDSCHEME_H
