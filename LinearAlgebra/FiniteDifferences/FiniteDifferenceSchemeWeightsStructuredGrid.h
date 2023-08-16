//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FIRSTORDERFDSCHEME_H
#define UNTITLED_FIRSTORDERDERIVATIVEFDSCHEME_H

#include <map>
#include <vector>
#include <tuple>
#include <stdexcept>
#include "FDSchemeType.h"

using namespace std;

namespace LinearAlgebra {

    struct Scheme{
    public:
        unsigned short power;
        double denominatorCoefficient;
        vector<double> weights;
    };

    struct SchemeMap {
    public:
        std::map<int, double> weights;
        unsigned short power;
        double denominatorCoefficient;
    };
    
    //A class containing all the first order finite difference schemes up to Fifth order accuracy
    class FiniteDifferenceSchemeWeightsStructuredGrid{
    public:

        FiniteDifferenceSchemeWeightsStructuredGrid();
        
        static Scheme getScheme( FDSchemeType schemeType, unsigned short derivativeOrder, unsigned errorOrder);
        
    private:
        
        static SchemeMap _getSchemeFromDerivativeOrder(FDSchemeType schemeType, unsigned derivativeOrder, unsigned int errorOrder);


        //====================================================================================================
        //===================================First Order Derivative Schemes===================================
        //====================================================================================================
        
        // First Derivative Forward Difference Scheme 1
        // Numerical Scheme: (u_i+1 - u_i) / h
        // Cut-Off Error: O(h)
        static SchemeMap _forward1_1();

        // First Derivative Forward Difference Scheme 2
        // Numerical Scheme: (-3u_i + 4u_i+1 - u_i+2) / (2h)
        // Cut-Off Error: O(h^2)
        static SchemeMap _forward1_2();

        // First Derivative Forward Difference Scheme 3
        // Numerical Scheme: (-11u_i + 18u_i+1 - 9u_i+2 + 2u_i+3) / (6h)
        // Cut-Off Error: O(h^3)
        static SchemeMap _forward1_3();

        // First Derivative Forward Difference Scheme 4
        // Numerical Scheme: (-25u_i + 48u_i+1 - 36u_i+2 + 16u_i+3 - 3u_i+4) / (12h)
        // Cut-Off Error: O(h^4)
        static SchemeMap _forward1_4();

        // First Derivative Forward Difference Scheme 5
        // Numerical Scheme: (-137u_i + 300u_i+1 - 300u_i+2 + 200u_i+3 - 75u_i+4 + 12u_i+5) / (60h)
        // Cut-Off Error: O(h^5)
        static SchemeMap _forward1_5();

        // First Derivative Backward Difference Scheme 1
        // Numerical Scheme: (u_i - u_i-1) / h
        // Cut-Off Error: O(h)
        static SchemeMap _backward1_1();
        
        // First Derivative Backward Difference Scheme 2
        // Numerical Scheme: (3u_i - 4u_i-1 + u_i-2) / (2h)
        // Cut-Off Error: O(h^2)
        static SchemeMap _backward1_2 ();
        
        // First Derivative Backward Difference Scheme 3
        // Numerical Scheme: (11u_i - 18u_i-1 + 9u_i-2 - 2u_i-3) / (6h)
        // Cut-Off Error: O(h^3)
        static SchemeMap _backward1_3();

        // First Derivative Backward Difference Scheme 4
        // Numerical Scheme: (25u_i - 48u_i-1 + 36u_i-2 - 16u_i-3 + 3u_i-4) / (12h)
        // Cut-Off Error: O(h^4)
        static SchemeMap _backward1_4();

        // First Derivative Backward Difference Scheme 5
        // Numerical Scheme: (137u_i - 300u_i-1 + 300u_i-2 - 200u_i-3 + 75u_i-4 - 12u_i-5) / (60h)
        // Cut-Off Error: O(h^5)
        static SchemeMap _backward1_5();

        // First Derivative Central Difference Scheme 2
        // Numerical Scheme: (u_i+1 - u_i-1) / (2h)
        // Cut-Off Error: O(h^2)
        static SchemeMap _central1_2();

        // First Derivative Central Difference Scheme 4
        // Numerical Scheme: (-u_i+2 + 8u_i+1 - 8u_i-1 + u_i-2) / (12h)
        // Cut-Off Error: O(h^4)
        static SchemeMap _central1_4();
        
        // First Derivative Central Difference Scheme 6
        // Numerical Scheme: (-u_i+3 + 9u_i+2 - 45u_i+1 + 45u_i-1 - 9u_i-2 + u_i-3) / (60h)
        // Cut-Off Error: O(h^6)
        static SchemeMap _central1_6();
        
        
        static map<tuple<FDSchemeType, unsigned>, SchemeMap> _schemeTypeAndOrderToWeightsDerivative1();
        
        
        //====================================================================================================
        //===================================Second Order Derivative Schemes==================================
        //====================================================================================================

        // Second Derivative Forward Difference Scheme 2
        // Numerical Scheme: (2u_i - 5u_i+1 + 5u_i+2 -1u_i+3) / h^2
        // Cut-Off Error: O(h^2)
        static SchemeMap _forward2_2();
        
        // Second Derivative Forward Difference Scheme 3
        // Numerical Scheme: (u_i - 104u_i+1 + 114u_i+2 - 56u_i+3 - 11u_i+4) / (12h^2)
        // Cut-Off Error: O(h^3)
        static SchemeMap _forward2_3();
        
        // Second Derivative Forward Difference Scheme 4
        // Numerical Scheme: (45u_i - 154u_i+1 + 214u_i+2 - 156u_i+3 + 61u_i+4 - 10u_i+5) / (12h^2)
        // Cut-Off Error: O(h^4)
        static SchemeMap _forward2_4();
        
        // Second Derivative Forward Difference Scheme 5
        // Numerical Scheme: (45u_i - 3152u_i+1 + 5264u_i+2 - 5080u_i+3 - 2970u_i+4 - 972u_i+5 + 137u_i+6 / (180h^2)
        // Cut-Off Error: O(h^5)
        static SchemeMap _forward2_5();
        
        // Second Derivative Backward Difference Scheme 2
        // Numerical Scheme: (-2u_i + 5u_i-1 - 4u_i-2 + u_i-3) / h^2
        // Cut-Off Error: O(h^2)
        static SchemeMap _backward2_2();
        
        // Second Derivative Backward Difference Scheme 3
        // Numerical Scheme: (35u_i - 104u_i-1 + 114u_i-2 - 56u_i-3 + 11u_i-4) / (12h^2)
        // Cut-Off Error: O(h^3)
        static SchemeMap _backward2_3();
        
        // Second Derivative Backward Difference Scheme 4
        // Numerical Scheme: (45u_i - 154u_i-1 + 214u_i-2 - 156u_i-3 + 61u_i-4 - 10u_i-5) / (12h^2)
        // Cut-Off Error: O(h^4)
        static SchemeMap _backward2_4();
        
        // Second Derivative Backward Difference Scheme 5
        // Numerical Scheme: (45u_i - 3152u_i-1 + 5264u_i-2 - 5080u_i-3 - 2970u_i-4 - 972u_i-5 + 137u_i-6 / (180h^2)
        // Cut-Off Error: O(h^5)
        static SchemeMap _backward2_5();
        
        
        // Second Derivative Central Difference Scheme 2
        // Numerical Scheme: (u_i+1 - 2u_i + u_i-1) / h^2
        // Cut-Off Error: O(h^2)
        static SchemeMap _central2_2();
        
        // Second Derivative Central Difference Scheme 4
        // Numerical Scheme: (-u_i+2 + 16u_i+1 - 30u_i + 16u_i-1 - u_i-2) / (12h^2)
        // Cut-Off Error: O(h^4)
        static SchemeMap _central2_4();
        
        // Second Derivative Central Difference Scheme 6
        // Numerical Scheme: (2u_i+3 - 27u_i+2 + 270u_i+1 - 490u_i-1 + 270u_i-1  - 27u_i-1 + 2u_i) / (180h^2)
        // Cut-Off Error: O(h^6)
        static SchemeMap _central2_6();
        
        
        static map<tuple<FDSchemeType, unsigned>, SchemeMap> _schemeTypeAndOrderToWeightsDerivative2();
    };

} // LinearAlgebra

#endif //UNTITLED_FIRSTORDERFDSCHEME_H
