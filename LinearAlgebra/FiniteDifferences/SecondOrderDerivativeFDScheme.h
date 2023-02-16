//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_SECONDORDERDERIVATIVEFDSCHEME_H
#define UNTITLED_SECONDORDERDERIVATIVEFDSCHEME_H

#include <tuple>
#include <map>
#include "FDScheme.h"

using namespace std;

namespace LinearAlgebra {

    class SecondOrderDerivativeFDScheme : public FDScheme {
public:
        SecondOrderDerivativeFDScheme();
        
        
        //Second Derivative Backward Finite Difference Scheme of order 1
        //Number of backward points: 2
        //Error: O(h)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward1();
                
        //Second Derivative Backward Finite Difference Scheme of order 2
        //Number of backward points: 3
        //Error: O(h^2)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward2();
        
        //Second Derivative Backward Finite Difference Scheme of order 3
        //Number of backward points: 4
        //Error: O(h^3)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward3();
        
        //Second Derivative Backward Finite Difference Scheme of order 4
        //Number of backward points: 5
        //Error: O(h^4)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward4();
        
        //Second Derivative Backward Finite Difference Scheme of order 5
        //Number of backward points: 6
        //Error: O(h^5)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward5();
        
        //Second Derivative Backward Finite Difference Scheme of order 6
        //Number of backward points: 7
        //Error: O(h^6)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward6();


        //Second Derivative Forward Finite Difference Scheme of order 1
        //Number of forward points: 2
        //Error: O(h)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward1();
        
        //Second Derivative Forward Finite Difference Scheme of order 2
        //Number of forward points: 3
        //Error: O(h^2)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward2();
        
        //Second Derivative Forward Finite Difference Scheme of order 3
        //Number of forward points: 4
        //Error: O(h^3)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward3();
        
        //Second Derivative Forward Finite Difference Scheme of order 4
        //Number of forward points: 5
        //Error: O(h^4)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward4();
        
        //Second Derivative Forward Finite Difference Scheme of order 5
        //Number of forward points: 6
        //Error: O(h^5)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward5();
        
        //Second Derivative Forward Finite Difference Scheme of order 6
        //Number of forward points: 7
        //Error: O(h^6)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward6();
        
        //Second Derivative Central Finite Difference Scheme of order 2
        //Number of backward points: 1
        //Number of forward points: 1
        //Error: O(h^2)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Central2();
            
        //Second Derivative Central Finite Difference Scheme of order 4
        //Number of backward points: 2
        //Number of forward points: 2
        //Error: O(h^4)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Central4();
        
        //Second Derivative Central Finite Difference Scheme of order 6
        //Number of backward points: 3
        //Number of forward points: 3
        //Error: O(h^6)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Central6();
        
        

    };

} // LinearAlgebra

#endif //UNTITLED_SECONDORDERDERIVATIVEFDSCHEME_H
