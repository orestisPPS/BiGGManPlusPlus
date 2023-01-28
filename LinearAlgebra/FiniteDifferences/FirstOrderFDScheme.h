//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FIRSTORDERFDSCHEME_H
#define UNTITLED_FIRSTORDERFDSCHEME_H

#include <map>
#include <tuple>
using namespace std;

namespace LinearAlgebra {
    //A class containing all the first order finite difference schemes up to Fifth order accuracy
    class FirstOrderDerivativeFDScheme {
    public:
        FirstOrderDerivativeFDScheme();
        
        //Backward Finite Difference Scheme 1
        //Number of backward points: 1
        //Error: O(h)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                    Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                    Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward1();
        
        //Backward Finite Difference Scheme 2
        //Number of backward points: 2
        //Error: O(h^2)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                   Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                   Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward2();
        
        //Backward Finite Difference Scheme 3
        //Number of backward points: 3
        //Error: O(h^3)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                   Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                   Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward3();
        
        //Backward Finite Difference Scheme 4
        //Number of backward points: 4
        //Error: O(h^4)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                   Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                   Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward4();
        
        //Backward Finite Difference Scheme 5
        //Number of backward points: 5
        //Error: O(h^5)
        //Output :  map<int,double, double, int>:
        // tuple<map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Backward5();
        
        
        //Forward Finite Difference Scheme 1
        //Number of forward points: 1
        //Error: O(h)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward1();
        
        //Forward Finite Difference Scheme 2
        //Number of forward points: 2
        //Error: O(h^2)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward2();
        
        //Forward Finite Difference Scheme 3
        //Number of forward points: 3
        //Error: O(h^3)
        //Output :  tuple<map<int,double, double, int>
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        
        static tuple<map<int,double>, double, int> Forward3();
        
        //Forward Finite Difference Scheme 4
        //Number of forward points: 4
        //Error: O(h^4)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        
        static tuple<map<int,double>, double, int> Forward4();
        
        //Forward Finite Difference Scheme 5
        //Number of forward points: 5
        //Error: O(h^5)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Forward5();
        
        //Central Finite Difference Scheme 1
        //Number of backward points: 1
        //Number of forward points: 1
        //Error: O(h^2)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Central2();
        
        //Central Finite Difference Scheme 2
        //Number of backward points: 2
        //Number of forward points: 2
        //Error: O(h^4)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Central4();
        
        //Central Finite Difference Scheme 3
        //Number of backward points: 3
        //Number of forward points: 3
        //Error: O(h^6)
        //Output :  tuple<map<int,double, double, int>:
        // map<int,double> : map of the coefficients of the scheme.
        //                         Key : 0 for central, -i for ith backward point, +i for ith forward point
        //                         Value : coefficient
        // double : step coefficient
        // int : step power
        static tuple<map<int,double>, double, int> Central6();
        
        
    };

} // LinearAlgebra

#endif //UNTITLED_FIRSTORDERFDSCHEME_H
