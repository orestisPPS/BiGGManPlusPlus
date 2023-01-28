//
// Created by hal9000 on 1/28/23.
//

#include "FirstOrderFDScheme.h"

namespace LinearAlgebra {
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Backward1() {
        map<int,double> coefficients;
        coefficients[0] = 1;
        coefficients[-1] = -1;
        return {coefficients, 1, 1};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Backward2() {
        map<int,double> coefficients;
        coefficients[0] = 3;
        coefficients[-1] = -4;
        coefficients[-2] = 1;
        return {coefficients, 2, 2};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Backward3() {
        map<int,double> coefficients;
        coefficients[0] = 11;
        coefficients[-1] = -18;
        coefficients[-2] = 9;
        coefficients[-3] = -2;
        return {coefficients, 6, 3};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Backward4() {
        map<int,double> coefficients;
        coefficients[0] = 25;
        coefficients[-1] = -48;
        coefficients[-2] = 36;
        coefficients[-3] = -16;
        coefficients[-4] = 3;
        return {coefficients, 12, 4};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Backward5() {
        map<int,double> coefficients;
        coefficients[0] = 137;
        coefficients[-1] = -300;
        coefficients[-2] = 300;
        coefficients[-3] = -200;
        coefficients[-4] = 75;
        coefficients[-5] = -12;
        return {coefficients, 60, 5};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Forward1() {
        map<int,double> coefficients;
        coefficients[0] = 1;
        coefficients[1] = -1;
        return {coefficients, 1, 1};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Forward2() {
        map<int,double> coefficients;
        coefficients[0] = 3;
        coefficients[1] = -4;
        coefficients[2] = 1;
        return {coefficients, 2, 2};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Forward3() {
        map<int,double> coefficients;
        coefficients[0] = 11;
        coefficients[1] = -18;
        coefficients[2] = 9;
        coefficients[3] = -2;
        return {coefficients, 6, 3};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Forward4() {
        map<int,double> coefficients;
        coefficients[0] = 25;
        coefficients[1] = -48;
        coefficients[2] = 36;
        coefficients[3] = -16;
        coefficients[4] = 3;
        return {coefficients, 12, 4};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Forward5() {
        map<int,double> coefficients;
        coefficients[0] = 137;
        coefficients[1] = -300;
        coefficients[2] = 300;
        coefficients[3] = -200;
        coefficients[4] = 75;
        coefficients[5] = -12;
        return {coefficients, 60, 1};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Central2() {
        map<int,double> coefficients;
        coefficients[-1] = -1;
        coefficients[1] = 1;
        return {coefficients, 2, 1};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Central4() {
        map<int,double> coefficients;
        coefficients[-2] = 1;
        coefficients[-1] = -8;
        coefficients[1] = 8;
        coefficients[2] = -1;
        return {coefficients, 12, 1};
    }
    
    tuple<map<int,double>, double, int> FirstOrderDerivativeFDScheme:: Central6() {
        map<int,double> coefficients;
        coefficients[-3] = -1;
        coefficients[-2] = 9;
        coefficients[-1] = -45;
        coefficients[1] = 45;
        coefficients[2] = -9;
        coefficients[3] = 1;
        return {coefficients, 60, 1};
    }
    
    
} // LinearAlgebra