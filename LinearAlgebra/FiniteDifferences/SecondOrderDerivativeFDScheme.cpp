//
// Created by hal9000 on 1/28/23.
//

#include "SecondOrderDerivativeFDScheme.h"

namespace LinearAlgebra {
    
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Backward1() {
            map<int,double> coefficients;
            coefficients[0] = 1;
            coefficients[-1] = -2;
            coefficients[-2] = 1;
            return {coefficients, 1, 2};
        }
    
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Backward2() {
            map<int,double> coefficients;
            coefficients[0] = 2;
            coefficients[-1] = -5;
            coefficients[-2] = 4;
            coefficients[-3] = -1;
            return {coefficients, 1, 2};
        }
    
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Backward3() {
            map<int,double> coefficients;
            coefficients[0] = 35;
            coefficients[-1] = -104;
            coefficients[-2] = 114;
            coefficients[-3] = -56;
            coefficients[-4] = 11;
            return {coefficients, 12, 2};
        }
    
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Backward4() {
            map<int,double> coefficients;
            coefficients[0] = 45;
            coefficients[-1] = -154;
            coefficients[-2] = 214;
            coefficients[-3] = -156; 
            coefficients[-4] = 61;
            coefficients[-5] = -10;
            return {coefficients, 12, 2};
        }
    
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Backward5() {
            map<int,double> coefficients;
            coefficients[0] = 812;
            coefficients[-1] = -3132;
            coefficients[-2] = 5265;
            coefficients[-3] = -5080;
            coefficients[-4] = 2970;
            coefficients[-5] = -972;
            coefficients[-6] = 137;
            return {coefficients, 180, 2};
        }
            
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Forward1() {
            map<int,double> coefficients;
            coefficients[0] = 1;
            coefficients[1] = -2;
            coefficients[2] = 1;
            return {coefficients, 1, 2};
        }
        
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Forward2() {
            map<int,double> coefficients;
            coefficients[0] = 2;
            coefficients[1] = -5;
            coefficients[2] = 4;
            coefficients[3] = -1;
            return {coefficients, 1, 2};
        }
        
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Forward3() {
            map<int,double> coefficients;
            coefficients[0] = 35;
            coefficients[1] = -104;
            coefficients[2] = 114;
            coefficients[3] = -56;
            coefficients[4] = 11;
            return {coefficients, 12, 2};
        }
        
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Forward4() {
            map<int,double> coefficients;
            coefficients[0] = 45;
            coefficients[1] = -154;
            coefficients[2] = 214;
            coefficients[3] = -156;
            coefficients[4] = 61;
            coefficients[5] = -10;
            return {coefficients, 12, 2};
        }
        
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Forward5() {
            map<int,double> coefficients;
            coefficients[0] = 812;
            coefficients[1] = -3132;
            coefficients[2] = 5265;
            coefficients[3] = -5080;
            coefficients[4] = 2970;
            coefficients[5] = -972;
            coefficients[6] = 137;
            return {coefficients, 180, 2};
        }
        
        
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Central2() {
            map<int,double> coefficients;
            coefficients[-1] = 1;
            coefficients[0] = -2;
            coefficients[1] = 1;
            return {coefficients, 2, 2};
        }
        
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Central4() {
            map<int,double> coefficients;
            coefficients[-2] = 1;
            coefficients[-1] = -16;
            coefficients[0] = -30;
            coefficients[1] = 16;
            coefficients[2] = -1;
            return {coefficients, 12, 2};
        }
        
        tuple<map<int,double>, double, int> SecondOrderDerivativeFDScheme:: Central6() {
            map<int,double> coefficients;
            coefficients[-3] = 2;
            coefficients[-2] = -27;
            coefficients[-1] = 270;
            coefficients[0] = -490;
            coefficients[1] = 270;
            coefficients[2] = -27;
            coefficients[3] = 2;
            return {coefficients, 180, 2};
        }
        
} // LinearAlgebra