//
// Created by hal9000 on 4/13/23.
//

#include "FirstOrderDerivativeFiniteDifferenceScheme.h"

namespace LinearAlgebra {
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::FirstOrderFDScheme
    {
        public:
        // The order of the cut-off error of the scheme
        unsigned order;

        // The coefficient that scales the step size
        unsigned stepCoefficient;

        //Func that returns the function at the given index
        double derivativeValue;
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::forward1 : public FirstOrderFDScheme
    {
        forward1(double central, double next1, double stepSize) : FirstOrderFDScheme()
        {
            order = 1;
            stepCoefficient = 1;
            derivativeValue = value(central, next1, stepSize);
        }

        double value(double central, double next1, double stepSize)
        {
            return (next1 - central) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::forward2 : public FirstOrderFDScheme
    {
        forward2(double central, double next1, double next2, double stepSize) : FirstOrderFDScheme()
        {
            order = 2;
            stepCoefficient = 2;
            derivativeValue = value(central, next1, next2, stepSize);
        }

        double value(double central, double next1, double next2, double stepSize)
        {
            return (-3.0 * central + 4.0 * next1 - next2) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::forward3 : public FirstOrderFDScheme
    {
        forward3(double central, double next1, double next2, double next3, double stepSize) : FirstOrderFDScheme()
        {
            order = 3;
            stepCoefficient = 6;
            derivativeValue = value(central, next1, next2, next3, stepSize);
        }

        double value(double central, double next1, double next2, double next3, double stepSize)
        {
            return (2.0 * central - 9.0 * next1 + 18.0 * next2 - 11.0 * next3) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::forward4 : public FirstOrderFDScheme
    {
        forward4(double central, double next1, double next2, double next3, double next4, double stepSize) : FirstOrderFDScheme()
        {
            order = 4;
            stepCoefficient = 12;
            derivativeValue = value(central, next1, next2, next3, next4, stepSize);
        }

        double value(double central, double next1, double next2, double next3, double next4, double stepSize)
        {
            return (-5.0 * central + 18.0 * next1 - 24.0 * next2 + 14.0 * next3 - 3.0 * next4) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::forward5 : public FirstOrderFDScheme
    {
        forward5(double central, double next1, double next2, double next3, double next4, double next5, double stepSize) : FirstOrderFDScheme()
        {
            order = 5;
            stepCoefficient = 60;
            derivativeValue = value(central, next1, next2, next3, next4, next5, stepSize);
        }

        double value(double central, double next1, double next2, double next3, double next4, double next5, double stepSize)
        {
            return (3.0 * central - 16.0 * next1 + 36.0 * next2 - 48.0 * next3 + 25.0 * next4 - 5.0 * next5)
                   / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::backward1 : public FirstOrderFDScheme
    {
        backward1(double central, double previous1, double stepSize) : FirstOrderFDScheme()
        {
            order = 1;
            stepCoefficient = 1;
            derivativeValue = value(central, previous1, stepSize);
        }

        double value(double central, double previous1, double stepSize)
        {
            return (central - previous1) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::backward2 : public FirstOrderFDScheme
    {
        backward2(double central, double previous1, double previous2, double stepSize) : FirstOrderFDScheme()
        {
            order = 2;
            stepCoefficient = 2;
            derivativeValue = value(central, previous1, previous2, stepSize);
        }

        double value(double central, double previous1, double previous2, double stepSize)
        {
            return (3.0 * central - 4.0 * previous1 + previous2) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::backward3 : public FirstOrderFDScheme
    {
        backward3(double central, double previous1, double previous2, double previous3, double stepSize) : FirstOrderFDScheme()
        {
            order = 3;
            stepCoefficient = 6;
            derivativeValue = value(central, previous1, previous2, previous3, stepSize);
        }

        double value(double central, double previous1, double previous2, double previous3, double stepSize)
        {
            return (-2.0 * central + 9.0 * previous1 - 18.0 * previous2 + 11.0 * previous3) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::backward4 : public FirstOrderFDScheme
    {
        backward4(double central, double previous1, double previous2, double previous3, double previous4, double stepSize) : FirstOrderFDScheme()
        {
            order = 4;
            stepCoefficient = 12;
            derivativeValue = value(central, previous1, previous2, previous3, previous4, stepSize);
        }

        double value(double central, double previous1, double previous2, double previous3, double previous4, double stepSize)
        {
            return (5.0 * central - 18.0 * previous1 + 24.0 * previous2 - 14.0 * previous3 + 3.0 * previous4) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::backward5 : public FirstOrderFDScheme
    {
        backward5(double central, double previous1, double previous2, double previous3, double previous4, double previous5, double stepSize) : FirstOrderFDScheme()
        {
            order = 5;
            stepCoefficient = 60;
            derivativeValue = value(central, previous1, previous2, previous3, previous4, previous5, stepSize);
        }

        double value(double central, double previous1, double previous2, double previous3, double previous4, double previous5, double stepSize)
        {
            return (-3.0 * central + 16.0 * previous1 - 36.0 * previous2 + 48.0 * previous3 - 25.0 * previous4 + 5.0 * previous5)
                   / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::central2 : public FirstOrderFDScheme
    {
        central2(double previous1, double next1, double stepSize) : FirstOrderFDScheme()
        {
            order = 2;
            stepCoefficient = 2;
            derivativeValue = value(previous1, next1, stepSize);
        }

        double value(double previous1, double next1, double stepSize)
        {
            return (next1 - previous1) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::central4 : public FirstOrderFDScheme
    {
        central4(double previous2, double previous1, double next1, double next2, double stepSize) : FirstOrderFDScheme()
        {
            order = 4;
            stepCoefficient = 12;
            derivativeValue = value(previous2, previous1, next1, next2, stepSize);
        }

        double value(double previous2, double previous1, double next1, double next2, double stepSize)
        {
            return (-previous2 + 8.0 * previous1 - 8.0 * next1 + next2) / (stepCoefficient * stepSize);
        }
    };
    
    struct FirstOrderDerivativeFiniteDifferenceScheme::central6 : public FirstOrderFDScheme
    {
        central6(double previous3, double previous2, double previous1, double next1, double next2, double next3, double stepSize) : FirstOrderFDScheme()
        {
            order = 6;
            stepCoefficient = 60;
            derivativeValue = value(previous3, previous2, previous1, next1, next2, next3, stepSize);
        }

        double value(double previous3, double previous2, double previous1, double next1, double next2, double next3, double stepSize)
        {
            return (previous3 - 9.0 * previous2 + 45.0 * previous1 - 45.0 * next1 + 9.0 * next2 - next3) / (stepCoefficient * stepSize);
        }
    };


    
        

}// LinearAlgebra