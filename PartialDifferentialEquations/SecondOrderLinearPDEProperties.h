//
// Created by hal9000 on 12/6/22.
//

#include "../Primitives/Matrix.h"
#include "../Discretization/Node/Node.h"
#include "iostream"
#include "vector"
#include "map"
using namespace std;

using namespace Primitives;
using namespace Discretization;

namespace PartialDifferentialEquations {
    enum PropertiesDistributionType
    {
        Isotropic,
        FieldAnisotropic,
        LocallyAnisotropic
    };
    class SecondOrderLinearPDEProperties {
    public :
        SecondOrderLinearPDEProperties(Matrix<double> *secondOrderCoefficients,
                                       vector<double> *firstOrderCoefficients,
                                       double *zerothOrderCoefficient,
                                       double *sourceTerm, bool *isTransient);
            
        SecondOrderLinearPDEProperties(map<int*, Matrix<double>> *secondOrderCoefficients,
                                       map<int*, vector<double>> *firstOrderCoefficients,
                                       map<int*, double> *zerothOrderCoefficients,
                                       map<int*, double> *sourceTerms, bool *isTransient);
        
        PropertiesDistributionType Type();
        
        bool IsTransient();

        template<class T>
        T *SecondOrderCoefficients();
        
        template<class T>
        T *FirstOrderCoefficients();
        
        template<class T>
        T *ZerothOrderCoefficient();
        
        template<class T>
        T *SourceTerm();

        
    
    private:
        PropertiesDistributionType _type;
        bool *_isTransient;
        
        Matrix<double> *_secondDerivativeIsotropicProperties;
        vector<double> *_firstDerivativeIsotropicProperties;
        double *_zeroDerivativeIsotropicProperties;
        double *_sourceProperties;
                
        map<int*, Matrix<double>> *_secondDerivativeLocallyAnisotropicProperties;
        map<int*, vector<double>> *_firstDerivativeLocallyAnisotropicProperties;
        map<int*, double> *_zeroDerivativeLocallyAnisotropicProperties;
        map<int*, double> *_sourceLocallyAnisotropicProperties;
    };

} // PartialDifferentialEquations
