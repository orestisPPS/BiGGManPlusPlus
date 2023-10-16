//
// Created by hal9000 on 5/5/23.
//

#ifndef UNTITLED_SPATIALPDEPROPERTIES_H
#define UNTITLED_SPATIALPDEPROPERTIES_H
#include "../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "PDEProperties/SecondOrderLinearPDEProperties.h"

namespace MathematicalEntities {

    struct SpatialScalarFieldPDEProperties {
        unique_ptr<NumericalVector<double>> firstOrderCoefficients;
        unique_ptr<NumericalMatrix<double>> secondOrderCoefficients;
        unique_ptr<double> zerothOrderCoefficient;
        unique_ptr<double> sourceTerm;
    };

    struct SpatialVectorFieldPDEProperties {
        unique_ptr<NumericalMatrix<double>> firstOrderCoefficients;
        unique_ptr<NumericalMatrix<double>> secondOrderCoefficients;
        unique_ptr<NumericalVector<double>> zerothOrderCoefficient;
        unique_ptr<double> sourceTerm;
    };

    struct TemporalVectorFieldPDECoefficients {
        unique_ptr<NumericalVector<double>> firstOrderCoefficients;
        unique_ptr<NumericalVector<double>> secondOrderCoefficients;
    };
    
    struct TemporalScalarFieldPDECoefficients {
        unique_ptr<double> firstOrderCoefficient;
        unique_ptr<double> secondOrderCoefficient;
    };
    
} // MathematicalEntities

#endif //UNTITLED_SPATIALPDEPROPERTIES_H
