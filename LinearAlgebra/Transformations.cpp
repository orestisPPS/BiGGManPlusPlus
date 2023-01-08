//
// Created by hal9000 on 1/7/23.
//

#include "Transformations.h"

namespace LinearAlgebra {
    Transformations::Transformations() = default;
    
    vector<double> Transformations::translate(vector<double> &vector, Direction direction, double distance) {
        
    }
    
    vector<double> Transformations::_translate(vector<double> &vector, double translation1, double translation2, double translation3) {
        std::vector<double> result(vector.size());
        result[0] = vector[0] + translation1 * vector[2];
        result[1] = vector[1] + translation2 * vector[2];
        result[2] = vector[2] + translation3 * vector[0];
        return result;
    }
    
    vector<double> Transformations::_scale(vector<double> &vector, double scale1, double scale2, double scale3) {
        std::vector<double> result(vector.size());
        result[0] = vector[0] * scale1;
        result[1] = vector[1] * scale2;
        result[2] = vector[2] * scale3;
        return result;
    }
    
    vector<double> Transformations::_rotateAroundAxis1(vector<double> &vector, double angle) {
        std::vector<double> result(vector.size());
        angle = Utility::Calculators::DegreesToRadians(angle);
        result[0] =  vector[0];
        result[1] =  vector[1] * cos(angle) + vector[2] * sin(angle);
        result[2] = -vector[1] * sin(angle) + vector[2] * cos(angle);
        return result;
    }
    
    vector<double> Transformations::_rotateAroundAxis2(vector<double> &vector, double angle) {
        std::vector<double> result(vector.size());
        angle = Utility::Calculators::DegreesToRadians(angle);
        result[0] = vector[0] * cos(angle) - vector[2] * sin(angle);
        result[1] = vector[1];
        result[2] = vector[0] * sin(angle) + vector[2] * cos(angle);
        return result;
    }
    
    vector<double> Transformations::_rotateAroundAxis3(vector<double> &vector, double angle) {
        std::vector<double> result(vector.size());
        angle = Utility::Calculators::DegreesToRadians(angle);
        result[0] = vector[0] * cos(angle) - vector[1] * sin(angle);
        result[1] = vector[0] * sin(angle) + vector[1] * cos(angle);
        result[2] = vector[2];
        return result;
    }
    
    vector<double> Transformations::_shearPlane12(vector<double> &vector, double shear1, double shear2) {
        std::vector<double> result(vector.size());
        shear1 = Utility::Calculators::DegreesToRadians(shear1);
        shear2 = Utility::Calculators::DegreesToRadians(shear2);
        result[0] = vector[0] + shear1 * vector[1];
        result[1] = vector[1] + shear2 * vector[0];
        result[2] = vector[2];
        return result;
    }
    
    vector<double> Transformations::_shearPlane13(vector<double> &vector, double shear1, double shear3) {
        std::vector<double> result(vector.size());
        shear1 = Utility::Calculators::DegreesToRadians(shear1);
        shear3 = Utility::Calculators::DegreesToRadians(shear3);
        result[0] = vector[0] + shear1 * vector[2];
        result[1] = vector[1];
        result[2] = vector[2] + shear3 * vector[0];
        return result;
    }
    
    vector<double> Transformations::_shearPlane23(vector<double> &vector, double shear2, double shear3) {
        std::vector<double> result(vector.size());
        shear2 = Utility::Calculators::DegreesToRadians(shear2);
        shear3 = Utility::Calculators::DegreesToRadians(shear3);
        result[0] = vector[0];
        result[1] = vector[1] + shear2 * vector[2];
        result[2] = vector[2] + shear3 * vector[1];
        return result;
    }
    
    vector<double> Transformations::_reflectAboutAxis1(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] =  vector[1];
        result[2] =  vector[2];
        return result;
    }
    
    vector<double> Transformations::_reflectAboutAxis2(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] =  vector[0];
        result[1] = -vector[1];
        result[2] =  vector[2];
        return result;
    }
    
    vector<double> Transformations::_reflectAboutAxis3(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] =  vector[0];
        result[1] =  vector[1];
        result[2] = -vector[2];
        return result;
    }
    
    vector<double> Transformations::_reflectAboutPlane12(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] =  vector[0];
        result[1] = -vector[1];
        result[2] = -vector[2];
        return result;
    }
    
    vector<double> Transformations::_reflectAboutPlane13(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] =  vector[1];
        result[2] = -vector[2];
        return result;
    }
    
    vector<double> Transformations::_reflectAboutPlane23(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] = -vector[1];
        result[2] =  vector[2];
        return result;
    }
    
    vector<double> Transformations::_reflectAboutOrigin(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] = -vector[1];
        result[2] = -vector[2];
        return result;
    }
    
    

    
} // LinearAlgebra