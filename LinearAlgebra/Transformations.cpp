//
// Created by hal9000 on 1/7/23.
//

#include "Transformations.h"

namespace LinearAlgebra {
    Transformations::Transformations() = default;
    
    vector<double> Transformations::translate(vector<double> &vector, double &distance1, double &distance2, double &distance3) {
        vector = _translate(vector, distance1, distance2, distance3);
        return vector;
    }
    vector<double> Transformations::scale(vector<double> &vector, double &amount1, double &amount2, double &amount3){
        vector = _scale(vector, amount1, amount2, amount3);
        return vector;
    }
    
    vector<double> Transformations::rotate(vector<double> &vector, double &angle1, double &angle2, double &angle3){
        vector = _rotateAroundAxis1(vector, angle1);
        vector = _rotateAroundAxis2(vector, angle2);
        vector = _rotateAroundAxis3(vector, angle3);
        return vector;
    }
    
    vector<double> Transformations::shear(vector<double> &vector, double &angle12, double &angle23, double &angle13){
        vector = _shearPlane12(vector, angle12, angle13);
        vector = _shearPlane13(vector, angle12, angle13);
        vector = _shearPlane23(vector, angle23, angle13);
        return vector;
    }
    
    vector<double> Transformations::reflect(vector<double> &vector, bool &axis1, bool &axis2, bool &axis3){
        throw runtime_error("Not implemented");
        return vector;
    }
    vector<double> Transformations::reflect(vector<double> &vector, PhysicalSpaceEntities plane){
        throw runtime_error("Not implemented");
        return vector;
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