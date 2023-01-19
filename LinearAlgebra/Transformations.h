//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_TRANSFORMATIONS_H
#define UNTITLED_TRANSFORMATIONS_H

#include <vector>
#include <cmath>
#include <limits>
#include "Array.h"
#include "../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
#include "../Utility/Calculators.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace LinearAlgebra {

    class Transformations {
        //Source1: https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg
        //Source2 :https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        public:
            Transformations();

            //Translates the input vector in the 3d space by an input amount in the given direction.
            //distance1,2,3 : Amount to translate in the direction of axis 1,2,3 respectively.
            static vector<double> &translate(vector<double> &vector, double &distance1, double &distance2, double &distance3);

            //Translates the input vector in the input amount in the given direction.
            //distance : Amount to translate.
            //direction : Direction to translate in. (Axis1, Axis2, Axis3)
            static vector<double> &translateInDirection(vector<double> &vector, double &distance, PhysicalSpaceEntities &direction);
            
            //Scales the input vector in the 3d space by an input amount in the given direction.
            //amount1,2,3 : Amount to scale in the direction of axis 1,2,3 respectively.
            static vector<double> &scale(vector<double> &vector, double &amount1, double &amount2, double &amount3);

            //Translates the input vector in the input amount in the given direction.
            //amount : Amount to scale.
            //direction : Direction to translate in. (Axis1, Axis2, Axis3)
            static vector<double> &scaleInDirection(vector<double> &vector, double &amount, PhysicalSpaceEntities &direction);
            

            //Rotates the input vector in the 3d space by an input angle [deg] around all three axis.
            //angle1,2,3 : Amount to rotate around axis 1,2,3 respectively.
            static vector<double> &rotate(vector<double> &vector, double &angle1, double &angle2, double &angle3);

            //Rotates the input vector around the input axis by the given angle.
            //angle : Angle to rotate.
            //Axis : Axis to rotate around. (Axis1, Axis2, Axis3)
            static vector<double> &rotateAroundAxis(vector<double> &vector, double &angle, PhysicalSpaceEntities &axis);
            
            //Shears the input vector in the 3d space by an input angle in the given plane.
            //angle12,13,23 : Amount to shear in the 12,23,13 planes respectively.
            static vector<double> &shear(vector<double> &vector, double &angle12, double &angle23, double &angle13);


            //Shears the input vector in the input plane by the given angles.
            //angle1 : First angle to shear.
            //angle1 : Second angle to shear.
            //plane : plane to shear on. (plane12, plane23, plane 13)
            static vector<double> &shearPlane(vector<double> &vector, double &angle1, double &angle2, PhysicalSpaceEntities &plane);
            
            //Reflects the input vector in the 3d space.
            //Axis1,2,3 : true if the vector should be reflected in the direction of axis 1,2,3 respectively. false otherwise.
            static vector<double> &reflect(vector<double> &vector, bool &axis1, bool &axis2, bool &axis3);
            
            //Reflects the input vector around the given plane.
            static vector<double> &reflect(vector<double> &vector, PhysicalSpaceEntities &plane);
            
            
        private:
            //private property that translates a vector with 3 components at all 3 directions.
            static vector<double> _translate(vector<double> &vector, double translation1, double translation2, double translation3);
            //private property that scales a vector[3] at all 3 directions.
            static vector<double> _scale(vector<double> &vector, double scale1, double scale2, double scale3);
            //Rotates the given array by the given angle[o] around axis 1.
            static vector<double> _rotateAroundAxis1(vector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] around axis 2.
            static vector<double> _rotateAroundAxis2(vector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] around axis 3.
            static vector<double> _rotateAroundAxis3(vector<double> &vector, double angle);
            //Shears the given array by the given angles[o] in direction 1 and 2 in the 1-2 plane.
            static vector<double> _shearPlane12(vector<double> &vector, double angle1, double angle2);
            //Shears the given array by the given angles[o] in direction 1 and 3 in the 1-3 plane.
            static vector<double> _shearPlane13(vector<double> &vector, double angle1, double angle3);
            //Shears the given array by the given angles[o] in direction 2 and 3 in the 2-3 plane.
            static vector<double> _shearPlane23(vector<double> &vector, double angle2, double angle3);
            //Reflects the given array about axis1.
            static vector<double> _reflectAboutAxis1(vector<double> &vector);
            //Reflects the given array about axis2.
            static vector<double> _reflectAboutAxis2(vector<double> &vector);
            //Reflects the given array about axis3.
            static vector<double> _reflectAboutAxis3(vector<double> &vector);
            //Reflects the given array about the 1-2 plane.
            static vector<double> _reflectAboutPlane12(vector<double> &vector);
            //Reflects the given array about the 1-3 plane.
            static vector<double> _reflectAboutPlane13(vector<double> &vector);
            //Reflects the given array about the 2-3 plane.
            static vector<double> _reflectAboutPlane23(vector<double> &vector);
            //Reflects the given array about the origin.
            static vector<double> _reflectAboutOrigin(vector<double> &vector);
            
            
            
            
            

    };

} // LinearAlgebra

#endif //UNTITLED_TRANSFORMATIONS_H
