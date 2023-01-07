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

namespace LinearAlgebra {

    class Transformations {
        public:
            Transformations();
            static Array<double> translate(vector<double> &vector, Direction direction, double distance);
            static Array<double> scale(vector<double> &vector, Direction direction, double factor);
            static Array<double> rotate(vector<double> &vector, PhysicalSpaceEntities plane, double angle);
            static Array<double> shear(vector<double> &vector, Direction direction, PhysicalSpaceEntities plane, double shear);
            static Array<double> reflect(vector<double> &vector, Direction direction, PhysicalSpaceEntities plane);
            

        private:
            static vector<double> translateDirection1(vector<double> &vector, double amount);
            //Translates the given array by the given amount in direction 2.
            static vector<double> translateDirection2(vector<double> &vector, double amount);
            //Translates the given array by the given amount in direction 3.
            static vector<double> translateDirection3(vector<double> &vector, double amount);
            //Scales the given array by the given amount in direction 1.
            static vector<double> scaleDirection1(vector<double> &vector, double amount);
            //Scales the given array by the given amount in direction 2.
            static vector<double> scaleDirection2(vector<double> &vector, double amount);
            //Scales the given array by the given amount in direction 3.
            static vector<double> scaleDirection3(vector<double> &vector, double amount);
            //Rotates the given array by the given angle[o] in direction 1.
            static vector<double> rotateDirection1(vector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] in direction 2.
            static vector<double> rotateDirection2(vector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] in direction 3.
            static vector<double> rotateDirection3(vector<double> &vector, double angle);
            //Shears the given array by the given angle[o] in direction 1 in the 1-2 plane.
            static vector<double> shearDirection1_plane12(vector<double> &vector, double angle);
            //Shears the given array by the given amount in direction 2 in the 1-2 plane.
            static vector<double> shearDirection2_plane12(vector<double> &vector, double angle);
            //Shears the given array by the given amount in direction 1 in the 1-3 plane.
            static vector<double> shearDirection1_plane13(vector<double> &vector, double angle);
            //Shears the given array by the given amount in direction 3 in the 1-3 plane.
            static vector<double> shearDirection3_plane13(vector<double> &vector, double angle);
            //Shears the given array by the given amount in direction 2 in the 2-3 plane.
            static vector<double> shearDirection2_plane23(vector<double> &vector, double angle);
            //Shears the given array by the given amount in direction 3 in the 2-3 plane.
            static vector<double> shearDirection3_plane23(vector<double> &vector, double angle);
            //Reflects the given array about the origin in direction 1 in the 1-2 plane.
            static vector<double> reflectDirection1_plane12(vector<double> &vector, double angle);
            //Reflects the given array about the origin in direction 2 in the 1-2 plane.
            static vector<double> reflectDirection2_plane12(vector<double> &vector, double angle);
            //Reflects the given array about the origin in direction 1 in the 1-3 plane.
            static vector<double> reflectDirection1_plane13(vector<double> &vector, double angle);
            //Reflects the given array about the origin in direction 3 in the 1-3 plane.
            static vector<double> reflectDirection3_plane13(vector<double> &vector, double angle);
            //Reflects the given array about the origin in direction 2 in the 2-3 plane.
            static vector<double> reflectDirection2_plane23(vector<double> &vector, double angle);
            //Reflects the given array about the origin in direction 3 in the 2-3 plane.
            static vector<double> reflectDirection3_plane23(vector<double> &vector, double angle);
            
            
            
            

    };

} // LinearAlgebra

#endif //UNTITLED_TRANSFORMATIONS_H
