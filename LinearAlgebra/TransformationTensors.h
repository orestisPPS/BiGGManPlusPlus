//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_TRANSFORMATIONTENSORS_H
#define UNTITLED_TRANSFORMATIONTENSORS_H

#include <vector>
#include <cmath>
#include <limits>
#include "Array.h"
#include "../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
#include "../Utility/Calculators.h"

using namespace std;

namespace LinearAlgebra {
    namespace Transformations{
        static vector<double> Rotate(vector<double> &array, double angleInDegrees, PhysicalSpace physicalSpace);
        
static vector<double> Shear(vector<double> &array, double shearAngleInDegrees, PositioningInSpace::Direction axis){
            vector<double> shearedArray;
            shearAngleInDegrees = Utility::Calculators::DegreesToRadians(shearAngleInDegrees);
            
            switch (array.size()) {
                case 2:
                    if (axis == 0)
                        shearedArray.push_back()
                case 3:
                    if (axis == 0) {
                        shearedArray.push_back(array[0]);
                        shearedArray.push_back(array[1] + array[0] * tan(shearAngleInDegrees));
                        shearedArray.push_back(array[2]);
                    } else if (axis == 1) {
                        shearedArray.push_back(array[0] + array[1] * tan(shearAngleInDegrees));
                        shearedArray.push_back(array[1]);
                        shearedArray.push_back(array[2]);
                    } else if (axis == 2) {
                        shearedArray.push_back(array[0]);
                        shearedArray.push_back(array[1]);
                        shearedArray.push_back(array[2] + array[0] * tan(shearAngleInDegrees));
                    }
                    break;
                default:
                    //return NaN vector
                    for (int i = 0; i < array.size(); i++) {
                        shearedArray.push_back(numeric_limits<double>::quiet_NaN());
                    }
            }
            return shearedArray;
        }
        
        
        
    }

    static class TransformationTensors {
    public:
        TransformationTensors();

        static vector<double> Shear(vector<double>, double shearAngleInDegrees);
        static vector<double> Scale(vector<double>, double stepOne);
        static vector<double> Scale(vector<double>, double stepOne, double stepTwo);
        static vector<double> Scale(vector<double>, double stepOne, double stepTwo, double stepThree);
        static vector<double> MirrorOne(vector<double>);
        static vector<double> MirrorTwo(vector<double>);
        static vector<double> MirrorThree(vector<double>);
        static vector<double> Translate(vector<double>, double stepOne);
        static vector<double> Translate(vector<double>, double stepOne, double stepTwo);
        static vector<double> Translate(vector<double>, double stepOne, double stepTwo, double stepThree);
    };

    };// LinearAlgebra

#endif //UNTITLED_TRANSFORMATIONTENSORS_H
