//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_TRANSFORMATIONTENSORS_H
#define UNTITLED_TRANSFORMATIONTENSORS_H

#include <vector>
using namespace std;

namespace LinearAlgebra {

    static class TransformationTensors {
    public:
        TransformationTensors();
        static vector<double> Rotate(vector<double>, double angleInDegrees);
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
