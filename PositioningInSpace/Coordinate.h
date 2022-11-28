//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_COORDINATE_H
#define UNTITLED_COORDINATE_H

#include <tuple>
#include "DirectionsPositions.h"

namespace PositioningInSpace{
    enum CoordinateType
    {
        Natural,
        Parametric,
        Template
    };

    class Coordinate {
    public:

        Coordinate(CoordinateType type, Direction direction);

        Coordinate(CoordinateType type, Direction direction, double value);

        ~Coordinate();

        CoordinateType type();

        Direction direction();

        double value();

        void setValue(double value);

        bool operator == (const Coordinate& dof);

        bool operator != (const Coordinate& dof);

        void Print();

    private:
        CoordinateType _type;
        Direction _direction;
        double *_value;
    };
}



#endif //UNTITLED_COORDINATE_H
