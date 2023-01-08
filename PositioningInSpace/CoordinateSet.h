//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_COORDINATESET_H
#define UNTITLED_COORDINATESET_H

#include <vector>
#include "SpaceCharacteristics.h"
#include "PhysicalSpaceEntities/PhysicalSpaceEntity.h"
using namespace std;
using namespace PositioningInSpace;

namespace PositioningInSpace {

    class CoordinateSet {
    public:
        vector<double> getCoordinateVector(PhysicalSpaceEntity &physicalSpace);
        vector<double> getCoordinateVector();
        void setCordinateVector(vector<double> coordinateVector, PhysicalSpaceEntity &physicalSpace);
        unsigned dimensions();
    private:
        vector<double> *_values;
        
    };

} // PositioningInSpace

#endif //UNTITLED_COORDINATESET_H
