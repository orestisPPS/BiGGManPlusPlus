//
// Created by hal9000 on 1/5/23.
//

#ifndef UNTITLED_SPACECHARACTERISTICS_H
#define UNTITLED_SPACECHARACTERISTICS_H
#include <map>
#include <iostream>
#include "DirectionsPositions.h"
namespace PositioningInSpace {

    class SpaceCharacteristics {
        public:
            explicit SpaceCharacteristics(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection, CoordinateSystem coordinateSystem);
            unsigned Dimensions;
            PhysicalSpaceEntities physicalSpace;
            CoordinateSystem coordinateSystem;
            
        private:
            static void CheckInput(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection);
            void CalculateDimensions();
            void CalculatePhysicalSpace(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection);
        };
    }; // PositioningInSpace

#endif //UNTITLED_SPACECHARACTERISTICS_H
