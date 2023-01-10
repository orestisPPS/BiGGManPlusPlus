//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_VOLUME_H
#define UNTITLED_VOLUME_H

#include "PhysicalSpaceEntity.h"

namespace PositioningInSpace {

    class Volume : PhysicalSpaceEntity {
    public:
        Volume(PhysicalSpaceEntities type);
        const PhysicalSpaceEntities& type() override;
        
    private:
        PhysicalSpaceEntities _type;
        static bool _checkInput(PhysicalSpaceEntities type);
    };

} // PositioningInSpace

#endif //UNTITLED_VOLUME_H
