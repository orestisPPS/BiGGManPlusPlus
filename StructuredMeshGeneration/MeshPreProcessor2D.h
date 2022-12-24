//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MeshSpecs2D.h"
namespace StructuredMeshGenerator {
    
    class MeshPreProcessor2D {
    public:
        MeshPreProcessor2D(MeshSpecs2D *meshSpecs);
        ~MeshPreProcessor2D();
        
        MeshSpecs2D *meshSpecs;
    };

};