//
// Created by hal9000 on 12/17/22.
//

#include "MeshSpecs2D.h"

namespace StructuredMeshGenerator {
    MeshSpecs2D::MeshSpecs2D(int nnx, int nny, double templateHx, double templateHy,
                             double templateRotAngle,
                             double templateShearX, double templateShearY) {
        this->nnx = nnx;
        this->nny = nny;
        this->templateHx = templateHx;
        this->templateHy = templateHy;
        this->templateRotAngle = templateRotAngle;
        this->templateShearX = templateShearX;
        this->templateShearY = templateShearY;
    }

} // StructuredMeshGenerator