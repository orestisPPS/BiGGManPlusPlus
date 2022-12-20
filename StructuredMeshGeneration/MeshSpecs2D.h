//
// Created by hal9000 on 12/17/22.
//

#ifndef UNTITLED_MESHSPECS2D_H
#define UNTITLED_MESHSPECS2D_H

namespace StructuredMeshGenerator {
    
    class MeshSpecs2D {
    public :
        MeshSpecs2D(int nnx, int nny, double templateHx, double templateHy,
                        double templateRotAngle,
                        double templateShearX, double templateShearY);
        ~MeshSpecs2D();
        
        int nnx, nny;
        double templateHx, templateHy, templateRotAngle, templateShearX, templateShearY;
        };
            
            
    
    };

#endif //UNTITLED_MESHSPECS2D_H
