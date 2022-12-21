//
// Created by hal9000 on 12/17/22.
//

#include "MeshSpecs2D.h"
namespace StructuredMeshGenerator {
    
    class MeshPreProcessor2D {
    public:
        MeshPreProcessor2D(MeshSpecs2D *meshSpecs);
        ~MeshPreProcessor2D();
        
        MeshSpecs2D *meshSpecs;
        Mesh2D *mesh;

    };

    };

} // StructuredMeshGenerator

#endif //UNTITLED_MESHPREPROCESSOR2D_H
