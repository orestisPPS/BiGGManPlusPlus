//
// Created by hal9000 on 5/13/23.
//

#ifndef UNTITLED_DOMAINBOUNDARYFACTORY_H
#define UNTITLED_DOMAINBOUNDARYFACTORY_H

#include "../BoundaryConditions/DomainBoundaryConditions.h"
#include "../Discretization/Mesh/Mesh.h"
#include "../LinearAlgebra/Operations/Transformations.h"
using namespace BoundaryConditions;

namespace StructuredMeshGenerator {

    class DomainBoundaryFactory {
        
    public:
        DomainBoundaryFactory(Mesh* mesh);

        DomainBoundaryConditions* parallelogram(unsigned nnX, unsigned nnY, double lengthX, double lengthY,
                                                double rotAngle = 0, double shearX = 0, double shearY = 0);
        
        DomainBoundaryConditions* parallelepiped(unsigned nnX, unsigned nnY, unsigned nnZ,
                                                 double stepX, double stepY, double stepZ,
                                                 double rotAngleX = 0, double rotAngleY = 0, double rotAngleZ = 0,
                                                 double shearX = 0, double shearY = 0, double shearZ = 0);
        
    private:
        Mesh* _mesh;
    };

} // StructuredMeshGenerator

#endif //UNTITLED_DOMAINBOUNDARYFACTORY_H
