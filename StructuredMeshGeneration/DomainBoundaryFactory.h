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
        explicit DomainBoundaryFactory(Mesh* mesh);
        
        ~DomainBoundaryFactory();

        void parallelogram(map<Direction, unsigned>& nodesPerDirection, double lengthX, double lengthY,
                           double rotAngle = 0, double shearX = 0, double shearY = 0);
        
        void parallelepiped(map<Direction, unsigned>& nodesPerDirection,
                                                 double stepX, double stepY, double stepZ,
                                                 double rotAngleX = 0, double rotAngleY = 0, double rotAngleZ = 0,
                                                 double shearX = 0, double shearY = 0, double shearZ = 0);
        
        DomainBoundaryConditions* getDomainBoundaryConditions() const;
        
    private:
        Mesh* _mesh;
        
        DomainBoundaryConditions* _domainBoundaryConditions;
    };

} // StructuredMeshGenerator

#endif //UNTITLED_DOMAINBOUNDARYFACTORY_H
