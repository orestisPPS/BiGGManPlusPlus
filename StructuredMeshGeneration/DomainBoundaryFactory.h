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
        explicit DomainBoundaryFactory(const shared_ptr<Mesh> &mesh);
        
        void parallelogram(map<Direction, unsigned>& nodesPerDirection, double lengthX, double lengthY,
                           double rotAngle = 0, double shearX = 0, double shearY = 0);

        void ellipse(map<Direction, unsigned> &nodesPerDirection, double radius1, double radius2);
        
        void parallelepiped(map<Direction, unsigned>& nodesPerDirection,
                                                 double stepX, double stepY, double stepZ,
                                                 double rotAngleX = 0, double rotAngleY = 0, double rotAngleZ = 0,
                                                 double shearX = 0, double shearY = 0, double shearZ = 0);

        void annulus_ripGewrgiou(map<Direction, unsigned int> &nodesPerDirection, double rIn, double rOut, double thetaStart,
                                 double thetaEnd);

        void cavityBot(map<Direction, unsigned>& nodesPerDirection, double lengthX, double lengthY);
        
        void gasTankHorizontal(map<Direction, unsigned>& nodesPerDirection, double lengthX, double lengthY);
        
        void gasTankVertical(map<Direction, unsigned>& nodesPerDirection, double lenghtX, double lengthY);

        void sinusRiver(map<Direction, unsigned int> &nodesPerDirection, double lengthX, double bankDistanceY, double amplitude, double frequency);
        
        shared_ptr<DomainBoundaryConditions> getDomainBoundaryConditions() const;
        
    private:
        shared_ptr<Mesh> _mesh;
        
        shared_ptr<DomainBoundaryConditions> _domainBoundaryConditions;


    };

} // StructuredMeshGenerator

#endif //UNTITLED_DOMAINBOUNDARYFACTORY_H
