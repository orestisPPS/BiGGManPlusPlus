//
// Created by hal9000 on 5/13/23.
//

#ifndef UNTITLED_DOMAINBOUNDARYFACTORY_H
#define UNTITLED_DOMAINBOUNDARYFACTORY_H

#include "../MathematicalEntities/BoundaryConditions/DomainBoundaryConditions.h"
#include "MeshFactory.h"
#include "../LinearAlgebra/Operations/Transformations.h"
#include "../Utility/Calculators.h"
using namespace MathematicalEntities;

namespace StructuredMeshGenerator {

    class DomainBoundaryFactory {

    public:
        explicit DomainBoundaryFactory(const shared_ptr<Mesh> &mesh);


        shared_ptr<DomainBoundaryConditions>
        parallelogram(map<Direction, unsigned> &nodesPerDirection, double lengthX, double lengthY,
                      double rotAngle = 0, double shearX = 0, double shearY = 0);


        shared_ptr<DomainBoundaryConditions>
        ellipse(map<Direction, unsigned> &nodesPerDirection, double radius1, double radius2);


        shared_ptr<DomainBoundaryConditions>
        annulus_ripGewrgiou(map<Direction, unsigned int> &nodesPerDirection, double rIn, double rOut, double thetaStart,
                            double thetaEnd);


        shared_ptr<DomainBoundaryConditions>
        cavityBot(map<Direction, unsigned> &nodesPerDirection, double lengthX, double lengthY);


        shared_ptr<DomainBoundaryConditions>
        gasTankHorizontal(map<Direction, unsigned> &nodesPerDirection, double lengthX, double lengthY);


        shared_ptr<DomainBoundaryConditions>
        gasTankVertical(map<Direction, unsigned> &nodesPerDirection, double lenghtX, double lengthY);


        shared_ptr<DomainBoundaryConditions>
        sinusRiver(map<Direction, unsigned int> &nodesPerDirection, double lengthX, double bankDistanceY,
                   double amplitude, double frequency);


        shared_ptr<DomainBoundaryConditions> parallelepiped(map<Direction, unsigned> &nodesPerDirection,
                                                            double lengthX, double lengthY, double lengthZ,
                                                            double rotAngleX = 0, double rotAngleY = 0,
                                                            double rotAngleZ = 0,
                                                            double shearX = 0, double shearY = 0, double shearZ = 0);

        shared_ptr<DomainBoundaryConditions>
        annulus_3D_ripGewrgiou(map<Direction, unsigned int> &nodesPerDirection, double rIn, double rOut,
                               double thetaStart,
                               double thetaEnd, double height);


    private:
        shared_ptr<Mesh> _mesh;
    };

} // StructuredMeshGenerator

#endif //UNTITLED_DOMAINBOUNDARYFACTORY_H
