//
// Created by hal9000 on 5/13/23.
//

#include "DomainBoundaryFactory.h"

namespace StructuredMeshGenerator {
    
    DomainBoundaryFactory::DomainBoundaryFactory(Mesh* mesh) : _mesh(mesh) { }
    
    DomainBoundaryConditions* DomainBoundaryFactory::parallelogram(unsigned nnX, unsigned nnY, double lengthX, double lengthY,
                                                                   double rotAngle, double shearX, double shearY) {
        
        auto stepX = lengthX / (nnX - 1.0);
        auto stepY = lengthY / (nnY - 1.0);
        vector<double> coordinateVector(2, 0);
        auto boundaryConditionsSet = new map<Position, map<unsigned, BoundaryCondition *>>();

        for (auto &boundary: *_mesh->boundaryNodes) {
            
            boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(
                    boundary.first, map<unsigned, BoundaryCondition *>()));\
                    
            switch (boundary.first) {
                case Bottom:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = node->coordinates.positionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = 0.0;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = new map<DOFType, double>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        auto bc =
                                boundaryConditionsSet->at(boundary.first).insert(pair<unsigned, BoundaryCondition *>(
                                        *node->id.global, new BoundaryCondition(Dirichlet,  dofBC)));
                    }
                    break;
                case Right:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = node->coordinates.positionVector(Parametric);
                        coordinateVector[0] = lengthX;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = new map<DOFType, double>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        auto bc = 
                        boundaryConditionsSet->at(boundary.first).insert(pair<unsigned, BoundaryCondition *>(
                                *node->id.global, new BoundaryCondition(Dirichlet,  dofBC)));
                    }
                    break;
                case Top:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = node->coordinates.positionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = lengthY;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = new map<DOFType, double>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        auto bc =
                                boundaryConditionsSet->at(boundary.first).insert(pair<unsigned, BoundaryCondition *>(
                                        *node->id.global, new BoundaryCondition(Dirichlet,  dofBC)));
                    }
                    break;
                case Left:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = node->coordinates.positionVector(Parametric);
                        coordinateVector[0] = 0.0;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = new map<DOFType, double>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        auto bc =
                                boundaryConditionsSet->at(boundary.first).insert(pair<unsigned, BoundaryCondition *>(
                                        *node->id.global, new BoundaryCondition(Dirichlet,  dofBC)));
                    }
                    break;
                default:
                    throw runtime_error("A parallelogram can only have bottom, right, top, and left boundaries.");
            }
        }
        return new DomainBoundaryConditions(boundaryConditionsSet);
    }
    
    DomainBoundaryConditions* DomainBoundaryFactory::parallelepiped(unsigned nnX, unsigned nnY, unsigned nnZ,
                                                                     double stepX, double stepY, double stepZ,
                                                                     double rotAngleX, double rotAngleY, double rotAngleZ,
                                                                     double shearX, double shearY, double shearZ) {

    }
} // StructuredMeshGenerator