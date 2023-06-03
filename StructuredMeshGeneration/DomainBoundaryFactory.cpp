//
// Created by hal9000 on 5/13/23.
//

#include "DomainBoundaryFactory.h"

namespace StructuredMeshGenerator {
    
    DomainBoundaryFactory::DomainBoundaryFactory(Mesh* mesh) : _mesh(mesh), _domainBoundaryConditions(nullptr) {
    }
    
    DomainBoundaryFactory::~DomainBoundaryFactory() {
        delete _domainBoundaryConditions;
    }
    
    void DomainBoundaryFactory::parallelogram(map<Direction, unsigned>& nodesPerDirection, double lengthX, double lengthY,
                                              double rotAngle, double shearX, double shearY) {
        auto nnX = nodesPerDirection[One];
        auto nnY = nodesPerDirection[Two];
        
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
                        boundaryConditionsSet->at(boundary.first).insert(pair<unsigned, BoundaryCondition *>(
                                        *node->id.global, new BoundaryCondition(Dirichlet,  dofBC)));
                    }
                    break;
                default:
                    throw runtime_error("A parallelogram can only have bottom, right, top, and left boundaries.");
            }
        }
        _domainBoundaryConditions = new  DomainBoundaryConditions(boundaryConditionsSet);
    }
    
    void DomainBoundaryFactory::parallelepiped(map<Direction, unsigned>& nodesPerDirection,
                                                 double stepX, double stepY, double stepZ,
                                                 double rotAngleX, double rotAngleY, double rotAngleZ,
                                                 double shearX, double shearY, double shearZ) {

    }
    
    

    DomainBoundaryConditions *DomainBoundaryFactory::getDomainBoundaryConditions() const {
        return _domainBoundaryConditions;
    }

    void DomainBoundaryFactory::ellipse(map<Direction, unsigned int> &nodesPerDirection, double radius1, double radius2) {
        auto pi = acos(-1.0);
        auto theta = 0.0;
        unsigned numberOfNodes = 0;
        double stepTheta;
        vector<double> coordinateVector(2, 0);

        auto boundaryConditionsSet = new map<Position, map<unsigned, BoundaryCondition *>>();

        for (auto& boundary : *_mesh->boundaryNodes) {
            boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(
                    boundary.first, map<unsigned, BoundaryCondition *>()));
            switch (boundary.first) {
                case Bottom:
                    theta = 5.0 * pi / 4.0;
                    numberOfNodes = nodesPerDirection[One];
                    break;
                case Right:
                    theta = 7.0 * pi / 4.0;
                    numberOfNodes = nodesPerDirection[Two];
                    break;
                case Top:
                    theta = 3.0 * pi / 4.0;  // Corrected theta value for the top boundary
                    numberOfNodes = nodesPerDirection[One];
                    break;
                case Left:
                    theta = pi / 4.0;  // Corrected theta value for the left boundary
                    numberOfNodes = nodesPerDirection[Two];
                    break;
                default:
                    throw runtime_error("An ellipse can only have bottom, right, top, and left boundaries.");
            }
            stepTheta = (pi / 2.0) / (numberOfNodes - 1);
            for (int i = 0; i < numberOfNodes; i++) {
                auto dofBC = new map<DOFType, double>();
                dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta + i * stepTheta)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta + i * stepTheta)));
                boundaryConditionsSet->at(boundary.first).insert(
                        pair<unsigned, BoundaryCondition *>(*boundary.second->at(i)->id.global,
                                                            new BoundaryCondition(Dirichlet, dofBC)));
            }
        }
        _domainBoundaryConditions = new DomainBoundaryConditions(boundaryConditionsSet);
    }




} // StructuredMeshGenerator