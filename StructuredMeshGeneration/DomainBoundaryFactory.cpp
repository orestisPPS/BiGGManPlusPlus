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

/*    void DomainBoundaryFactory::ellipse(map<Direction, unsigned int> &nodesPerDirection, double radius1, double radius2) {
        const double pi = acos(-1.0);
        double theta = 0.0;
        unsigned int numberOfNodes = 0;
        double stepTheta = 0.0;
        vector<double> coordinateVector(2, 0.0);
        auto sign = 1;
        auto boundaryConditionsSet = new map<Position, map<unsigned, BoundaryCondition *>>();

        for (auto& boundary : *_mesh->boundaryNodes) {
            boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(
                    boundary.first, map<unsigned, BoundaryCondition *>()));
            switch (boundary.first) {
                case Bottom:
                    numberOfNodes = nodesPerDirection[One];
                    theta = 5.0 * pi / 4.0;
                    stepTheta = (pi / 2.0) / (numberOfNodes - 1);
                    sign = 1;
                    break;
                case Right:
                    numberOfNodes = nodesPerDirection[Two];
                    theta = 7.0 * pi / 4.0;
                    stepTheta = (pi / 2.0) / (numberOfNodes - 1);
                    sign = 1;
                    break;
                case Top:
                    numberOfNodes = nodesPerDirection[One];
                    theta = 3 * pi / 4.0;
                    stepTheta = (pi / 2.0) / (numberOfNodes - 1);
                    sign = -1;
                    break;
                case Left:
                    numberOfNodes = nodesPerDirection[Two];
                    theta = 3.0 * pi / 4.0;
                    stepTheta = (pi / 2.0) / (numberOfNodes - 1);
                    sign = 1;
                    break;
                default:
                    throw runtime_error("An ellipse can only have bottom, right, top, and left boundaries.");
            }

            for (unsigned int i = 0; i < numberOfNodes; i++) {
                auto dofBC = new map<DOFType, double>();
                dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta + sign * i * stepTheta)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta + sign * i * stepTheta)));
                boundaryConditionsSet->at(boundary.first).insert(
                        pair<unsigned, BoundaryCondition *>(*boundary.second->at(i)->id.global,
                                                            new BoundaryCondition(Dirichlet, dofBC)));
            }
        }

        _domainBoundaryConditions = new DomainBoundaryConditions(boundaryConditionsSet);
    }*/

    void DomainBoundaryFactory::ellipse(map<Direction, unsigned int> &nodesPerDirection, double radius1, double radius2) {
        const double pi = acos(-1.0);
        auto nn1 = nodesPerDirection[One];
        auto nn2 = nodesPerDirection[Two];

        auto boundaryConditionsSet = new map<Position, map<unsigned, BoundaryCondition *>>();
        unsigned int boundaryNodeID = 0;


        // Bottom & Top Boundaries

        //Bottom
        double theta1 = 5.0 * pi / 4.0;
        double theta2 = 7.0 * pi / 4.0;
        auto hTheta = (theta2 - theta1) / (nn1 - 1);
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Bottom, map<unsigned, BoundaryCondition *>()));
        for (unsigned i = 0; i < nn1; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Bottom)->at(i)->id.global;
            auto dofBC = new map<DOFType, double>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius2 * cos(theta1 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta1 + i * hTheta)));
            boundaryConditionsSet->at(Bottom).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));
        }
        
        //Right
        theta1 = 7.0 * pi / 4.0;
        theta2 = 9.0 * pi / 4.0;
        hTheta = (theta2 - theta1) / (nn2 - 1);
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Right, map<unsigned, BoundaryCondition *>()));
        for (unsigned i = 0; i < nn2; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Right)->at(i)->id.global;
            auto dofBC = new map<DOFType, double>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius1 * sin(theta1 + i * hTheta)));
            boundaryConditionsSet->at(Right).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));
        }
        
        //Top
        theta1 = 9.0 * pi / 4.0;
        theta2 = 11.0 * pi / 4.0;
        hTheta = -(theta2 - theta1) / (nn1 - 1);
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Top, map<unsigned, BoundaryCondition *>()));
        for (unsigned i = 0; i < nn1; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Top)->at(i)->id.global;
            auto dofBC = new map<DOFType, double>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius2 * cos(theta2 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta2 + i * hTheta)));
            boundaryConditionsSet->at(Top).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));
        }
        
        //Left
        theta1 = 11.0 * pi / 4.0;
        theta2 = 13.0 * pi / 4.0;
        hTheta = -(theta2 - theta1) / (nn2 - 1);
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Left, map<unsigned, BoundaryCondition *>()));
        for (unsigned i = 0; i < nn2; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Left)->at(i)->id.global;
            auto dofBC = new map<DOFType, double>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta2 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius1 * sin(theta2 + i * hTheta)));
            boundaryConditionsSet->at(Left).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));
        }
        
        _domainBoundaryConditions = new DomainBoundaryConditions(boundaryConditionsSet);
    }

    void DomainBoundaryFactory::annulus_ripGewrgiou(map<Direction, unsigned int> &nodesPerDirection, double rIn, double rOut, double  thetaStart, double thetaEnd) {
        if (thetaStart > thetaEnd) {
            throw runtime_error("Theta start must be less than theta end.");
        }
        const double pi = acos(-1.0);
        auto theta1 = Utility::Calculators::degreesToRadians(thetaStart);
        auto theta2 = Utility::Calculators::degreesToRadians(thetaEnd);
        //double theta = 2.0 * M_PI -  (theta2 - theta1);
        double theta = (theta2 - theta1);
        auto nn1 = nodesPerDirection[One];
        vector<double> coordinateVector(2, 0.0);
        auto boundaryConditionsSet = new map<Position, map<unsigned, BoundaryCondition *>>();
        unsigned int boundaryNodeID = 0;
        
        //Bottom & Top Boundaries
        auto hTheta = theta / (nn1 - 1);
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Bottom, map<unsigned, BoundaryCondition *>()));
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Top, map<unsigned, BoundaryCondition *>()));
        for (unsigned i = 0; i < nodesPerDirection[One]; i++) {
            //Bottom
            boundaryNodeID = *_mesh->boundaryNodes->at(Bottom)->at(i)->id.global;
            auto dofBC = new map<DOFType, double>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rOut * cos(theta1 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rOut * sin(theta1 + i * hTheta)));
            boundaryConditionsSet->at(Bottom).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));
            //Top
            boundaryNodeID = *_mesh->boundaryNodes->at(Top)->at(i)->id.global;
            dofBC = new map<DOFType, double>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rIn * cos(theta1 + (i) * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rIn * sin(theta1 + (i) * hTheta)));
            boundaryConditionsSet->at(Top).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));
        }
        //Left & Right Boundaries
        auto nn2 = nodesPerDirection[Two];
        auto hR = (rOut - rIn) / (nn2 - 1);
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Right, map<unsigned, BoundaryCondition *>()));
        boundaryConditionsSet->insert(pair<Position, map<unsigned, BoundaryCondition *>>(Left, map<unsigned, BoundaryCondition *>()));
        for (unsigned i = 0; i < nodesPerDirection[Two]; i++) {
            // calculate the current radius
            double rCurrent = rOut - i * hR;

            //Right
            auto dofBC = new map<DOFType, double>();
            boundaryNodeID = *_mesh->boundaryNodes->at(Right)->at(i)->id.global;
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rCurrent * cos(theta2)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rCurrent * sin(theta2)));
            boundaryConditionsSet->at(Right).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));

            //Left
            dofBC = new map<DOFType, double>();
            boundaryNodeID = *_mesh->boundaryNodes->at(Left)->at(i)->id.global;
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rCurrent * cos(theta1)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rCurrent * sin(theta1)));
            boundaryConditionsSet->at(Left).insert(pair<unsigned, BoundaryCondition *>(boundaryNodeID, new BoundaryCondition(Dirichlet, dofBC)));
        }
        _domainBoundaryConditions = new DomainBoundaryConditions(boundaryConditionsSet);
    }
    
} // StructuredMeshGenerator