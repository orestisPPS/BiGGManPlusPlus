//
// Created by hal9000 on 5/13/23.
//

#include "DomainBoundaryFactory.h"

namespace StructuredMeshGenerator {

    DomainBoundaryFactory::DomainBoundaryFactory(const shared_ptr<Mesh> &mesh) : _mesh(mesh) {
    }

    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::parallelogram(map<Direction, unsigned> &nodesPerDirection, double lengthX, double lengthY,
                                         double rotAngle, double shearX, double shearY) {
        auto nnX = nodesPerDirection[One];
        auto nnY = nodesPerDirection[Two];

        auto stepX = lengthX / (nnX - 1.0);
        auto stepY = lengthY / (nnY - 1.0);
        NumericalVector<double> coordinateVector(2, 0);
        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();

        for (auto &boundary: *_mesh->boundaryNodes) {

            boundaryConditionsSet->insert(pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(
                    boundary.first, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));

            switch (boundary.first) {
                case Bottom:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = 0.0;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        auto bc =
                                boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                        *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Right:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = lengthX;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Top:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = lengthY;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Left:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = 0.0;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        Transformations::rotate(coordinateVector, rotAngle);
                        Transformations::shear(coordinateVector, shearX, shearY);
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                default:
                    throw runtime_error("A parallelogram can only have bottom, right, top, and left boundaries.");
            }
        }
        return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
    }

    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::parallelepiped(map<Direction, unsigned> &nodesPerDirection,
                                               double lengthX, double lengthY, double lengthZ,
                                               double rotAngleX, double rotAngleY, double rotAngleZ,
                                               double shearX, double shearY, double shearZ) {
        auto nnX = nodesPerDirection[One];
        auto nnY = nodesPerDirection[Two];
        auto nnZ = nodesPerDirection[Three];

        auto stepX = lengthX / (nnX - 1.0);
        auto stepY = lengthY / (nnY - 1.0);
        auto stepZ = lengthZ / (nnZ - 1.0);
        
        NumericalVector<double> coordinateVector(3, 0);
        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();

        for (auto &boundary: *_mesh->boundaryNodes) {

            boundaryConditionsSet->insert(pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(
                    boundary.first, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));

            switch (boundary.first) {
                case Bottom:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        coordinateVector[2] = 0.0;

                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position3, coordinateVector[2]));
                        auto bc =
                                boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                        *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Top:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        coordinateVector[2] = lengthZ;
                        
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position3, coordinateVector[2]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Right:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = lengthX;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        coordinateVector[2] = nodalParametricCoords[2] * stepZ;
                        
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position3, coordinateVector[2]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;

                case Left:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = 0.0;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        coordinateVector[2] = nodalParametricCoords[2] * stepZ; 
                        
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position3, coordinateVector[2]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Front:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = lengthY;
                        coordinateVector[2] = nodalParametricCoords[2] * stepZ;
                        
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position3, coordinateVector[2]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Back:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = 0;
                        coordinateVector[2] = nodalParametricCoords[2] * stepZ;
                        
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position3, coordinateVector[2]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                default:
                    throw runtime_error("A parallelogram can only have bottom, right, top, and left boundaries.");
            }
        }
        return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
    }



    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::ellipse(map<Direction, unsigned int> &nodesPerDirection, double radius1, double radius2) {
        const double pi = acos(-1.0);
        auto nn1 = nodesPerDirection[One];
        auto nn2 = nodesPerDirection[Two];

        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();
        unsigned int boundaryNodeID = 0;
        
        // Bottom & Top Boundaries

        //Bottom
        double theta1 = 5.0 * pi / 4.0;
        double theta2 = 7.0 * pi / 4.0;
        auto hTheta = (theta2 - theta1) / (nn1 - 1);
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Bottom, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn1; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Bottom)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta1 + i * hTheta)));
            boundaryConditionsSet->at(Bottom)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }


        //THIS CREATES SUBMARINE EFFECT     
        //keep radius1 for both dofs to create submarine effect
        theta1 = 7.0 * pi / 4.0;
        theta2 = 9.0 * pi / 4.0;
        hTheta = (theta2 - theta1) / (nn2 - 1);
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Top, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn2; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Right)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta1 + i * hTheta)));
            boundaryConditionsSet->at(Right)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }

        //Top
        theta1 = 9.0 * pi / 4.0;
        theta2 = 11.0 * pi / 4.0;
        hTheta = -(theta2 - theta1) / (nn1 - 1);
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Top, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn1; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Top)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta2 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta2 + i * hTheta)));
            boundaryConditionsSet->at(Top)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }

        //Left
        theta1 = 11.0 * pi / 4.0;
        theta2 = 13.0 * pi / 4.0;
        hTheta = -(theta2 - theta1) / (nn2 - 1);
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Left, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn2; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Left)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta2 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta2 + i * hTheta)));
            boundaryConditionsSet->at(Left)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }

        return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
    }

/*    void DomainBoundaryFactory::ellipse3D(map<Direction, unsigned int> &nodesPerDirection, double radius1, double radius2, double radius3) {
        const double pi = acos(-1.0);
        auto nn1 = nodesPerDirection[One];
        auto nn2 = nodesPerDirection[Two];
        auto nn3 = nodesPerDirection[Three];

        auto boundaryConditionsSet = make_shared<map<Position, map<unsigned, shared_ptr<BoundaryCondition>>>>();
        unsigned int boundaryNodeID = 0;

        // Bottom & Top Boundaries

        //Bottom
        double theta1 = 5.0 * pi / 4.0;
        double theta2 = 7.0 * pi / 4.0;
        double phi1 = 5.0 * pi / 4.0;
        double phi2 = 7.0 * pi / 4.0;
        auto hTheta = (theta2 - theta1) / (nn1 - 1);
        auto hPhi = (phi2 - phi1) / (nn2 - 1);
        boundaryConditionsSet->insert(
                pair<Position, map<unsigned, shared_ptr<BoundaryCondition>>>(Bottom, map<unsigned, shared_ptr<BoundaryCondition>>()));
        for (unsigned i = 0; i < nn1; i++) {
            for (unsigned j = 0; j < nn2; j++) {
                boundaryNodeID = *_mesh->boundaryNodes->at(Bottom)->at(i * nn2 + j)->id.global;
                auto dofBC = make_shared<map<DOFType, double>>();
                dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position3, radius3 * sin(phi1 + j * hPhi)));
                boundaryConditionsSet->at(Bottom).insert(
                        pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
            }
        }
        
        //Top
        theta1 = 3.0 * pi / 4.0;
        theta2 = 5.0 * pi / 4.0;
        phi1 = 3.0 * pi / 4.0;
        phi2 = 5.0 * pi / 4.0;
        hTheta = (theta2 - theta1) / (nn1 - 1);
        hPhi = (phi2 - phi1) / (nn2 - 1);
        boundaryConditionsSet->insert(
                pair<Position, map<unsigned, shared_ptr<BoundaryCondition>>>(Top, map<unsigned, shared_ptr<BoundaryCondition>>()));
        for (unsigned i = 0; i < nn1; i++) {
            for (unsigned j = 0; j < nn2; j++) {
                boundaryNodeID = *_mesh->boundaryNodes->at(Top)->at(i * nn2 + j)->id.global;
                auto dofBC = make_shared<map<DOFType, double>>();
                dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position3, radius3 * sin(phi1 + j * hPhi)));
                boundaryConditionsSet->at(Top).insert(
                        pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
            }
        }
        

        //Left
        theta1 = 11.0 * pi / 4.0;
        theta2 = 13.0 * pi / 4.0;
        phi1 = 11.0 * pi / 4.0;
        phi2 = 13.0 * pi / 4.0;
        hTheta = (theta2 - theta1) / (nn2 - 1);
        hPhi = (phi2 - phi1) / (nn3 - 1);
        boundaryConditionsSet->insert(
                pair<Position, map<unsigned, shared_ptr<BoundaryCondition>>>(Left, map<unsigned, shared_ptr<BoundaryCondition>>()));
        for (unsigned i = 0; i < nn2; i++) {
            for (unsigned j = 0; j < nn3; j++) {
                boundaryNodeID = *_mesh->boundaryNodes->at(Left)->at(i * nn3 + j)->id.global;
                auto dofBC = make_shared<map<DOFType, double>>();
                dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius1 * sin(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position3, radius3 * sin(phi1 + j * hPhi)));
                boundaryConditionsSet->at(Left).insert(
                        pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
            }
        }
        
        //Right
        theta1 = 7.0 * pi / 4.0;
        theta2 = 9.0 * pi / 4.0;
        phi1 = 7.0 * pi / 4.0;
        phi2 = 9.0 * pi / 4.0;
        hTheta = (theta2 - theta1) / (nn2 - 1);
        hPhi = (phi2 - phi1) / (nn3 - 1);
        boundaryConditionsSet->insert(
                pair<Position, map<unsigned, shared_ptr<BoundaryCondition>>>(Right, map<unsigned, shared_ptr<BoundaryCondition>>()));
        for (unsigned i = 0; i < nn2; i++) {
            for (unsigned j = 0; j < nn3; j++) {
                boundaryNodeID = *_mesh->boundaryNodes->at(Right)->at(i * nn3 + j)->id.global;
                auto dofBC = make_shared<map<DOFType, double>>();
                dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius1 * sin(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position3, radius3 * sin(phi1 + j * hPhi)));
                boundaryConditionsSet->at(Right).insert(
                        pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
            }
        }

        //Top
        theta1 = 9.0 * pi / 4.0;
        theta2 = 11.0 * pi / 4.0;
        phi1 = 9.0 * pi / 4.0;
        phi2 = 11.0 * pi / 4.0;
        hTheta = (theta2 - theta1) / (nn1 - 1);
        hPhi = (phi2 - phi1) / (nn2 - 1);
        boundaryConditionsSet->insert(
                pair<Position, map<unsigned, shared_ptr<BoundaryCondition>>>(Top, map<unsigned, shared_ptr<BoundaryCondition>>()));
        for (unsigned i = 0; i < nn1; i++) {
            for (unsigned j = 0; j < nn2; j++) {
                boundaryNodeID = *_mesh->boundaryNodes->at(Top)->at(i * nn2 + j)->id.global;
                auto dofBC = make_shared<map<DOFType, double>>();
                dofBC->insert(pair<DOFType, double>(DOFType::Position1, radius1 * cos(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius2 * sin(theta1 + i * hTheta) * cos(phi1 + j * hPhi)));
                dofBC->insert(pair<DOFType, double>(DOFType::Position3, radius3 * sin(phi1 + j * hPhi)));
                boundaryConditionsSet->at(Top).insert(
                        pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
            }
        }
        

        
        //Bottom
        

    }*/

    shared_ptr<DomainBoundaryConditions>
    DomainBoundaryFactory::annulus_ripGewrgiou(map<Direction, unsigned int> &nodesPerDirection, double rIn, double rOut,
                                               double thetaStart, double thetaEnd) {
        if (thetaStart > thetaEnd) {
            throw runtime_error("Theta start must be less than theta end.");
        }
        auto theta1 = Utility::Calculators::degreesToRadians(thetaStart);
        auto theta2 = Utility::Calculators::degreesToRadians(thetaEnd);
        //double theta = 2.0 * M_PI -  (theta2 - theta1);
        double theta = (theta2 - theta1);
        auto nn1 = nodesPerDirection[One];
        NumericalVector<double> coordinateVector(2, 0.0);
        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();
        unsigned int boundaryNodeID = 0;

        //Bottom & Top Boundaries
        auto hTheta = theta / (nn1 - 1);
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Bottom, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Top, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nodesPerDirection[One]; i++) {
            //Bottom
            boundaryNodeID = *_mesh->boundaryNodes->at(Bottom)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rOut * cos(theta1 + i * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rOut * sin(theta1 + i * hTheta)));
            boundaryConditionsSet->at(Bottom)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
            //Top
            boundaryNodeID = *_mesh->boundaryNodes->at(Top)->at(i)->id.global;
            dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rIn * cos(theta1 + (i) * hTheta)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rIn * sin(theta1 + (i) * hTheta)));
            boundaryConditionsSet->at(Top)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }
        //Left & Right Boundaries
        auto nn2 = nodesPerDirection[Two];
        auto hR = (rOut - rIn) / (nn2 - 1);
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Left, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Right, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nodesPerDirection[Two]; i++) {
            // calculate the current radius
            double rCurrent = rOut - i * hR;

            //Right
            auto dofBC = make_shared<map<DOFType, double>>();
            boundaryNodeID = *_mesh->boundaryNodes->at(Right)->at(i)->id.global;
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rCurrent * cos(theta2)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rCurrent * sin(theta2)));
            boundaryConditionsSet->at(Right)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
            //Left
            dofBC = make_shared<map<DOFType, double>>();
            boundaryNodeID = *_mesh->boundaryNodes->at(Left)->at(i)->id.global;
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, rCurrent * cos(theta1)));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, rCurrent * sin(theta1)));
            boundaryConditionsSet->at(Left)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));}
        return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
    }

    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::cavityBot(map<Direction, unsigned> &nodesPerDirection, double lengthX, double lengthY) {
        auto nnX = nodesPerDirection[One];
        auto nnY = nodesPerDirection[Two];

        auto stepX = lengthX / (nnX - 1.0);
        auto stepY = lengthY / (nnY - 1.0);
        NumericalVector<double> coordinateVector(2, 0);
        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();

        for (auto &boundary: *_mesh->boundaryNodes) {

            boundaryConditionsSet->insert(pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(
                    boundary.first, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));

            switch (boundary.first) {
                case Bottom: {
                    auto counter = 0;
                    for (auto &node: *boundary.second) {
                        auto radius = lengthX / 2.0;
                        auto hTheta = M_PI / (nnX - 1.0);

                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1,
                                                            radius + radius * cos(M_PI + counter * hTheta)));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, radius * sin(M_PI + counter * hTheta)));
                        auto bc =
                                boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                        *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                        counter++;
                    }
                    break;
                }

                case Right:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = lengthX;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Top:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = lengthY;
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Left:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = 0.0;
                        coordinateVector[1] = nodalParametricCoords[1] * stepY;
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                default:
                    throw runtime_error("A parallelogram can only have bottom, right, top, and left boundaries.");
            }
        }
        return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
    }


    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::gasTankHorizontal(map<Direction, unsigned int> &nodesPerDirection, double lengthX,
                                                  double lengthY) {
        auto nnX = nodesPerDirection[One];
        auto nnY = nodesPerDirection[Two];

        auto stepX = lengthX / (nnX - 1.0);
        NumericalVector<double> coordinateVector(2, 0);
        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();

        for (auto &boundary: *_mesh->boundaryNodes) {

            boundaryConditionsSet->insert(pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(
                    boundary.first, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));

            switch (boundary.first) {
                case Bottom:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = 0.0;
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        auto bc =
                                boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                        *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;

                case Right: {
                    auto radius = lengthY / 2.0;
                    auto hTheta = M_PI / (nnY - 1.0);
                    auto counter = 0;
                    for (auto &node: *boundary.second) {
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1,
                                                            lengthX + radius * cos(3 * M_PI / 2.0 + counter * hTheta)));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2,
                                                            radius + radius * sin(3 * M_PI / 2.0 + counter * hTheta)));
                        auto bc = boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                        counter++;
                    }
                    break;
                }
                case Top:
                    for (auto &node: *boundary.second) {
                        auto nodalParametricCoords = *node->coordinates.getPositionVector(Parametric);
                        coordinateVector[0] = nodalParametricCoords[0] * stepX;
                        coordinateVector[1] = lengthY;
                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1, coordinateVector[0]));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2, coordinateVector[1]));
                        boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                    }
                    break;
                case Left: {
                    auto counter = 0;
                    for (auto &node: *boundary.second) {
                        auto radius = lengthY / 2.0;
                        auto hTheta = M_PI / (nnY - 1.0);

                        auto dofBC = make_shared<map<DOFType, double>>();
                        dofBC->insert(pair<DOFType, double>(DOFType::Position1,
                                                            radius * cos(3 * M_PI / 2.0 - counter * hTheta)));
                        dofBC->insert(pair<DOFType, double>(DOFType::Position2,
                                                            radius + radius * sin(3 * M_PI / 2.0 - counter * hTheta)));
                        auto bc = boundaryConditionsSet->at(boundary.first)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                                *node->id.global, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
                        counter++;
                    }
                    break;
                }
                default:
                    throw runtime_error("A parallelogram can only have bottom, right, top, and left boundaries.");
            }
            return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
        }
    }
    //set xrange [0:0.6]
    //set yrange [-3:1]
    //plot sin(15*x) , - 2 + sin(15*x)
    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::sinusRiver(map<Direction, unsigned int> &nodesPerDirection, double lengthX,
                                           double bankDistanceY, double amplitude, double frequency) {
        if (bankDistanceY <= 0.0)
            throw runtime_error("The bank distance must be positive.");
        if (amplitude <= 0.0)
            throw runtime_error("The amplitude must be positive.");

        const double pi = acos(-1.0);
        auto nn1 = nodesPerDirection[One];
        auto nn2 = nodesPerDirection[Two];
        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();
        unsigned int boundaryNodeID = 0;
        auto stepX = lengthX / (nn1 - 1.0);
        auto stepY = bankDistanceY / (nn2 - 1.0);

        //Bottom
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Bottom, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn1; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Bottom)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, i * stepX));
            dofBC->insert( pair<DOFType, double>(DOFType::Position2, -bankDistanceY + amplitude * sin(frequency * i * stepX)));
            boundaryConditionsSet->at(Bottom)->insert( pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }
        //Bottom
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Bottom, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn1; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Top)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, i * stepX));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, amplitude * sin(frequency * i * stepX)));
            boundaryConditionsSet->at(Top)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }

        //Left
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Left, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn2; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Left)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, 0.0));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, -bankDistanceY + i * stepY));
            boundaryConditionsSet->at(Left)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }

        //Right
        boundaryConditionsSet->insert(
                pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(Right, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        for (unsigned i = 0; i < nn2; i++) {
            boundaryNodeID = *_mesh->boundaryNodes->at(Right)->at(i)->id.global;
            auto dofBC = make_shared<map<DOFType, double>>();
            dofBC->insert(pair<DOFType, double>(DOFType::Position1, lengthX));
            dofBC->insert(pair<DOFType, double>(DOFType::Position2, -bankDistanceY + i * stepY));
            boundaryConditionsSet->at(Right)->insert(
                    pair<unsigned, shared_ptr<BoundaryCondition>>(boundaryNodeID, make_shared<BoundaryCondition>(Dirichlet, dofBC)));
        }

        return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
    }

    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::gasTankVertical(map<Direction, unsigned int> &nodesPerDirection, double lenghtX,
                                                double lengthY) {
    
        return nullptr;
    }

    shared_ptr<DomainBoundaryConditions> DomainBoundaryFactory::annulus_3D_ripGewrgiou(map<Direction, unsigned int> &nodesPerDirection, double rIn,
                                                       double rOut, double thetaStart, double thetaEnd, double height) {
        
    
        auto nn1 = nodesPerDirection[One];
        auto nn2 = nodesPerDirection[Two];
        auto nn3 = nodesPerDirection[Three];
        
        auto step3 = height / (nn3 - 1.0);

        auto boundaryConditionsSet = make_shared<map<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>>();

        map<Direction, unsigned int> nodesPerDirection2D = {{One, nn1}, {Two, nn2}};
        auto specs = make_shared<MeshSpecs>(nodesPerDirection2D, 1, 1, 0, 0, 0);
        auto meshFactory2D = new MeshFactory(specs);
        auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory2D->mesh)->annulus_ripGewrgiou(nodesPerDirection2D, rIn, rOut, thetaStart, thetaEnd);
        //auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory2D->mesh)->parallelogram(nodesPerDirection2D,4,4);
        meshFactory2D->buildMesh(2, meshBoundaries);
        auto mesh2D = meshFactory2D->mesh;
        
        
        unsigned int ksi, ita, iota;

        for (auto &boundaryPosition : *this->_mesh->boundaryNodes){
            boundaryConditionsSet->insert(pair<Position, shared_ptr<map<unsigned, shared_ptr<BoundaryCondition>>>>(
                    boundaryPosition.first, make_shared<map<unsigned, shared_ptr<BoundaryCondition>>>()));
        }
        
        //Bottom and Top
        for (unsigned i = 0; i < nn2; i++) {
            for (unsigned j = 0; j < nn1; j++) {
                auto nodalCoords = *mesh2D->node(j, i)->coordinates.getPositionVector(Natural);
                NumericalVector<double> new3DCoords = {nodalCoords[0], nodalCoords[1], 0};
                auto bottomBCValues = make_shared<map<DOFType, double>>(map<DOFType, double>({
                        {Position1, new3DCoords[0]},
                        {Position2, new3DCoords[1]},
                        {Position3, new3DCoords[2]}
                }));
                auto bcBottom = make_shared<BoundaryCondition>(Dirichlet, std::move(bottomBCValues));
                boundaryConditionsSet->at(Bottom)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(*this->_mesh->node(j, i, 0)->id.global, bcBottom));

                
                new3DCoords = {nodalCoords[0], nodalCoords[1], height};
                auto topBCValues = make_shared<map<DOFType, double>>(map<DOFType, double>({
                        {Position1, new3DCoords[0]},
                        {Position2, new3DCoords[1]},
                        {Position3, new3DCoords[2]}
                }));
                auto bcTop = make_shared<BoundaryCondition>(Dirichlet, std::move(topBCValues));
                boundaryConditionsSet->at(Top)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(*this->_mesh->node( j, i, nn3 - 1)->id.global, bcTop));
            }
        }
        
        //left and right
        auto leftNodes = mesh2D->boundaryNodes->at(Left);
        auto rightNodes = mesh2D->boundaryNodes->at(Right);
        for (unsigned i = 0; i < nn3; i++) {
            for (unsigned j = 0; j < nn2; j++) {
                auto nodalCoords = *leftNodes->at(j)->coordinates.getPositionVector(Natural);
                NumericalVector<double> new3DCoords = {nodalCoords[0], nodalCoords[1], i * step3};
                auto leftBCValues = make_shared<map<DOFType, double>>(map<DOFType, double>({
                        {Position1, new3DCoords[0]},
                        {Position2, new3DCoords[1]},
                        {Position3, new3DCoords[2]}
                }));
                auto bcLeft = new BoundaryCondition(Dirichlet, std::move(leftBCValues));
                boundaryConditionsSet->at(Left)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(*this->_mesh->node(0, j, i)->id.global, bcLeft));
                
                nodalCoords = *rightNodes->at(j)->coordinates.getPositionVector(Natural);
                new3DCoords = {nodalCoords[0], nodalCoords[1], i * step3};
                auto rightBCValues = make_shared<map<DOFType, double>>(map<DOFType, double>({
                        {Position1, new3DCoords[0]},
                        {Position2, new3DCoords[1]},
                        {Position3, new3DCoords[2]}
                }));
                auto bcRight = new BoundaryCondition(Dirichlet, std::move(rightBCValues));
                boundaryConditionsSet->at(Right)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(*this->_mesh->node(nn1 - 1, j, i)->id.global, bcRight));
            }
        }
        
        //front and back
        auto frontNodes = mesh2D->boundaryNodes->at(Top);
        auto backNodes = mesh2D->boundaryNodes->at(Bottom);
        
        for (unsigned i = 0; i < nn3; i++) {
            for (unsigned j = 0; j < nn1; j++) {
                auto nodalCoords = *frontNodes->at(j)->coordinates.getPositionVector(Natural);
                NumericalVector<double> new3DCoords = {nodalCoords[0], nodalCoords[1], i * step3};
                auto frontBCValues = make_shared<map<DOFType, double>>();
                frontBCValues->insert(pair<DOFType, double>(Position1, new3DCoords[0]));
                frontBCValues->insert(pair<DOFType, double>(Position2, new3DCoords[1]));
                frontBCValues->insert(pair<DOFType, double>(Position3, new3DCoords[2]));
                auto bcFront = make_shared<BoundaryCondition>(Dirichlet, std::move(frontBCValues));
                boundaryConditionsSet->at(Front)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                        *this->_mesh->node(j, nn2 - 1, i)->id.global, bcFront));

                nodalCoords = *backNodes->at(j)->coordinates.getPositionVector(Natural);
                new3DCoords = {nodalCoords[0], nodalCoords[1], i * step3};
                auto backBCValues = make_shared<map<DOFType, double>>();
                backBCValues->insert(pair<DOFType, double>(Position1, new3DCoords[0]));
                backBCValues->insert(pair<DOFType, double>(Position2, new3DCoords[1]));
                backBCValues->insert(pair<DOFType, double>(Position3, new3DCoords[2]));
                auto bcBack = make_shared<BoundaryCondition>(Dirichlet, std::move(backBCValues));
                //boundaryConditionsSet->at(Back).insert(pair<unsigned, shared_ptr<BoundaryCondition>>(*backNodes->at(j)->id//.global, bcBack));
                boundaryConditionsSet->at(Back)->insert(pair<unsigned, shared_ptr<BoundaryCondition>>(
                        *this->_mesh->node(j, 0, i)->id.global, bcBack));
            }
        }
        mesh2D.reset();
        return make_shared<DomainBoundaryConditions>(boundaryConditionsSet);
    }

}// StructuredMeshGenerator
