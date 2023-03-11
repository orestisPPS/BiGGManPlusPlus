//
// Created by hal9000 on 12/17/22.
//

#include "MeshPreProcessor.h"

namespace StructuredMeshGenerator{
    
    MeshPreProcessor :: MeshPreProcessor(MeshSpecs &meshSpecs){
        mesh = initiateMesh(meshSpecs);
        assignCoordinates(meshSpecs);
       // CalculateMeshMetrics();
    }

    Mesh* MeshPreProcessor::initiateMesh(MeshSpecs &meshSpecs) {
        auto nodeFactory = NodeFactory(meshSpecs.nodesPerDirection);
        auto space = calculateSpaceEntityType(meshSpecs);
        switch (space) {
            case Axis:
                return new Mesh1D(nodeFactory.nodesMatrix);
            case Plane:
                return new Mesh2D(nodeFactory.nodesMatrix);
            case Volume:
                return new Mesh3D(nodeFactory.nodesMatrix);
            default:
                throw runtime_error("Invalid space type");
        }
    }

    
    void MeshPreProcessor::assignCoordinates(MeshSpecs &meshSpecs) {
        switch (calculateSpaceEntityType(meshSpecs)) {
            case Axis:
                assign1DCoordinates(meshSpecs);
                break;
            case Plane:
                assign2DCoordinates(meshSpecs);
                break;
            case Volume:
                assign3DCoordinates(meshSpecs);
                break;
            default:
                throw runtime_error("Invalid space type");
        }
        
        auto space = calculateSpaceEntityType(meshSpecs);
        if (space == Axis) {
            assign1DCoordinates(meshSpecs);
        } else if (space == Plane) {
            assign2DCoordinates(meshSpecs);
        } else {
            assign3DCoordinates(meshSpecs);
        }
    }
    
    void MeshPreProcessor::assign1DCoordinates(MeshSpecs &meshSpecs) const {
        for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
            mesh->node(i)->coordinates.addPositionVector(Natural);
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i)}, Parametric);
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i) * meshSpecs.templateStepOne}, Template);
        }
    }
    
    void MeshPreProcessor::assign2DCoordinates(MeshSpecs &meshSpecs) const {
        for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(Two); ++j) {
            for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
                
                // Natural coordinates
                mesh->node(i, j)->coordinates.addPositionVector(Natural);
                // Parametric coordinates
                mesh->node(i, j)->coordinates.addPositionVector({static_cast<double>(i), static_cast<double>(j)}, Parametric);
                // Template coordinates
                vector<double> templateCoord = {static_cast<double>(i) * meshSpecs.templateStepOne,
                                                static_cast<double>(j) * meshSpecs.templateStepTwo};
                // Rotate 
                Transformations::rotate(templateCoord, meshSpecs.templateRotAngleOne);
                // Shear
                Transformations::shear(templateCoord, meshSpecs.templateShearOne,meshSpecs.templateShearTwo);

                mesh->node(i, j)->coordinates.setPositionVector(templateCoord, Template);
            }   
        }
    }
    
    void MeshPreProcessor::assign3DCoordinates(MeshSpecs &meshSpecs) const {
        for (unsigned k = 0; k < mesh->numberOfNodesPerDirection.at(Three); ++k) {
            for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(Two); ++j) {
                for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
                    // Natural coordinates
                    mesh->node(i, j, k)->coordinates.addPositionVector(Natural);
                    // Parametric coordinates
                    mesh->node(i, j, k)->coordinates.addPositionVector(
                            {static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)}, Parametric);
                    // Template coordinates
                    vector<double> templateCoord = {static_cast<double>(i) * meshSpecs.templateStepOne,
                                                    static_cast<double>(j) * meshSpecs.templateStepTwo,
                                                    static_cast<double>(k) * meshSpecs.templateStepThree};
                    // Rotate 
                    Transformations::rotate(templateCoord, meshSpecs.templateRotAngleOne);
                    // Shear
                    Transformations::shear(templateCoord, meshSpecs.templateShearOne,meshSpecs.templateShearTwo);
                }
            }
        }
    }
    
    SpaceEntityType MeshPreProcessor::calculateSpaceEntityType(MeshSpecs &meshSpecs) {
        auto space = NullSpace;
        if (meshSpecs.nodesPerDirection[Two]== 1 && meshSpecs.nodesPerDirection[Three] == 1){
            space = Axis;
        } else if (meshSpecs.nodesPerDirection[Three] == 1){
            space = Plane;
        } else {
            space = Volume;
        }
        return space;
    }

}// StructuredMeshGenerator

