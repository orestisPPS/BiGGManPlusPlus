//
// Created by hal9000 on 12/17/22.
//

#include "MeshFactory.h"



namespace StructuredMeshGenerator{
    
    MeshFactory :: MeshFactory(shared_ptr<MeshSpecs> meshSpecs) : _meshSpecs(std::move(meshSpecs)) {
        mesh = _initiateRegularMesh();
        _assignCoordinates();
        mesh->specs = _meshSpecs;
        mesh->calculateMeshMetrics(Template, true);
        _calculatePDEPropertiesFromMetrics();
    }
    
    void MeshFactory::buildMesh(unsigned short schemeOrder, shared_ptr<DomainBoundaryConditions> boundaryConditions) const {
        
        auto start = chrono::steady_clock::now();
        
        auto pdeProperties = make_shared<SpatialPDEProperties>(mesh->dimensions(), VectorField);
        pdeProperties->setLocallyAnisotropicSpatialProperties(pdePropertiesFromMetrics);

        auto pde = make_shared<SteadyStatePartialDifferentialEquation>(pdeProperties, Laplace);
        
        auto specs = make_shared<FDSchemeSpecs>(schemeOrder, schemeOrder, mesh->directions());
        

        Field_DOFType* dofTypes = nullptr;
        switch (mesh->dimensions()) {
            case 1:
                dofTypes = new nodalPositionVectorField1D_DOFType();
                break;
            case 2:
                dofTypes = new nodalPositionVectorField2D_DOFType();
                break;
            case 3:
                dofTypes = new nodalPositionVectorField3D_DOFType();
                break;
        }
        
        auto problem = make_shared<SteadyStateMathematicalProblem>(std::move(pde), std::move(boundaryConditions), dofTypes);
        
        auto solver = make_shared<ConjugateGradientSolver>(1E-12, 1E4, L2, 1);
        
        auto analysis = make_shared<SteadyStateFiniteDifferenceAnalysis>(problem, mesh, solver, specs, Template);
        analysis->solve();
        
        analysis->linearSystem->exportToMatlabFile("linearSystemMeshGen.m", "/home/hal9000/code/BiGGMan++/Testing/", true);
        //analysis->linearSystem->solution->printHorizontally();
        
        analysis->applySolutionToDegreesOfFreedom();
        
        for (auto &node : *mesh->totalNodesVector){
            auto coords = make_shared<NumericalVector<double>>(node->degreesOfFreedom->size());
            auto i = 0;
            for (auto &dof : *node->degreesOfFreedom){
                (*coords)[i] = dof->value();
                ++i;
            }
            for (auto &dof : *node->degreesOfFreedom){
                delete dof;
                dof = nullptr;
            }
            node->degreesOfFreedom->clear();
            node->coordinates.setPositionVector(coords);
        }
        //delete analysis->degreesOfFreedom;
        dofTypes->deallocate();
        delete dofTypes;
        

        auto end = chrono::steady_clock::now();
        cout<< "Mesh Built in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    }
    
    shared_ptr<Mesh> MeshFactory::_initiateRegularMesh() {
        auto nodesPerDirection = _meshSpecs->nodesPerDirection;
        auto nodeFactory = NodeFactory(_meshSpecs->nodesPerDirection);
        auto space = _calculateSpaceEntityType();
        switch (space) {
            case Axis:
                return make_shared<Mesh1D>(nodeFactory.nodesMatrix);
            case Plane:
                return make_shared<Mesh2D>(nodeFactory.nodesMatrix);
            case Volume:
                return make_shared<Mesh3D>(nodeFactory.nodesMatrix);
            default:
                throw runtime_error("Invalid space type");
        }
    }
    
    void MeshFactory::_assignCoordinates() {
        switch (_calculateSpaceEntityType()) {
            case Axis:
                _assign1DCoordinates();
                break;
            case Plane:
                _assign2DCoordinates();
                break;
            case Volume:
                _assign3DCoordinates();
                break;
            default:
                throw runtime_error("Invalid space type");
        }
        
        auto space = _calculateSpaceEntityType();
        if (space == Axis) {
            _assign1DCoordinates();
        } else if (space == Plane) {
            _assign2DCoordinates();
        } else {
            _assign3DCoordinates();
        }
    }
    
    void MeshFactory::_assign1DCoordinates() const {
        for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
            //mesh->node(i)->coordinates.setPositionVector(Natural);
            auto coords = {static_cast<double>(i)};
            mesh->node(i)->coordinates.setPositionVector(
                    make_shared<NumericalVector<double>>(coords), Template);
            coords = {static_cast<double>(i) * _meshSpecs->templateStepOne};
            mesh->node(i)->coordinates.setPositionVector(
                    make_shared<NumericalVector<double>>(coords), Parametric);   
        }
    }
    
    void MeshFactory::_assign2DCoordinates() const {
        for (unsigned j = 0; j < mesh->nodesPerDirection.at(Two); ++j) {
            for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
                // Parametric coordinates
                auto parametricCoords = make_shared<NumericalVector<double>>(2);
                (*parametricCoords)[0] = static_cast<double>(i);
                (*parametricCoords)[1] = static_cast<double>(j);
                
                mesh->node(i, j)->coordinates.setPositionVector(parametricCoords, Parametric);
                // Template coordinates
                auto templateCoords = make_shared<NumericalVector<double>>(2);
                (*templateCoords)[0] = static_cast<double>(i) * _meshSpecs->templateStepOne;
                (*templateCoords)[1] = static_cast<double>(j) * _meshSpecs->templateStepTwo;
                
                // Rotate
                //Transformations::rotate(templateCoord, _meshSpecs->templateRotAngleOne);
                // Shear
                //Transformations::shear(templateCoord, _meshSpecs->templateShearOne,_meshSpecs->templateShearTwo);
                
                mesh->node(i, j)->coordinates.setPositionVector(templateCoords, Template);
            }
        }
    }
    
    void MeshFactory::_assign3DCoordinates() const {
        for (unsigned k = 0; k < mesh->nodesPerDirection.at(Three); ++k) {
            for (unsigned j = 0; j < mesh->nodesPerDirection.at(Two); ++j) {
                for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
                    auto parametricCoords = make_shared<NumericalVector<double>>(3);
                    (*parametricCoords)[0] = static_cast<double>(i);
                    (*parametricCoords)[1] = static_cast<double>(j);
                    (*parametricCoords)[2] = static_cast<double>(k);
                    mesh->node(i, j, k)->coordinates.setPositionVector(parametricCoords, Parametric);
                    
                    auto templateCoords = make_shared<NumericalVector<double>>(3);
                    (*templateCoords)[0] = static_cast<double>(i) * _meshSpecs->templateStepOne;
                    (*templateCoords)[1] = static_cast<double>(j) * _meshSpecs->templateStepTwo;
                    (*templateCoords)[2] = static_cast<double>(k) * _meshSpecs->templateStepThree;
                    // Rotate 
                    //Transformations::rotate(templateCoord, _meshSpecs->templateRotAngleOne);
                    // Shear
                    //Transformations::shear(templateCoord, _meshSpecs->templateShearOne,_meshSpecs->templateShearTwo);
                    mesh->node(i, j, k)->coordinates.setPositionVector(templateCoords, Template);
                }
            }
        }
    }
    
    SpaceEntityType MeshFactory::_calculateSpaceEntityType(){
        auto space = NullSpace;
        if (_meshSpecs->nodesPerDirection[Two]== 1 && _meshSpecs->nodesPerDirection[Three] == 1){
            space = Axis;
        } else if (_meshSpecs->nodesPerDirection[Three] == 1){
            space = Plane;
        } else {
            space = Volume;
        }
        return space;
    }

    void MeshFactory::_calculatePDEPropertiesFromMetrics() {
        pdePropertiesFromMetrics = make_unique<map<unsigned*, SpatialVectorFieldPDEProperties>>();
        auto size = mesh->dimensions();
        for (auto &node : *mesh->totalNodesVector) {
            
            unsigned* id = node->id.global;
            
            auto nodeSpatialProperties = SpatialVectorFieldPDEProperties();
            //Diffusion
            nodeSpatialProperties.secondOrderCoefficients = std::move(mesh->metrics->at(node)->contravariantTensor);
            //Convection
            auto firstDerivativeCoefficients = make_unique<NumericalMatrix<double>>(mesh->dimensions(), mesh->dimensions());
            for (unsigned i = 0; i < size; ++i) {
                firstDerivativeCoefficients->setElement(i, i, 0);
            }
            nodeSpatialProperties.firstOrderCoefficients = std::move(firstDerivativeCoefficients);
            //Dependent variable coefficients
            nodeSpatialProperties.zerothOrderCoefficient = std::move(make_unique<NumericalVector<double>>(mesh->dimensions()));
            nodeSpatialProperties.sourceTerm = std::move(make_unique<double>(0));

            pdePropertiesFromMetrics->emplace(id, std::move(nodeSpatialProperties));
        }
    }



}// StructuredMeshGenerator

