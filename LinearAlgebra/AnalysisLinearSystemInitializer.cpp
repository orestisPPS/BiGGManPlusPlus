//
// Created by hal9 000 on 3/28/23.
//

#include "AnalysisLinearSystemInitializer.h"

namespace LinearAlgebra {
    
    
    
    AnalysisLinearSystemInitializer::AnalysisLinearSystemInitializer(AnalysisDegreesOfFreedom* analysisDegreesOfFreedom, Mesh* mesh,
                                                                     MathematicalProblem* problem, FDSchemeSpecs* specs) {
        linearSystem = nullptr;
        this->_analysisDegreesOfFreedom = analysisDegreesOfFreedom;
        this->_mesh = mesh;
        this->_mathematicalProblem = problem;
        numberOfFreeDOFs = new unsigned(_analysisDegreesOfFreedom->freeDegreesOfFreedom->size());
        numberOfFixedDOFs = new unsigned(_analysisDegreesOfFreedom->fixedDegreesOfFreedom->size());
        numberOfDOFs = new unsigned(_analysisDegreesOfFreedom->totalDegreesOfFreedomMap->size());
        _freeDOFMatrix = new Array<double>(*numberOfFreeDOFs, *numberOfFreeDOFs);
        _fixedDOFMatrix = new Array<double>(*numberOfFixedDOFs, *numberOfDOFs);
        _parametricCoordToNodeMap = _mesh->createParametricCoordToNodesMap();
        _specs = specs;
        //matrix->print();
    }
        
    AnalysisLinearSystemInitializer::~AnalysisLinearSystemInitializer() {
        delete linearSystem;
        delete numberOfFreeDOFs;
        delete numberOfFixedDOFs;
        delete numberOfDOFs;
        _mesh = nullptr;
    }
    
    void AnalysisLinearSystemInitializer::createLinearSystem() {
        _createMatrix();
        _createRHS();
        linearSystem = new LinearSystem(_matrix, _RHS);
    }
    
    void AnalysisLinearSystemInitializer::_createMatrix() {
        _createFixedDOFSubMatrix();
        _createFreeDOFSubMatrix();
    }
    
    //Creates a matrix with consistent order across the domain. The scheme type is defined by the node neighbourhood.
    //Error Order is user defined.
    void AnalysisLinearSystemInitializer::_createFreeDOFSubMatrix() {
        //Define the directions of the simulation
        auto directions = _mesh->directions();

        //Create Scheme Builder for utility functions
        auto schemeBuilder = FiniteDifferenceSchemeBuilder(_specs);

        //Define the error order for each derivative order
        auto errorOrderDerivative1 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(1);
        auto errorOrderDerivative2 = _specs->getErrorOrderOfVariableSchemeTypeForDerivativeOrder(2);

        //Find the number of points needed for the desired order of accuracy.
        //The scheme type varies depending on the available neighbours of the dof.
        auto schemeTypeToPointsDerivative1 = schemeBuilder.schemeOrderToSchemeTypePointsDerivative1()[errorOrderDerivative1];
        auto schemeTypeToPointsDerivative2 = schemeBuilder.schemeOrderToSchemeTypePointsDerivative2()[errorOrderDerivative2];

        //Convert scheme type to positions
        auto schemeTypeToPosition = schemeBuilder.schemeTypeToPositions();
        //Define the positions needed for the scheme at each direction as well as the number of points needed.
        auto positionsAtDirectionsDerivative1 = map<Direction, map<vector<Position>, short int>>();
        auto positionsAtDirectionsDerivative2 = map<Direction, map<vector<Position>, short int>>();


        for (auto &direction : directions) {
            positionsAtDirectionsDerivative1.insert(pair<Direction, map<vector<Position>, short int>>(
                    direction, map<vector<Position>, short int>()));
            positionsAtDirectionsDerivative2.insert(pair<Direction, map<vector<Position>, short int>>(
                    direction, map<vector<Position>, short int>()));
            
            auto schemeTypeToPositions = schemeTypeToPosition[direction];
            for (auto &scheme : schemeTypeToPositions) {
                positionsAtDirectionsDerivative1[direction].insert(pair<vector<Position>, short int>
                                                   (scheme.second, schemeTypeToPointsDerivative1[scheme.first]));
                positionsAtDirectionsDerivative2[direction].insert(pair<vector<Position>, short int>
                                                   (scheme.second, schemeTypeToPointsDerivative2[scheme.first]));
            }
        }
        //TODO make function checkIfRequiredPositionsAreAvailable. possibly in the node graph class

        //Find the maximum number of neighbours needed for the desired order of accuracy. Used as input node graph depth.
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        //Iterate over the free degrees of freedom
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            //for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);

            //Initiate node graph with depth equal to the maximum number of neighbours needed for the desired order of accuracy
            auto graph =IsoParametricNodeGraph(node, maxNeighbours, _parametricCoordToNodeMap, _mesh->nodesPerDirection);
            //TODO: check why node id is not ascending in direction 1. for free dof 1 (node 6)
            //      neighbours are free dof (1 (node 7), 2 (node 8), 10 (node 9). this maybe affects the sparsity pattern.
            auto neighbourDOF = graph.getSpecificDOFGraph(dof->type());
            auto positions = graph.getColinearPositions(directions);

            for (auto &direction : directions){
                //Check if the template positions are available in the node graph
                //Begin an if statement with central scheme starting first
                //->get the required positions
                //->get the scheme weights
                //->add to matrix as below
            }


            //Calculate Diagonal Element
            unsigned positionI = *dof->id->value;
            auto valueICoefficient = 0.0;
            for (auto &direction : directions) {

            }

/*            _freeDOFMatrix->at(positionI, positionI) = -2;

            //J Position in the matrix (neighbouring free DOF id)
            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    if (neighbourDof->id->constraintType() == Free) {
                        unsigned positionJ = *neighbourDof->id->value;
                        _freeDOFMatrix->at(positionI, positionJ) = 1;
                    }
                }
            }
            delete dofGraph;
            dofGraph = nullptr;*/




            //Find PDE Coefficients
            auto secondOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).secondOrderCoefficients;
            auto firstOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).firstOrderCoefficients;
            auto zerothOrderCoefficients =
                    _mathematicalProblem->pde->properties->getLocalProperties(*node->id.global).zerothOrderCoefficient;
        }

        _matrix = _freeDOFMatrix;
        _freeDOFMatrix = nullptr;

        cout<<"Free DOF matrix"<<endl;
        //matrix->print();
        cout << "  " << endl;

    }
    
    
    
    
    
    void AnalysisLinearSystemInitializer::_createFixedDOFSubMatrix() {
        for (auto &dof: *_analysisDegreesOfFreedom->fixedDegreesOfFreedom) {
            //Node with the DOF
            auto node = _mesh->nodeFromID(*dof->parentNode);
            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->nodesPerDirection).
                            getSpecificDOFGraph(dof->type());
            unsigned positionI;
            unsigned positionJ;
            
            positionI = *dof->id->value;
            positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
            _fixedDOFMatrix->at(positionI, positionJ) = 2;
            
            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    positionJ = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(neighbourDof);
                    _fixedDOFMatrix->at(positionI, positionJ) = 1;
                }
            }
            delete dofGraph;
            dofGraph = nullptr;
        }
        cout<<"Fixed DOF matrix"<<endl;
        //_fixedDOFMatrix->print();
    }
        
    void AnalysisLinearSystemInitializer::_createRHS() {
        //Marching through all the free DOFs
        _RHS = new vector<double>(*numberOfFreeDOFs, 0);
        for (auto &dof: *_analysisDegreesOfFreedom->freeDegreesOfFreedom) {
            auto node = _mesh->nodeFromID(*dof->parentNode);
            
            //Get all the neighbouring DOFs with the same type
            auto dofGraph =
                    IsoParametricNodeGraph(node, 1, _parametricCoordToNodeMap, _mesh->nodesPerDirection).
                            getSpecificDOFGraph(dof->type());
            //Marching through all the neighbouring DOFs
            for (auto &neighbour: *dofGraph) {
                for (auto &neighbourDof: neighbour.second) {
                    //Check if the neighbouring DOF is fixed 
                    if (neighbourDof->id->constraintType() == Fixed) {
                        unsigned i = *neighbourDof->id->value;
                        unsigned j = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
                        _RHS->at(*dof->id->value) -= _fixedDOFMatrix->at(i, j) * neighbourDof->value();
                    }
                }
            }            
            delete dofGraph;
            dofGraph = nullptr;
        }
        delete _fixedDOFMatrix;
        _fixedDOFMatrix = nullptr;
        
        //print vector
        for (auto &value: *_RHS) {
            cout << value << endl;
        }
    }
}// LinearAlgebra