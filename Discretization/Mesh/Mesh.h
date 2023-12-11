//
// Created by hal9000 on 12/17/22.
//
#ifndef UNTITLED_MESH_H
#define UNTITLED_MESH_H

#include "../Node/Node.h"
#include "../../LinearAlgebra/Array/Array.h"
#include "../../StructuredMeshGeneration/MeshSpecs.h"
#include "Metrics/Metrics.h"
#include "../Node/IsoparametricNodeGraph.h"
#include "../../LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeBuilder.h"
#include "../../LinearAlgebra/FiniteDifferences/FDWeightCalculator.h"
#include "../Elements/Element.h"
#include "../Elements/MeshElements.h"
#include "../../Logging/Logs.h"


using namespace Discretization;
using namespace StructuredMeshGenerator;
using namespace LinearAlgebra;


namespace Discretization {

    class Mesh {
    
    public:
        //Mesh(shared_ptr<Array<Node*>>nodes, map<Direction, int> nodesPerDirection);
        explicit Mesh(shared_ptr<Array<Node*>>nodes);
        
        virtual ~Mesh();
        
        //map<Direction, unsigned > *nodesPerDirection;
        map<Direction, unsigned > nodesPerDirection;
        
        shared_ptr<map<Position, shared_ptr<vector<Node*>>>> boundaryNodes;
        
        shared_ptr<vector<Node*>> totalNodesVector;
        
        unique_ptr<MeshElements> elements;
        
        bool isInitialized;
        
        shared_ptr<MeshSpecs> specs;
        
        shared_ptr<unordered_map<Node*, shared_ptr<Metrics>>> metrics;
        
        Logs logs;
        
        //---------------Implemented parent class methods--------------
        
        /// \brief Returns the number of nodes in the mesh
        /// \return An unsigned integer representing the total number of nodes in the mesh
        /// \exception std::runtime_error Thrown when the Mesh has not been initialized
        unsigned numberOfTotalNodes();
        
        /// \brief Returns the number of boundary nodes in the mesh
        /// \return An unsigned integer representing the total number of the boundary nodes of the mesh
        /// \exception std::runtime_error Thrown when the Mesh has not been initialized
        unsigned numberOfBoundaryNodes();
        
        /// \brief Retrieves a node given its ID
        /// \param ID  (unsigned int) The ID of the node to retrieve
        /// \return Pointer to the Node object if the mesh is initialized, nullptr otherwise
        Node* nodeFromID(unsigned* ID);

        /// \brief This method iterates over all boundary nodes in the mesh, adds them to a list if they are not already present,
        /// and finally returns a unique pointer to a vector containing all these nodes.
        /// \return Unique pointer to a vector of Node pointers that are on the boundary of the mesh
        unique_ptr<vector<Node*>> getBoundaryNodesVector();
        
        
        // Calculates the metrics of all the nodes based on the given coordinate system.
        // If coordinateSystem is Template then the metrics are calculated based on the template coordinate system before
        // the final coordinate system is calculated.
        // If coordinateSystem is Natural then the metrics are calculated based on the final calculated coordinate system.
        void calculateMeshMetrics(CoordinateType coordinateSystem, bool isUniformMesh);
        
        void initialize();

        shared_ptr<map<vector<double>, Node*>> getCoordinatesToNodesMap(CoordinateType coordinateType = Natural);

        unique_ptr<unordered_map<Node *, Position>> getBoundaryNodeToPositionMap() const;


        //-----------------Virtual parent class methods-----------------
        virtual unsigned dimensions();
        
        virtual SpaceEntityType space();
        
        virtual vector<Direction> directions();

        /// \brief Returns the number of internal (non boundary) nodes in the mesh
        /// \return An unsigned integer representing the total number of the internal nodes of the mesh
        /// \exception std::runtime_error Thrown when the Mesh has not been initialized
        virtual unsigned numberOfInternalNodes();
        
        virtual Node* node(unsigned i);
        
        virtual Node* node(unsigned i, unsigned j);
        
        virtual Node* node(unsigned i, unsigned j, unsigned k);
        
        virtual void printMesh();
        
        virtual NumericalVector<double> getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node);
        
        virtual unique_ptr<vector<Node*>> getInternalNodesVector();
        
        virtual void createElements(ElementType elementType, unsigned nodesPerEdge);

        virtual void storeMeshInVTKFile(const string& filePath, const string& fileName,
                                        CoordinateType coordinateType, bool StoreOnlyNodes) const;


    protected:
        
        shared_ptr<Array<Node*>>_nodesMatrix;
        
        unique_ptr<unordered_map<unsigned*, Node*>> _nodesMap;
        
        void _cleanMeshDataStructures();
        

        
        
        private:
        
        /// \brief Calculates the metrics of all the nodes of a uniformly spaced mesh.
        /// @param coordinateSystem _The coordinate system that the metrics will be calculated. If the metrics are calculated
        ///             during the mesh generation, then the coordinate system is Template. If the metrics are calculated for another
        ///             analysis, then the coordinate system is Natural. This is called twice for internal and boundary nodes 
        /// \param nodes A unique pointer to a vector of Node pointers that are either internal or boundary nodes
        void _arbitrarilySpacedMeshMetrics(CoordinateType coordinateSystem, unique_ptr<vector<Discretization::Node *>> nodes, bool areBoundary);
        
        virtual void _createNodeToIdMap();

        virtual void _addDBoundaryNodesToMap();

        virtual void _addTotalNodesToVector();
        

    };
}

#endif //UNTITLED_MESH_H
