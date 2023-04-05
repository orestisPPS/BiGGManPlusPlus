//
// Created by hal9000 on 4/2/23.
//

#ifndef UNTITLED_ISOPARAMETRICDOFFINDER_H
#define UNTITLED_ISOPARAMETRICDOFFINDER_H

#include "../Analysis/AnalysisDOFs/AnalysisDegreesOfFreedom.h"
using namespace NumericalAnalysis;

namespace DegreesOfFreedom {

    class IsoParametricDOFFinder {
        
    public:
        explicit IsoParametricDOFFinder(AnalysisDegreesOfFreedom* analysisDOF, Mesh* mesh);
        
        map<Position, vector<DegreeOfFreedom*>> getNeighbourDOF(unsigned nodeId, unsigned depth);
        
        map<Position, vector<DegreeOfFreedom*>> getSpecificNeighbourDOF(unsigned nodeId, DOFType dofType, unsigned depth);
    
    private:
        AnalysisDegreesOfFreedom* _analysisDOF;
        
        Mesh* _mesh;

        
        map<vector<double>, Node*> _parametricCoordinatesToNodeMap;

        map<Node*, vector<double>> _nodeToParametricCoordinatesMap;
        
        map<DegreeOfFreedom*, vector<double>> _dofToParametricCoordinatesMap;

        map<vector<double>, Node*> _createParametricCoordinatesToNodeMap();
        
        map<Node*, vector<double>> _createNodeToParametricCoordinatesMap();
        
        map<DegreeOfFreedom*, vector<double>> _createDOFToParametricCoordinatesMap();
        
    };

} // DegreesOfFreedom

#endif //UNTITLED_ISOPARAMETRICDOFFINDER_H
