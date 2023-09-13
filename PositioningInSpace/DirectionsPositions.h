//
// Created by hal9000 on 11/25/22.
//
#pragma once
#include <map>
#include <list>
#include <vector>
#include <set>
#include "../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalVector/NumericalVector.h"
using namespace std;
namespace PositioningInSpace {

    /// Directions of the simulation space.  Direction One can be x, ξ, r,
    /// Direction Two can be y, η, θ, Direction Three can be z, ζ, φ and
    /// Direction Time is the Time direction.
    enum Direction {
        One,
        Two,
        Three,
        Time,
        None
    };
    
    static map<Direction, unsigned> spatialDirectionToUnsigned =
        {{Direction::One,   0},
         {Direction::Two,   1},
         {Direction::Three, 2},
         {Direction::Time,  3},
         {Direction::None,  4}};
    
    static map<unsigned, Direction> unsignedToSpatialDirection =
        {{0, Direction::One},
         {1, Direction::Two},
         {2, Direction::Three}};


    //static list<Direction> directions = {Direction::One, Direction::Two, Direction::Three};

    // Relative Positions
    // Right represents a position to the right of the current position, which means an increase in the x-coordinate.
    // Left represents a position to the left of the current position, which means a decrease in the x-coordinate.
    // Front represents a position to the back of the current position, which means an increase in the z-coordinate.
    // Back represents a position to the front of the current position, which means a decrease in the z-coordinate.
    // Top represents a position on top of the current position, which means an increase in the y-coordinate.
    // Bottom represents a position at the bottom of the current position, which means a decrease in the y-coordinate.
    // The other positions represent combinations of the above movements in two or three directions and the multiplication
    // of their signs gives the sign of the position.
    enum Position {
        // i + 1, j, k -> (+)
        Right,
        // i - 1, j, k -> (-)
        Left,
        // i, j + 1, k -> (+)
        Top,
        // i, j - 1, k -> (-)
        Bottom,
        // i, j, k + 1 -> (+)
        Front,
        // i, j, k - 1 -> (-)
        Back,
        // i + 1, j + 1, k -> (+)
        RightTop,
        // i - 1, j + 1, k -> (-)
        LeftTop,
        // i + 1, j - 1, k -> (+)
        RightBottom,
        // i - 1, j - 1, k -> (-)
        LeftBottom,
        // i + 1, j, k + 1 -> (+)
        RightFront,
        // i - 1, j, k + 1 -> (-)
        LeftFront,
        // i, j + 1, k + 1 -> (+)
        TopFront,
        // i, j - 1, k + 1 -> (-)
        BottomFront,
        // i + 1, j + 1, k + 1 -> (+)
        RightTopFront,
        // i - 1, j + 1, k + 1 -> (-)
        LeftTopFront,
        // i + 1, j - 1, k + 1 -> (+)
        RightBottomFront,
        // i - 1, j - 1, k + 1 -> (-)
        LeftBottomFront,
        // i + 1, j, k - 1 -> (+)
        RightBack,
        // i - 1, j, k - 1 -> (-)
        LeftBack,
        // i, j + 1, k - 1 -> (+)
        TopBack,
        // i, j - 1, k - 1 -> (-)
        BottomBack,
        // i + 1, j + 1, k - 1 -> (+)
        RightTopBack,
        // i - 1, j + 1, k - 1 -> (-)
        LeftTopBack,
        // i + 1, j - 1, k - 1 -> (+)
        RightBottomBack,
        // i - 1, j - 1, k - 1 -> (-)
        LeftBottomBack,
    };

    static std::map<Position, NumericalVector<int>> normalUnitVectorsOfPositions = {
            {Right,           {1,  0,  0}},
            {Left,            {-1, 0,  0}},
            {Top,             {0,  1,  0}},
            {Bottom,          {0,  -1, 0}},
            
            {Front,           {0,  0,  1}},
            {Back,            {0,  0,  -1}},
            
            {RightTop,        {1,  1,  0}},
            {LeftTop,         {-1, 1,  0}},
            {RightBottom,     {1,  -1, 0}},
            {LeftBottom,      {-1, -1, 0}},
            {RightFront,      {1,  0,  1}},
            {LeftFront,       {-1, 0,  1}},
            {TopFront,        {0,  1,  1}},
            {BottomFront,     {0,  -1, 1}},
            {RightTopFront,   {1,  1,  1}},
            {LeftTopFront,    {-1, 1,  1}},
            {LeftBottomFront, {-1, -1, 1}},
            {RightBack,       {1,  0,  -1}},
            {LeftBack,        {-1, 0,  -1}},
            {TopBack,         {0,  1,  -1}},
            {BottomBack,      {0,  -1, -1}},
            {RightTopBack,    {1,  1,  -1}},
            {LeftTopBack,     {-1, 1,  -1}},
            {RightBottomBack, {1,  -1, -1}},
            {LeftBottomBack,  {-1, -1, -1}}
    };
    
    static map<Direction, map<Position, short int>> normalPositionSigns = {
            {One, {{Left, -1}, {Right, 1}}},
            {Two, {{Bottom, -1}, {Top, 1}}},
            {Three, {{Back, -1}, {Front, 1}}}
    };

    // Consider position vectors A, B, C. They are co-linear if and only if
    // AC x BC = 0
    static set<pair<Position, Position>> coLinearPositions() {
        list<list<Position>> coLinearPositions = {
                {Right,            Left},
                {Top,              Bottom},
                {Front,            Back},
                {RightTop,         LeftBottom},
                {RightBottom,      LeftTop},
                {RightFront,       LeftBack},
                {RightBack,        LeftFront},
                {TopFront,         BottomBack},
                {TopBack,          BottomFront},
                {RightTopFront,    LeftBottomBack},
                {RightTopBack,     LeftBottomFront},
                {RightBottomFront, LeftTopBack},
                {RightBottomBack,  LeftTopFront}
        };
        set<pair<Position, Position>> coLinearPositionsSet;
        for (auto &coLinearPosition : coLinearPositions) {
            coLinearPositionsSet.insert({coLinearPosition.front(), coLinearPosition.back()});
        }
        return coLinearPositionsSet;
    }
} // namespace PositioningInSpace
    
/*
    static list<Position> positions = {TopLeft, Top, TopRight,
                                       Left, Center, Right,
                                       BottomLeft, Bottom, BottomRight,
                                       FrontTopLeft, Position::FrontTop, Position::FrontTopRight,
                                       Position::FrontLeft, Position::Front, Position::FrontRight,
                                       Position::FrontBottomLeft, Position::FrontBottom, Position::FrontBottomRight,
                                       Position::BackTopLeft, Position::BackTop, Position::BackTopRight,
                                       Position::BackLeft, Position::Back, Position::BackRight,
                                       Position::BackBottomLeft, Position::BackBottom, Position::BackBottomRight};
    
*/



