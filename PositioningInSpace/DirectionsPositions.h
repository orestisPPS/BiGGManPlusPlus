//
// Created by hal9000 on 11/25/22.
//
#pragma once
namespace PositioningInSpace {
    
    /// Directions of the simulation space.  Direction One can be x, ξ, r,
    /// Direction Two can be y, η, θ, Direction Three can be z, ζ, φ and
    /// Direction Time is the Time direction.
    enum Direction {
        One,
        Two,
        Three,
        Time
    };
    
    //static list<Direction> directions = {Direction::One, Direction::Two, Direction::Three};

    ///Relative Positions
    enum Position {
        TopLeft,
        Top,
        TopRight,
        Left,
        Center,
        Right,
        BottomLeft,
        Bottom,
        BottomRight,
        
        FrontTopLeft,
        FrontTop,
        FrontTopRight,
        FrontLeft,
        Front,
        FrontRight,
        FrontBottomLeft,
        FrontBottom,
        FrontBottomRight,
        
        BackTopLeft,
        BackTop,
        BackTopRight,
        BackLeft,
        Back,
        BackRight,
        BackBottomLeft,
        BackBottom,
        BackBottomRight
    };
    
/*
    static list<Position> positions = {Position::TopLeft, Position::Top, Position::TopRight,
                                       Position::Left, Position::Center, Position::Right,
                                       Position::BottomLeft, Position::Bottom, Position::BottomRight,
                                       Position::FrontTopLeft, Position::FrontTop, Position::FrontTopRight,
                                       Position::FrontLeft, Position::Front, Position::FrontRight,
                                       Position::FrontBottomLeft, Position::FrontBottom, Position::FrontBottomRight,
                                       Position::BackTopLeft, Position::BackTop, Position::BackTopRight,
                                       Position::BackLeft, Position::Back, Position::BackRight,
                                       Position::BackBottomLeft, Position::BackBottom, Position::BackBottomRight};
    
*/

    
/*    struct Directions{
        static list<Direction> getDirections(){
            return directions;
        }
    };
    
    struct Positions{
        static list<Position> getPositions(){
            return positions;
        }
    };*/
} // PositioningInSpace

