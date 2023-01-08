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
    
    enum PhysicalSpaceEntities {
        //Axis 1. Can be x,ξ,θ
        One_axis,
        //Axis 2, Can be y,η,φ
        Two_axis,
        //Axis_3, Can be z,ζ,r
        Three_axis,
        //The physical space defined by Axis 1 and Axis 2.
        OneTwo_plane,
        //The physical space defined by Axis 1 and Axis 3.
        OneThree_plane,
        //The physical space defined by Axis 2 and Axis 3.
        TwoThree_plane,
        //The 3D physical space.
        OneTwoThree_volume,
        //0D physical space. Enter the void.
        NullSpace
    };
    

} // PositioningInSpace

