import React, {Component } from "react";
import Node from "./Node";
export default class Pathfinding extends Component { 
    constructor() {
        super();
        this.state = {
            grid: [],
            mouseIsDown: false
        }
    }

    componentDidMount() {
        const newGrid = createGrid();
        this.setState(newGrid);
    }

    createGrid() {
        
    }

}