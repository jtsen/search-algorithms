# Lab 1

This lab consists of three different graph search algorithms written with Python3 using standard libraries:
* Breadth First Search (BFS)
* Iterative Deepening Search (IDS) using a visited set
* A*

## Usage
To compile and run this program:
```bash
python3 path [-v] -start $start -goal $goal -alg $alg graph-file
```
Example:
```bash
python3 search.py -v -start S -goal G -alg BFS graph.txt
```
Where:
* the verbose [-v] flag is optional
* start-node and end-node are both alphabumeric strings
* algorithm = [BFS, ID, ASTAR] (all uppercase!)
* graph-file is the relative path and the text file containing the graph information, while absolute path will work as well.

Note:
* If running ID (Iterative Deepening), a depth flag and value is required:
```bash
python3 path [-v] -start $start -goal $goal -alg $alg -depth $depth graph-file
```
Where:
* Depth is an integer
  
Note 2:
* Depth flag can be included for other algorithms, it will simply be ignored. However, specifying ID as algorithm will require depth. The current implementation of the ID algorithm does not have a termination condition for failures (assumes there is always a path). Will need to manually terminate the process if that does not exist a path from the specified start to end node.