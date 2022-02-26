import sys, math, copy, argparse

class Node:
    """
    Node object
    :params:
    x (int) -> x coordinate on 2D plane
    y (int) -> y coordinate on 2D plane
    visited (bool) -> whether a node has been visited
    h (float) -> heuristic value of node to specified end node
    """
    def __init__(self, name, x, y, visited=False):
        self.name = name
        self.x = x
        self.y = y
        self.visited = visited
        self.h = 0.0
    
    def get_coordinates(self):
        """Get the coordinates of the node as a string"""
        return "[" + str(self.x) + ", " + str(self.y) + "]"
    
    def get_name(self):
        """Get the node label of the node"""
        return self.name

class Edge:
    """
    Edge object
    :params:
    n1 (Node) -> the "from" node
    n2 (Node) -> the "to" node
    length (Float) -> Euclidean distance between the two nodes the edge connects
    """
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.length = round(math.sqrt((float(self.n1.x) - float(self.n2.x))**2 + (float(self.n1.y) - float(self.n2.y))**2),2)
    
    def get_nodes(self):
        """Get the nodes labels of the nodes the edge connects as a string"""
        return "[" + self.n1.get_name() + ", " + self.n2.get_name() + "]"

    def get_length(self):
        """Get the edge's length"""
        return self.length


class Graph:
    """
    Graph object
    Args:
        nodes ([Node]): [list of nodes in the graph]
        edges ([Edge]): [list of edges in the graph]
    """
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node (self, node):
        """add a node to the graph if the node label doesn't already exist"""
        if node.name:
            return self.nodes.append(node)

    def add_edge (self, edge):
        """add an edge to the graph if the nodes it connects are in the graph"""
        if edge.n1 in self.nodes and edge.n2 in self.nodes:
            return self.edges.append(edge)

    def print_graph(self):
        """Print node labels and edges in as strings"""
        all_nodes = (node.name for node in self.nodes)
        all_edges = ("[" + edge.n1.name + "," + edge.n2.name + "]" for edge in self.edges)
        print("nodes: {}".format(list(all_nodes)))
        print("edges: {}".format(list(all_edges)))

    def find_node(self, name):
        """
        Find and return a Node in the Graph

        Args:
            name (str): [Node label]

        Returns:
            Node : [Node object]
            None : [None onject]
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def find_edge(self, name):
        """
        Find all all going edges of a node

        Args:
            name (str): [node label of a node]

        Returns:
            Edge: [list of outgoing edges of the specified node sorted Alphabetically]
        """
        edge_list = []
        for edge in self.edges:
            if edge.n1.name == name:
                edge_list.append(edge) 
        #https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
        newlist = sorted(edge_list, key=lambda edge: edge.n2.name)
        return newlist
    
    def find_specific_edge(self, n1, n2):
        """Given two node labels (to and from, in that order), return the edge"""
        for edge in self.edges:
            if edge.n1.name == n1 and edge.n2.name == n2:
                return edge

    def clear_visited(self):
        """mark all the nodes in the graph as not visited"""
        for node in self.nodes:
            if node.visited:
                node.visited = False  

def build_graph(input):
    """
    This function creates the graph from a given graph file]

    Args:
        input ([str]): [a list of str read from the graph file]

    Returns:
        Graph : [Graph object built according to the input]
    """
    graph = Graph()
    for item in input:
        if any(c.isdigit() for c in item) and len(item.split())==3: #if the line has three strings (for node label, x, y) extra check for digits just in case
            curr_node_detail = item.split() #split the input into lists
            graph.add_node(Node(curr_node_detail[0], #instantiate Node object according to specifications
                            int(curr_node_detail[1]), 
                            int(curr_node_detail[2])))
        elif len(item.split()) == 2: #if the line has two strings, its probably an edge
            curr_edge_detail = item.split() #split the input
            n1, n2 = graph.find_node(curr_edge_detail[0]), graph.find_node(curr_edge_detail[1]) #make sure that the nodes the edge connects exists in the graph already
            if n1 and n2: #if both of the end nodes of the edge exists
                new_edge = Edge(n1, n2) #Instantiate Edge object
                new_edge_flipped  = Edge(n2, n1) #intantiate Edge object in reverse due to undirected edges
                graph.add_edge(new_edge) #add the first Edge object
                graph.add_edge(new_edge_flipped) #add the second
            else: #if at least one of the nodes that the edge connects does not exist in the graph, exit gracefully.
                print("Bad graph file: Edge referencing a vertex not in the file")
                sys.exit()
    return graph

def euclidean(x1, x2, y1, y2):
    """
    Calculate the Euclidean distance of two Node objects

    Args:
        x1 (int): [x coordinate of Node 1]
        x2 (int): [x coordinate of Node 2]
        y1 (int): [y coordinate of Node 1]
        y2 (int): [y coordinate of Node 2]

    Returns:
        Float : [the euclidean distance computed]
    """
    return round(math.sqrt(((float(x1) - float(x2))**2) + ((float(y1) - float(y2))**2)), 2)

def build_heuristics(graph, end):
    """
    Function to add h values to all of the Node objects in a Graph object
    by calculating the node and the goal node's euclidean distance.

    Args:
        graph (Graph): [A built graph objects]
        end (Node): [The goal node object]

    Returns:
        Graph : [Graph object with every node's h value computed]
        None : [None object if the end node does not exist in the graph given]
    """
    if end:
        for node in graph.nodes:
            node.h = euclidean(node.x, end.x, node.y, end.y)
        return graph
    else:
        print("The specified end node does not exist in the graph.")
        return None

# start and end should both be of Node type
def bfs(graph, start, end, verbose):
    """
    Breadth First Search Algorithm

    Args:
        graph (Graph): [a built Graph object]
        start (Node): [the start node specified]
        end (Node): [the end node specified]
        verbose (Bool): [Verbose output / or not]

    Returns:
        ([Node]): [a list of nodes that the path consists of]
    """
    queue = [start] #instantiate a queue with the start node
    result_path = [] #instantiate a result list
    prev = {} #instantiate a dictionary mapping previous nodes during traversal
    while queue: #while the queue is not empty
        curr_node = queue.pop(0) #grab the first thing out of the queue
        if curr_node.visited: #if the node has been visited then we skip to the next node in the queue
            continue
        if curr_node.name == end.name: #if the current node is the goal node
            node_name = curr_node.name #populate the result list with nodes from path
            result_path.insert(0, curr_node.name)
            while node_name != start.name:
                result_path.insert(0, prev[node_name]) #trace back to how the traversal got to the end
                node_name = prev[node_name]
            print("Solution:", end=" ") #Printing the result path
            for node_name in result_path:
                if node_name == end.name:
                    print(node_name)
                else:
                    print(node_name, end = " -> ")
            return result_path
        curr_node.visited = True #set the current node as visited
        if verbose:
            print("Expanding: {}".format(curr_node.name)) # print current node if verbose output is specified
        candidates_edges = graph.find_edge(curr_node.name) # find all edges traversable (neighbors) from current node
        for edge in candidates_edges:
            if edge.n2.visited: #if the node has been visited, skip it
                continue
            else: # otherwise, add it into the queue and record current node as its predecessor
                if edge.n2.name not in prev.keys():
                    prev[edge.n2.name] = curr_node.name
                queue.append(edge.n2)
    print("No solution found.") #if the while loop exits, we know all nodes have been visited and that there does not exist a path.
    return result_path

# still need a terminating condition -> [Failure, Limited, Success]
def ids(graph, start, end, depth, max_depth, verbose):
    """
    Iterative Deepening Search Algorithm

    Args:
        graph (Graph): [a built graph from the graph file]
        start (Node): [the starting Node object]
        end (Node): [the goal Node object]
        depth (int): [depth to start; Driver call is always 0]
        max_depth (int): [max depth the algorithm in allowed]
        verbose (bool): [verbose output to stdout / or not]

    Returns:
        [str]: [list of Node objects' labels consisting of the result path]
    """
    start.visited = True #mark the start node as visited
    if start == end: #if the start node is the goal node, we return the node label in the list
        return [start.name]
    if depth == max_depth: #if we hit maximum depth, we return an empty list as there is no path found
        if verbose:
            print("hit depth={}:{}".format(depth, start.name)) #verbose output to indicate which node hit the depth limit
        return []
    if verbose:
        print("Expand: {}".format(start.name)) #verbose output to indicate the current node being expanded
    candidate_edges = graph.find_edge(start.name) #find the outgoing edges of the current node
    for edge in candidate_edges:
        if edge.n2.visited: #if the "to" Node has already been visited, skip it
            continue
        result = ids(graph, edge.n2, end, depth+1,max_depth, verbose) #otherwise, recursively call this function on the "to" Node and add 1 to depth param
        if result:
            result.insert(0, start.name) #adding current Node to the result path if one of the child processes reaches the goal node successfully
            return result
    return [] #no path were found, return empty list.
    
def astar(graph, start, end, verbose):
    """
    A* search algorithm 

    Args:
        graph (Graph): [a built Graph object]
        start (Node): [the start node specified]
        end (Node): [the end node specified]
        verbose (Bool): [Verbose output / or not]

    Returns:
        ([Node]): [a list of nodes that the path consists of]
    """
    results = [start.name]
    curr = start
    costs = [] #create cost array (priority queue), sorted at each step to get the minimum cost associated with a path (cost,path):tuple
    total_cost = 0.0 #running current total cost associated with a path being considered
    if start == end: #if the start node is the end node, return it in a list
        return [start]
    while results[-1] != end.name: #while the goal node is not the last node added to the results
        candidate_edges = graph.find_edge(curr.name) #find neighbors of current Node object
        for edge in candidate_edges:
            curr_result = copy.deepcopy(results) #create a deep copy of the current result list
            if verbose: #verbose output that shows the computations of g and h of each path
                print("{} -> {} ; g={} h={} = {}".format(edge.n1.name,
                                                         edge.n2.name,
                                                         round(edge.length+total_cost,2),
                                                         edge.n2.h,
                                                         round((edge.length+total_cost+edge.n2.h),2)))
            curr_result.append(edge.n2.name) #append "to" Node to the current copy of the result list to create a new path to be considered
            costs.append((round((edge.length + total_cost + edge.n2.h),2), curr_result)) #append the path and the cost (g+h) of the current path
        #https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        #https://www.geeksforgeeks.org/python-ways-to-sort-list-of-float-values/
        costs = sorted(costs) #sort by the cost (priority) to order the cost list
        selected = costs.pop(0) #remove the first item from priority queue (cost list)
        total_cost = selected[0] #the cost (g+h) of the current selected path
        path = selected[1] #the path as a list of Node objects
        if path[-1] == end.name: #if the last node is the goal node then return the results since its selected already
            results = path
            break
        #Assuming no negative edges, picking a new path that revisit the start node means no path found.
        if path.count(start.name) > 1:
            print("No solution found.")
            return []
        if verbose:
            print("adding", end=" ") #verbose output of path being considered
            for node in path:
                if node == path[-1]:
                    print(node)
                else:
                    print(node, end = " -> ")
        n1, n2 = path[-2:][0], path[-2:][1] #get to current node and the node it connects to
        edge_to_add = graph.find_specific_edge(n1,n2) #find the corresponding edge
        node_to_add = edge_to_add.n2 #the node to be added to the current path
        total_cost -= node_to_add.h #remove the heuristics of the "to" Node (last Node) since we took the edge
        curr = graph.find_node(path[-1]) #switch current to the "to" Node (last Node) of the selected path
        results = path #current running result is the one chosen with the lowest cost
    print("Solution:", end=" ") #once the goal state is reached, print the results to stdout
    for node in results:
        if node == end.name:
            print(node)
        else:
            print(node, end = " -> ")
    return results
    
if __name__ == "__main__":
    #https://docs.python.org/3/library/argparse.html
    #https://mkaz.blog/code/python-argparse-cookbook/
    parser = argparse.ArgumentParser() #create the argument parser object and add the required and optional flags and required graph file
    parser.add_argument("-v", "--verbose", action='store_true', help="a flag that turns verbose on when used")
    parser.add_argument("-start", "--start", dest="start", default=None, help="node label of the start node")
    parser.add_argument("-goal", "--goal", dest="end", default=None, help="node label of the goal node")
    parser.add_argument("-alg", "--alg", dest="alg", default=None ,help="acronym of algorithm to be chosen")
    parser.add_argument("-depth", "--depth", nargs='?', default=None, help="depth parameter if running IDS")
    parser.add_argument("graphfile")

    args = parser.parse_args() #create object that holds all the command-line arguments as instance variables that are accessable 

    if args.alg == 'ID' and not args.depth: #if the current specified algorithm is iterative deepening but depth is not provided, exit gracefully
        print("Please pass in a depth param for IDS: -depth [int]")
        sys.exit()

    try:#Open and read the graph file provided while skipping all the lines that are "commented out into lists of strings"
        with open(args.graphfile, 'r') as reader:
                input = reader.read().split('\n')
                input_copy = copy.deepcopy(input)
                for line in input_copy:
                    if line.startswith('#') or line == '':
                        input.remove(line)
                reader.close()
    except FileNotFoundError: #if the file could not be found, exit gracefully
        print("Invalid graph file: File could not be found. (Check spelling)")
        sys.exit()

    graph = build_graph(input)#build the graph from the graph file
    
    if args.start: #check to see if a start node label is provided
        if args.start.isalnum():#check the start argument to make sure its alphanumerical
            start_node = graph.find_node(args.start) #find the corresponding Node object
    else:#if no start argument provided, exit gracefully
        print("Please pass in an alphanumeric start node!")
        sys.exit()

    if args.end:#check to see if a goal node label is provided
        if args.end.isalnum(): #check the goal argument to make sure its alphanumerical
            end_node = graph.find_node(args.end) #find the corresponding Node object
    else:#if no goal argument provided, exit gracefully
        print("Please pass in an alphanumeric end node!")
        sys.exit()

    if not start_node:#if the start node is not found in the graph (returned None), exit gracefully
        print("Start node does not exist in the graph...")
        sys.exit()
    if not end_node: #if the goal node is not found in the graph (returned None), exit gracefully
        print("End node does not exist in the graph...")
        sys.exit()
    
    if args.alg: #if an algorithm argument is provided, switch between the driver calls according to the specified algorithm
        if args.alg == 'BFS': #BFS driver code
            path = bfs(graph, start_node, end_node, args.verbose)
        elif args.alg == 'ID': #Iterative Deepening driver code
            if not args.depth.isdigit(): #if the depth is not a digit, exit gracefully
                print("Please pass in an integer for depth!")
                sys.exit()
            depth = int(args.depth)
            path = ids(graph, start_node, end_node, 0, depth, args.verbose)#first driver call with the given max depth
            while not path:#while the return is empty list
                depth+=1# add 1 to max depth
                graph.clear_visited() #clear the visited attribute in the Graph's Node objects
                path = ids(graph, start_node, end_node, 0, depth, args.verbose) #call the algorithm again with new depth from the beginning
            print("Solution:", end=" ") #print the solution of iterative deepening
            for node_name in path:
                if node_name == end_node.name:
                    print(node_name)
                else:
                    print(node_name, end = " -> ")
        elif args.alg == "ASTAR": #if A* is specified
            graph = build_heuristics(graph, end_node) #build the heuristics by adding h values to every Node object using Euclidean Distance
            if graph: #if heuristics building was successful
                path = astar(graph, start_node, end_node, args.verbose) #A* algorithm driver 
        else: #if the algorithm argument is passed in incorrectly or specified one that was not implemented, exit gracefully
            print("The algorithm specified is not implemented...")
            print("Please use one of the following: [BFS, ID, ASTAR]")
            sys.exit()
    else: #if no algirthm argument is provided, exit gracefully
        print("Please specify an algorithm to run from this list: [BFS, ID, ASTAR]")
        sys.exit()