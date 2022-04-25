# Harpreet Dhunna
# CSCI 164 Search w/ Eight Puzzle (and 15-Puzzle) Project


# imports
from queue import PriorityQueue
import string
from typing import Tuple
import time




# StateDimension^2 gives us the total number of tiles in the puzzle
StateDimension = 3
# StateDimension = 4

# returns available actions for a state to take
Actions = lambda : ['u', 'd', 'l', 'r']
# Actions = lambda s: ['u', 'd', 'l', 'r']
# Opposite = dict([('u','d'), ('d','u'), ('l','r'), ('r','l'), (None, None)])


# returns the result of applying a move to a state
def Result(state, action):
    i = state.index('0')
    newState = list(state)
    row,col = i//StateDimension, i % StateDimension
    if (    (action=='u' and row==0) or
            (action=='d' and row==StateDimension-1) or
            (action=='l' and col==0) or
            (action=='r' and col==StateDimension-1)):
        return newState
    if action=='u':
        l,r = row*StateDimension+col, (row-1)*StateDimension+col
    elif action=='d':
        l,r = row*StateDimension+col, (row+1)*StateDimension+col
    elif action=='l':
        l,r = row*StateDimension+col, row*StateDimension+col-1
    elif action=='r' :
        l,r = row*StateDimension+col, row*StateDimension+col+1
    newState[l], newState[r] = newState[r], newState[l] 
    return ''.join(newState)

# applies a series of moves to a state
def ApplyMoves(actions, state):
    for action in actions:
        state = Result(state, action)
    return state

# displays current state to console
def PrintState(s):
    for i in range(0,len(s),StateDimension):
        print(s[i:i+StateDimension])

# returns whether a move is legal or not
def LegalMove(state, action):
    i = state.index('0')
    row,col = i//StateDimension, i % StateDimension
    if (    (action=='u' and row==0) or
            (action=='d' and row==StateDimension-1) or
            (action=='l' and col==0) or
            (action=='r' and col==StateDimension-1)):
        return False
    return True

## Additional methods
# returns all possible next legal states from a state
def GetPossibleStates(state:string) -> list[str]:
    return [Result(state, i) for i in Actions() if LegalMove(state, i)]

# returns the action taken between 2 states
def InferAction(lhs:string, rhs:string) -> string:
    states = GetPossibleStates(lhs)
    for a in Actions():
        res = Result(lhs, a)
        if res == rhs:
            return a
    return ''

# returns the actions taken with a list of states
def InferActionsFromStates(states:list[str]) -> str:
    actions = []
    for i in range(1, len(states)):
        actions.append(InferAction(states[i-1], states[i]))
    return ''.join(actions)




## Heuristics
# ManhattanDistance returns the sum of the manhattan distances of each tile from its goal state
def SingleTileManhattanDistance(tile, left, right):
    leftIndex = left.index(tile)
    rightIndex = right.index(tile)
    return (    abs(leftIndex//StateDimension - rightIndex//StateDimension) +
                abs(leftIndex%StateDimension - rightIndex%StateDimension))
  
def ManhattanDistance(left, right):
    distances = []
    if StateDimension == 4:
        distances = [SingleTileManhattanDistance(tile, left, right)
            for tile in ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']]
    else:
        distances = [SingleTileManhattanDistance(tile, left, right)
            for tile in [str(c) for c in range(1, StateDimension**2)]]
    # print ("Distances= ", distances)
    return sum(distances)

# returns the number of tiles not in their goal state
def OutOfPlace(left, right):
    distances = [right[i] != '0' and left[i] != right[i]
        for i in range(StateDimension**2)]
    return sum(distances)

def HammingDistance(lhs:str, rhs:str) -> int:
    if len(lhs) != len(rhs):
        print("STRINGS NOT EQUAL SIZE")
    return sum(c2 != '0' and c1 != c2 for c1, c2 in zip(lhs, rhs))








## Timer
class Timer():
    def __init__(self) -> None:
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        self.start_time = time.time()
        self.stop_time = 0

    def stop(self):
        self.stop_time = time.time()

    def getTime(self):
        return self.stop_time - self.start_time

    def reset(self):
        self.start_time = 0
        self.end_time = 0








### SEARCH ALORITHMS
## A*
# Data class for holding nodes
class State(object):
    def __init__(self, value:string, parent=False):
        self.parent = parent
        self.value = value
        # If a parent is passed in, this node's path become's parent's path
        # plus this state
        # Also, the actions are inherited from the parent, and a new one is
        # inferred to get to this state
        if parent:
            self.path = parent.path[:]
            self.actions = parent.actions[:]
            self.actions.append(InferAction(self.path[-1], value))
            self.path.append(value)
        else:
            self.path = [value]
            self.actions = []




# A* - Manhattan
def AStarManhattan(source_state:str, destination_state:str) -> str:
    # setup
    priority_queue = PriorityQueue()
    visited_queue = []
    path = []
    actions = []
    nodes_expanded = 0
    count = 0
    # seed frontier
    priority_queue.put( (0, 0, State(source_state)) )
    # run until the either we have a path or the frontier is empty
    while(path == [] and not priority_queue.empty()):
        # get node of least cost and add it to visited list
        current_child = priority_queue.get()[2]
        visited_queue.append(current_child.value)
        # return if goal is found
        if current_child.value == destination_state:
            print("Start state is same as goal state")
            return ""
        # generate child states from current state
        nodes_expanded += 1
        children = [State(s, current_child) for s in GetPossibleStates(current_child.value)]
        for child in children:
            if child.value not in visited_queue:
                # return if goal is found
                if child.value == destination_state:
                    path = child.path[:]
                    actions = child.actions[:]
                    break
                # get new costs for this child, and put it in the frontier
                count += 1
                manhattan_distance = ManhattanDistance(current_child.value, destination_state)
                priority_queue.put( (manhattan_distance, count, child) )
    # if entire search space has been exhausted without a solution, print error message
    if path == []:
        print("Could not find path!")
    # return solution
    print(f"nodes expanded: {nodes_expanded}")
    return ''.join(actions)




# A* - Out of Place (Hamming Distance)
def AStarHamming(source_state:str, destination_state:str) -> str:
    # setup
    priority_queue = PriorityQueue()
    visited_queue = []
    path = []
    actions = []
    nodes_expanded = 0
    count = 0
    # seed frontier
    priority_queue.put( (0, 0, State(source_state)) )
    # run until the either we have a path or the frontier is empty
    while(path == [] and not priority_queue.empty()):
        # get node of least cost and add it to visited list
        current_child = priority_queue.get()[2]
        visited_queue.append(current_child.value)
        # return if goal is found
        if current_child.value == destination_state:
            print("Start state is same as goal state")
            return ""
        # generate child states from current state
        nodes_expanded += 1
        children = [State(s, current_child) for s in GetPossibleStates(current_child.value)]
        for child in children:
            if child.value not in visited_queue:
                # return if goal is found
                if child.value == destination_state:
                    path = child.path[:]
                    actions = child.actions[:]
                    break
                # get new costs for this child, and put it in the frontier
                count += 1
                hamming_distance = HammingDistance(current_child.value, destination_state)
                priority_queue.put( (hamming_distance, count, child) )
    # if entire search space has been exhausted without a solution, print error message
    if path == []:
        print("Could not find path!")
    # return solution
    print(f"nodes expanded: {nodes_expanded}")
    return ''.join(actions)








## DFS, BFS
# Depth First Search
def DepthFirstSearch(source_state:str, destination_state:str) -> str:
    # init
    # works as stack
    frontier = [[source_state]]
    explored = {source_state}
    nodes_expanded = 0
    # while frontier isnt empty
    while len(frontier) > 0:
        # retrieve next node
        curr_path = frontier.pop(0)
        curr_state = curr_path[-1]
        # expand node
        children = GetPossibleStates(curr_state)
        nodes_expanded += 1
        # return if goal is found in children
        if destination_state in children:
            curr_path.append(destination_state)
            print(f"nodes expanded: {nodes_expanded}")
            return InferActionsFromStates(curr_path)
        # add paths of children to stack
        for child in children:
            if child not in explored:
                new_path = curr_path[:]
                new_path.append(child)
                frontier.insert(0, new_path)
                explored.add(child)
    # if our stack is empty, no path has been found
    print("Could not find path!")
    return ""




# Depth First Search
def BreadthFirstSearch(source_state:str, destination_state:str) -> str:
    # init
    # works as queue
    paths = [[source_state]]
    explored = {source_state}
    nodes_expanded = 0
    # while frontier isnt empty
    while len(paths) > 0:
        curr_path = paths.pop(0)
        curr_state = curr_path[-1]
        # expand node
        children = GetPossibleStates(curr_state)
        nodes_expanded += 1
        # return if goal is found
        if destination_state in children:
            curr_path.append(destination_state)
            print(f"nodes expanded: {nodes_expanded}")
            return InferActionsFromStates(curr_path)
        # add paths of children to queue
        for child in children:
            if not (child in explored):
                new_path = curr_path[:]
                new_path.append(child)
                paths.append(new_path)
                explored.add(child)
    # if our queue is empty, no path has been found
    print("Could not find path!")
    return ""








## Iterative Deepening
# Iterative Deepening Depth First Search
def IDDFS(source_state:str, destination_state:str, max_depth:int) -> str:
    nodes_expanded = 0
    # for an increasing amount of depth. up to max
    for i in range(max_depth):
        # return call to Helper function
        result, count = DepthFirstSearchID( source_state,
                                            destination_state,
                                            i)
        nodes_expanded += count
        # if a solution was found, return it
        if result != "":
            print(f"nodes expanded: {nodes_expanded}")
            return result
    # if we ran to max depth without finding a solution, return nothing
    print(f"nodes expanded: {nodes_expanded}")
    return "path not found"

def DepthFirstSearchID( source_state:str,
                        destination_state:str,
                        max_depth:int) -> Tuple[str, int]:
    #init
    nodes_expanded = 0
    frontier = [[source_state]]
    explored = []
    # while frontier isnt empty
    while (len(frontier) > 0):
        # retrieve
        curr_path = frontier.pop(0)
        curr_state = curr_path[-1]
        # return if goal is found
        if curr_state == destination_state:
            curr_path.append(destination_state)
            return (InferActionsFromStates(curr_path), nodes_expanded)
        # skip if max_depth has been reached
        if len(curr_path) > max_depth:
            continue
        # if curent state is new, add it to explored
        if curr_state not in explored:
            explored.append(curr_state)
            # expand
            children = GetPossibleStates(curr_state)
            nodes_expanded += 1
            # add paths to stack
            for child in children:
                new_path = curr_path[:]
                new_path.append(child)
                frontier.insert(0, new_path)
    # if our stack is empty, no path has been found
    return ("", nodes_expanded)




# Iterative Deepening for A* - Manhattan
def IDAStarManhattan(source_state:str, destination_state:str, max_depth:int) -> str:
    # init, these resources are shared by all calls to A*
    priority_queue = PriorityQueue()
    visited_queue = []
    nodes_expanded = 0
    curr_nodes_expanded = 0
    # seed frontier
    priority_queue.put( (0, 0, State(source_state)) )
    # for an increasing amount of depths, up to max
    for i in range(1, max_depth):
        result, curr_nodes_expanded = AStarID(  source_state,
                                                destination_state,
                                                ManhattanDistance,
                                                i,
                                                priority_queue,
                                                visited_queue)
        nodes_expanded += curr_nodes_expanded
        # if a solution was found, return it
        if result != "":
            print(f"nodes expanded: {nodes_expanded}")
            return result
    # if we ran to max depth without finding a solution, return nothing
    print(f"nodes expanded: {nodes_expanded}")
    return "path not found"


# Iterative Deepening for A* - Hamming
def IDAStarHamming(source_state:str, destination_state:str, max_depth:int) -> str:
    # init, these resources are shared by all calls to A*
    priority_queue = PriorityQueue()
    visited_queue = []
    nodes_expanded = 0
    curr_nodes_expanded = 0
    # seed frontier
    priority_queue.put( (0, 0, State(source_state)) )
    # for an increasing amount of depths, up to max
    for i in range(1, max_depth):
        result, curr_nodes_expanded = AStarID(  source_state,
                                                destination_state,
                                                OutOfPlace,
                                                i,
                                                priority_queue,
                                                visited_queue)
        nodes_expanded += curr_nodes_expanded
        # if a solution was found, return it
        if result != "":
            print(f"nodes expanded: {nodes_expanded}")
            return result
    # if we ran to max depth without finding a solution, return nothing
    print(f"nodes expanded: {nodes_expanded}")
    return "path not found"


# Generic version of A* that supports max_depth and any heuristic
def AStarID(    source_state:str,
                destination_state:str,
                heuristic,
                max_depth:int,
                ftr:PriorityQueue,
                exp:list[str],
                cnt:int=0   ) -> str:
    path = []
    actions = []
    count = 0
    # run until the either we have a path or the frontier is empty
    while(path == [] and not ftr.empty()):
        # get node of least cost and add it to visited list
        current_child = ftr.get()[2]
        exp.append(current_child.value)
        # return if goal is found
        if current_child.value == destination_state:
            print("Start state is same as goal state")
            return ""
        # generate child states from current state
        cnt += 1
        children = [State(s, current_child) for s in GetPossibleStates(current_child.value)]
        for child in children:
            if child.value not in exp:
                # return if goal is found
                if child.value == destination_state:
                    path = child.path[:]
                    actions = child.actions[:]
                    break
                # get new costs for this child, and put it in the frontier
                count += 1
                hueristic_cost = heuristic(current_child.value, destination_state)
                ftr.put( (hueristic_cost, count, child) )
    # if entire search space has been exhausted without a solution, print error message
    if path == []:
        print("Could not find path!")
    # return solution
    return (''.join(actions), cnt)








## Tests
GoalState3x3 = "123456780"
TestSuite3x3 = [
    "160273485", 
    "462301587", 
    "821574360", 
    "840156372", 
    "530478126", 
    "068314257", 
    "453207186", 
    "128307645", 
    "035684712", 
    "684317025", 
    "028514637", 
    "618273540", 
    "042385671", 
    "420385716", 
    "054672813", 
    "314572680", 
    "637218045", 
    "430621875", 
    "158274036", 
    "130458726"]
Discussion3x3 = [
    "123456078",
    "274651380"
]

GoalState4x4 = "123456789ABCDEF0"
TestSuite4x41 = [
    "16235A749C08DEBF", 
    "0634217859ABDEFC", 
    "012456379BC8DAEF", 
    "51246A38097BDEFC", 
    "12345678D9CFEBA0"]
TestSuite4x42 = [
    "71A92CE03DB4658F", 
    "02348697DF5A1EBC", 
    "39A1D0EC7BF86452", 
    "EAB480FC19D56237", 
    "7DB13C52F46E80A9"]
Discussion4x4 = [
    "12345E98F7DA60CB",
    "40A19738E65BDFC2"
]

# Testing Methods
def RunTests(test_suite:list[str], goal_state:str, algorithm):
    for test in test_suite:
        print("Test: " + test)
        print("Goal: " + goal_state)
        res = algorithm(test, goal_state)
        print(res); print('')

def RunTests(test_suite:list[str], goal_state:str, algorithm, timer):
    for test in test_suite:
        print("Test: " + test)
        print("Goal: " + goal_state)
        timer.reset()
        timer.start()
        res = algorithm(test, goal_state)
        timer.stop()
        print(res)
        print(f"Time taken: {timer.getTime()}"); print('')

def RunTestsID(test_suite:list[str], goal_state:str, algorithm, max_depth:int):
    for test in test_suite:
        print("Test: " + test)
        print("Goal: " + goal_state)
        res = algorithm(test, goal_state, max_depth)
        print(res); print('')








## Main function
if __name__ == "__main__":
    # print("\n\nstarting...\n")
    print("\n\nA* Manhattan\n")

    mytimer = Timer()

    # InitialState = "123405786"
    # InitialState = "821764350"
    # InitialState = "436501278"
    # InitialState = "821574360"
    # InitialState = "16235A749C08DEBF"
    # InitialState = "A931BC486E057D2F"
    # GoalState = "123456780"
    # GoalState = "123456789ABCDEF0"

    # 123
    # 405
    # 786






    # StateDimension = 3
    # StateDimension = 4

    # print('Start: ' + InitialState)
    # print('Goal: ' + GoalState)
    # print(IDDFS(InitialState, GoalState, search_depth) + "\n")
    # mytimer.start()
    # print(AStarManhattan(InitialState, GoalState) + "\n")
    # mytimer.stop()
    # print(f"Time taken: {mytimer.getTime()}")

    search_algorithm = AStarManhattan
    # search_depth = 100

    StateDimension = 3
    # RunTests(TestSuite3x3, GoalState3x3, search_algorithm)
    RunTests(Discussion3x3, GoalState3x3, search_algorithm, mytimer)
    # RunTestsID(TestSuite3x3, GoalState3x3, search_algorithm, search_depth)

    StateDimension = 4
    # RunTests(TestSuite4x41, GoalState4x4, search_algorithm)
    RunTests(Discussion4x4, GoalState4x4, search_algorithm, mytimer)
    # RunTestsID(TestSuite4x41, GoalState4x4, search_algorithm, search_depth)
    # RunTests(TestSuite4x42, GoalState4x4, search_algorithm)
    # RunTestsID(TestSuite4x42, GoalState4x4, search_algorithm, search_depth)

    print("============================")


# A* Manhattan

# Start: 123405786
# Goal: 123456780
# rd
# Time taken: 0.0010020732879638672

# Start: 16235A749C08DEBF
# Goal: 123456789ABCDEF0
# luurrddldr
# Time taken: 0.0020017623901367188

# Start: 436501278
# Goal: 123456780
# nodes expanded: 894
# ldrruuldlurddruldlurrdluulddrruldlurrdlluurddruulldrurdd
# Time taken: 0.04800152778625488

# Start: A931BC486E057D2F
# Goal: 123456789ABCDEF0
# nodes expanded: 18692
# dluuuldddrruluulddruurdllurrrddlurulldrulldrrulddrdruuldlurrddllluuurdldrdrulldrurdlurdrulldrurdlldrurddrdluurdldruulddruldlurrdluldluurdldrrruldlurrdllurdruuldlurrdd
# Time taken: 7.336951971054077
