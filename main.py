# Harpreet Dhunna
# CSCI 164 Search w/ Eight Puzzle (and 15-Puzzle) Project


from cmd import IDENTCHARS
from inspect import stack
from queue import PriorityQueue
import string
from typing import Tuple


# StateDimension^2 gives us the total number of tiles in the puzzle
StateDimension = 3
# StateDimension = 3
# StateDimension = 4

# InitialState = "120453786"
# InitialState = "102345678"
# InitialState = "123456708"

# # InitialState = "123456789A0BCDEF"
# InitialState = "16235A749C08DEBF"

# GoalState = "123456780"
# GoalState = "123456789ABCDEF0"

# Actions = lambda s: ['u', 'd', 'l', 'r']
Actions = lambda : ['u', 'd', 'l', 'r']
Opposite = dict([('u','d'), ('d','u'), ('l','r'), ('r','l'), (None, None)])


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





























class State(object):
    def __init__(self, value:string, parent=False):
        self.parent = parent
        self.value = value
        if parent:
            self.path = parent.path[:]
            self.actions = parent.actions[:]
            self.actions.append(InferAction(self.path[-1], value))
            self.path.append(value)
        else:
            self.path = [value]
            self.actions = []


class Stack():
    def __init__(self):
        self._stack = []
    
    def push(self, item):
        self._stack.insert(0, item)

    def peek(self):
        return self._stack[0]
    
    def pop(self):
        return self._stack.pop(0)
    
    def size(self):
        return len(self._stack)


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

































def DepthFirstSearch(source_state:str, destination_state:str) -> str:
    # init
    # works as stack
    frontier = [[source_state]]
    explored = {source_state}
    nodes_expanded = 0

    while len(frontier) > 0:
        curr_path = frontier.pop(0)
        curr_state = curr_path[-1]
        children = GetPossibleStates(curr_state)
        nodes_expanded += 1
        # return if goal is found
        if destination_state in children:
            curr_path.append(destination_state)
            print(f"nodes expanded: {nodes_expanded}")
            return InferActionsFromStates(curr_path)
        # add paths to queue
        for child in children:
            if child not in explored:
                new_path = curr_path[:]
                new_path.append(child)
                frontier.insert(0, new_path)
                explored.add(child)
    
    # if our queue is empty, no path has been found
    print("Could not find path!")
    return ""




def BreadthFirstSearch(source_state:str, destination_state:str) -> str:
    # init
    paths = [[source_state]]
    explored = {source_state}
    nodes_expanded = 0

    while len(paths) > 0:
        curr_path = paths.pop(0)
        curr_state = curr_path[-1]
        children = GetPossibleStates(curr_state)
        nodes_expanded += 1
        # return if goal is found
        if destination_state in children:
            curr_path.append(destination_state)
            print(f"nodes expanded: {nodes_expanded}")
            return InferActionsFromStates(curr_path)
        # add paths to queue
        for child in children:
            if not (child in explored):
                new_path = curr_path[:]
                new_path.append(child)
                paths.append(new_path)
                explored.add(child)
    
    # if our queue is empty, no path has been found
    print("Could not find path!")
    return ""




























def IterativeDeepeningDepthFirstSearch(source_state:str, destination_state:str, max_depth:int) -> str:
    # init
    f_stack = [[source_state]]
    explored = {source_state}
    nodes_expanded = 0

    for i in range(max_depth):
        res, count = DepthLimitedSearch(source_state, destination_state, i)
        nodes_expanded += count
        if(res != ""):
            print(f"nodes expanded: {nodes_expanded}")
            return "path found"
    print(f"nodes expanded: {nodes_expanded}")
    return "path not found"


def DepthLimitedSearch(source_state:str, destination_state:str, max_depth:int) -> Tuple[str, int]:
    nodes_expanded = 0
    if source_state == destination_state: return ("path found", nodes_expanded)
    if max_depth <= 0: return ("", nodes_expanded)
    children = GetPossibleStates(source_state)
    nodes_expanded += 1
    for child in children:
        res, count = DepthLimitedSearch(child, destination_state, max_depth-1)
        nodes_expanded += count
        if(res != ""):
            return ("path found", nodes_expanded)
    return ("", nodes_expanded)

def IterativeDeepening(search_algorithm, source_state:str, destination_state:str, max_depth:int) -> str:
    nodes_expanded = 0
    frontier = [[source_state]]
    explored = {source_state}
    for i in range(1, max_depth):
        result, count = search_algorithm(destination_state, i, frontier, explored)
        nodes_expanded += count
        if result != "":
            print(f"nodes expanded: {nodes_expanded}")
            return result
    print(f"nodes expanded: {nodes_expanded}")
    return "path not found"
    

# init stk with [[source_state]]
# init exp with {source_state}
def DepthFirstSearchID(destination_state:str, max_depth:int, ftr:list[list[str]], exp:set, cnt:int=0) -> Tuple[str, int]:
    while True:
        curr_path = ftr.pop(0)
        curr_state = curr_path[-1]
        children = GetPossibleStates(curr_state)
        cnt += 1
        # return if goal is found
        if destination_state in children:
            curr_path.append(destination_state)
            return (InferActionsFromStates(curr_path), cnt)
        # returnif max_depth has been reached
        if len(curr_path) > max_depth:
            return ("", cnt)
        # add paths to stack
        for child in children:
            if child not in exp:
                new_path = curr_path[:]
                new_path.append(child)
                ftr.insert(0, new_path)
                exp.add(child)
        # do-while pattern
        if(len(ftr) > 0): break
    # if our stack is empty, no path has been found
    return ("", cnt)








# Iterative Deepening for A*
def IDAStarManhattan(source_state:str, destination_state:str, max_depth:int) -> str:
    priority_queue = PriorityQueue()
    visited_queue = []
    nodes_expanded = 0
    curr_nodes_expanded = 0
    # seed frontier
    priority_queue.put( (0, 0, State(source_state)) )

    for i in range(1, max_depth):
        result, curr_nodes_expanded = AStarID(  source_state,
                                                destination_state,
                                                ManhattanDistance,
                                                i,
                                                priority_queue,
                                                visited_queue)
        nodes_expanded += curr_nodes_expanded
        if result != "":
            print(f"nodes expanded: {nodes_expanded}")
            return result
    print(f"nodes expanded: {nodes_expanded}")
    return "path not found"

def IDAStarHamming(source_state:str, destination_state:str, max_depth:int) -> str:
    priority_queue = PriorityQueue()
    visited_queue = []
    nodes_expanded = 0
    curr_nodes_expanded = 0
    # seed frontier
    priority_queue.put( (0, 0, State(source_state)) )

    for i in range(1, max_depth):
        result, curr_nodes_expanded = AStarID(  source_state,
                                                destination_state,
                                                OutOfPlace,
                                                i,
                                                priority_queue,
                                                visited_queue)
        nodes_expanded += curr_nodes_expanded
        if result != "":
            print(f"nodes expanded: {nodes_expanded}")
            return result
    print(f"nodes expanded: {nodes_expanded}")
    return "path not found"

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

















def dfs2(source_state:str, destination_state:str):
    frontier = [[source_state]]
    explored = [source_state]
    # if goal then return
    if source_state == destination_state:
        print("src same as dst")
        return ""
    while len(frontier):
        curr_path = frontier.pop(0)
        curr_state = curr_path[-1]
        children = GetPossibleStates(curr_state)
        for child in children:
            new_path = curr_path[:]
            new_path.append(child)
            if child == destination_state:
                curr_path.append(destination_state)
                return InferActionsFromStates(curr_path)
            if new_path not in explored:
                explored.append(new_path)
                frontier.insert(0, new_path)
    return "failure"














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

def RunTests(test_suite:list[str], goal_state:str, algorithm):
    for test in test_suite:
        print("Test: " + test)
        print("Goal: " + goal_state)
        res = algorithm(test, goal_state)
        print(res); print('')

def RunTestsID(test_suite:list[str], goal_state:str, algorithm, max_depth:int):
    for test in test_suite:
        print("Test: " + test)
        print("Goal: " + goal_state)
        res = algorithm(test, goal_state, max_depth)
        print(res); print('')


# def foo():
#     my_set = {8,9}
#     bar(my_set)
#     return my_set

# def bar(val):
#     val.add(1)


## Main function
if __name__ == "__main__":
    print("\n\nstarting...\n")

    # InitialState = "123405786"
    # InitialState = "130458726"
    # InitialState = "821574360"
    # GoalState = "123456780"
    # mod_state = ApplyMoves(['u','u','l','d','d','r'], InitialState)

    # 123
    # 405
    # 786

    # StateDimension = 3
    search_depth = 6

    # print('Start: ' + InitialState)
    # print('Goal: ' + GoalState)
    # print(IDAStarManhattan(InitialState, GoalState, search_depth) + "\n")
    # print(dfs2(mod_state, GoalState) + "\n")

    search_algorithm = IDAStarHamming

    StateDimension = 3
    # RunTests(TestSuite3x3, GoalState3x3, search_algorithm)
    RunTestsID(TestSuite3x3, GoalState3x3, search_algorithm, search_depth)

    StateDimension = 4
    # RunTests(TestSuite4x41, GoalState4x4, search_algorithm)
    RunTestsID(TestSuite4x41, GoalState4x4, search_algorithm, search_depth)
    # RunTests(TestSuite4x42, GoalState4x4, search_algorithm)
    RunTestsID(TestSuite4x42, GoalState4x4, search_algorithm, search_depth)

    print("============================")

