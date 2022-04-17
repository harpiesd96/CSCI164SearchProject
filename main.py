# Harpreet Dhunna
# CSCI 164 Search w/ Eight Puzzle (and 15-Puzzle) Project


import heapq
from operator import concat
from queue import PriorityQueue
import string
from tracemalloc import start
from turtle import distance


# StateDimension^2 gives us the total number of tiles in the puzzle
# StateDimension = 3
StateDimension = 4

# InitialState = "120453786"
# InitialState = "102345678"
# InitialState = "123456708"

# InitialState = "123456789A0BCDEF"
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

# returns all possible next legal states from a state
def GetPossibleStates(state:string):
    return [Result(state, i) for i in Actions() if LegalMove(state, i)]

# returns the action taken between 2 states
def InferAction(lhs:string, rhs:string) -> string:
    states = GetPossibleStates(lhs)
    for a in Actions():
        res = Result(lhs, a)
        if res == rhs:
            return a
    return ''




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
    distances = [left[i]!=right[i] and right[i] != '0'
        for i in range(StateDimension**2)]
    return sum(distances)
















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
TestSuite4x4 = [
    "16235A749C08DEBF", 
    "0634217859ABDEFC", 
    "012456379BC8DAEF", 
    "51246A38097BDEFC", 
    "12345678D9CFEBA0", 
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



if __name__ == "__main__":
    print("\n\nstarting...\n")

    # print('Start: ' + InitialState)
    # print('Goal: ' + GoalState)
    # print(AStarManhattan(InitialState, GoalState) + "\n")
    StateDimension = 3
    # print(AStarManhattan("130458726", "123456780") + "\n")

    RunTests(TestSuite3x3, GoalState3x3, AStarManhattan)

    print("============================")
