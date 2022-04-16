# Harpreet Dhunna
# CSCI 164 Search w/ Eight Puzzle (and 15-Puzzle) Project


import heapq


# StateDimension^2 gives us the toal number of tiles in the puzzle
StateDimension = 3
# StateDimension = 4

InitialState = "123450678"
# InitialState = "123456789A0BCDEF"

GoalState = "123456780"
# GoalState = "123456789ABCDEF0"

Actions = lambda s: ['u', 'd', 'l', 'r']
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

# ManhattanDistance returns the sum of the manhattan distances of each tile from its goal state
def SingleTileManhattanDistance(tile, left, right):
    leftIndex = left.index(tile)
    rightIndex = right.index(tile)
    return (    abs(leftIndex//StateDimension - rightIndex//StateDimension) +
                abs(leftIndex%StateDimension - rightIndex%StateDimension))
  
def ManhattanDistance(left, right):
    distances = [SingleTileManhattanDistance(tile, left, right) 
        for tile in [str(c) for c in range(1, StateDimension**2)]]
    # print ("Distances= ", distances)
    return sum(distances)

# returns the number of tiles not in their goal state
def OutOfPlace(left, right):
    distances = [left[i]!=right[i] and right[i] != '0'
        for i in range(StateDimension**2)]
    return sum(distances)




PrintState(InitialState)
print("ManhattanDistance =  ", ManhattanDistance(InitialState, GoalState))
print("OutOfPlace = ", OutOfPlace(InitialState, GoalState))

# PrintState("103526478")
# NewState = ApplyMoves("dldrr", "103526478")
# PrintState(NewState)