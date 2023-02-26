# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import itertools
from enum import Enum

from numpy import sort

import capture
from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions
import game
import math


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='AstarAgent', second='AstarAgent', numTraining=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    code_version = "v2.5"
    print("MDJ2 Code Version: ", code_version)
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


# Useful Functions to lower the length of references
def getAgentState(gameState, index):
    return gameState.getAgentState(index)


# Agent
class AstarAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Assign varaibles that we will use a lot to save resources on accessing other classes.
        self.red = gameState.isOnRedTeam(self.index)
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        self.walls = gameState.getWalls().asList()
        self.startPosition = gameState.getAgentState(self.index).getPosition()
        self.captureList = self.getFood(gameState).asList()
        self.defenderList = self.getFoodYouAreDefending(gameState).asList()
        self.maxTime = gameState.data.timeleft
        self.evacuationTime = self.width * 5
        self.deaths = 0
        self.resetGoal = False
        self.enemySpace = capture.halfGrid(capture.Grid(self.width, self.height, True),
                                           red=False if self.index in gameState.getRedTeamIndices() else False)
        self.teamSpace = self.getDefendingPositions(gameState).asList()
        self.deadEnds = self.initDeadEnds()

        self.tempGoal = None
        self.foodList = self.getFood(gameState).asList()

        self.enemyFoodClusterValue = self.foodCluster(gameState)

        # Ensure we have an even split of defenders and attackers
        team = self.getTeam(gameState)
        divider = len(team) // 2
        defenderTeam, offensiveTeam = team[:divider], team[divider:]
        self.teamIndex = team
        self.defender = True if defenderTeam.__contains__(self.index) else False

        # Is our agent currently an attack because they are scared?
        self.isAttackerFromFear = False

        # Reflects the value of the upper or lower portions This will need to be changed so that we intially use
        # the cluster heuristic meaning it's good to be near clusters of food.
        self.goal = self.resetGoalPosition(gameState)
        self.investigating = False

    def convertTuple(self, tup):
        # initialize an empty string
        s = ''
        for item in tup:
            s = s + str(item)
        return s

    def chooseAction(self, gameState):
        agentPos = getAgentState(gameState, self.index).getPosition()
        enemyPositions = self.meaningfulPacmanPositions(gameState)

        if self.detectFoodEaten():
            self.foodList = self.getFood(gameState).asList()

        if self.getPreviousObservation() is not None:
            if len(self.foodList) != len(self.getFood(self.getPreviousObservation()).asList()):
                self.enemyFoodClusterValue = self.foodCluster(gameState)

        # check if goal has been reached than reset goal? initial travel to the middleish of the map
        if agentPos == self.goal:
            # # print("Resetting goal action")
            self.goal = None

        if self.goal is not None:
            # # print("Traveling to Goal: ", self.goal, agentPos)
            return self.aStar(gameState, self.goal, self.distHeuristic)

        # if death has happened multiple times reset the goal position (may need to make this better)
        if self.checkIfAgentDied(self.index):
            self.resetGoal = True

        agentMidlinePoint = self.getNearestMidlinePoint(agentPos)
        for index in self.teamIndex:
            if index != self.index:
                if (self.checkIfAgentDied(self.index) and not self.checkIfAgentDied(index)) or (not self.checkIfAgentDied(self.index) and self.checkIfAgentDied(index)):
                    # We have died and we're checking if our teammate hasn't.
                    self.defender = not self.defender

        # This is to swith a defending agent to be an attacker if it's not able to do anything.
        self.isAttackerFromFear = gameState.getAgentState(self.index).scaredTimer > 0 and len(
            self.getFoodYouAreDefending(gameState).asList()) > (len(self.defenderList) / 2)

        # If we are a defender than we use defensive actions
        if self.defender and not self.isAttackerFromFear:
            return self.getDefensiveAction(gameState)

        # Otherwise offensive actions, this is a touch cleaner than using separate classes
        return self.getOffensiveAction(gameState)

    def resetGoalPosition(self, gameState):
        eupperFoodCount = 0
        elowerFoodCount = 0
        tupperFoodCount = 0
        tlowerFoodCount = 0
        foodList = self.getFood(gameState).asList()
        for coord in foodList:
            x, y = coord
            if y >= self.height / 2:
                if self.isPositionInTeamArea(gameState, coord):
                    tupperFoodCount += 1
                else:
                    eupperFoodCount += 1
            else:
                if self.isPositionInTeamArea(gameState, coord):
                    tlowerFoodCount += 1
                else:
                    elowerFoodCount += 1

        lowerMid, higherMid = self.middlePosSplit()
        return random.choice((lowerMid if tlowerFoodCount > tupperFoodCount else higherMid) if self.defender else (
            higherMid if eupperFoodCount > elowerFoodCount else lowerMid))

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getMiddlePositions(self):
        if self.red:
            mid_line = [((self.width / 2) - 1, y) for y in range(0, self.height)]
        else:
            mid_line = [((self.width / 2), y) for y in range(0, self.height)]
        valid_positions = [p for p in mid_line if p not in self.walls]
        return valid_positions

    def middlePosSplit(self):
        midLine = self.getMiddlePositions()
        if midLine is None or len(midLine) == 0:
            # print("Something Strange Happened, issue with initializing??")
            return None

        sortLine = sorted(midLine, key=lambda tup: tup[1])
        middleValueIndex = len(sortLine) // 2

        return sortLine[:middleValueIndex], sortLine[middleValueIndex:]

    def foodCluster(self, gameState):
        enemyFood = self.getFood(gameState).asList()
        foodClusterValue = {}
        # Creates map of food (Double for loop, probably better way to do this)
        for food in enemyFood:
            foodClusterValue[food] = 0
        # Gets surrounding food
        for food in enemyFood:
            surroundingPositions = self.getSurroundingNodes(food, False)
            # Increments food value of surrounding food
            for position in surroundingPositions:
                if position in enemyFood:
                    foodClusterValue[position] = foodClusterValue[position] + 1

        return foodClusterValue

    def getOpponentsPositions(self, gameState):
        enemyPositions = []
        opponentIndex = self.getOpponentsPositions(gameState)
        for agent in opponentIndex:
            if gameState.getAgentPosition(agent) is not None:
                enemyPositions.append(gameState.getAgentPosition(agent))

        return enemyPositions

    def getSurroundingNodes(self, position, directional):

        # Get north, east, south, west directions
        offsets = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # If desired position is not directional,
        if not directional:
            # Probably better way to get offsets
            offsets = [[-1, 1], [-1, -1], [1,1], [1,-1], [0, 1], [0, -1], [1, 0], [-1, 0]]

        neighboringPositions = []
        for x, y in offsets:
            neighboringPositions.append((position[0] + x, position[1] + y))
        # self.debugDraw(neighboringPositions, [0,1,0], True)
        return neighboringPositions

    def deadEndPosition(self, closestFoodPos, validationPosition):
        if validationPosition in self.deadEnds:
            # self.debugDraw(validationPosition, [1,0,0], False)
            return 3
        if validationPosition in self.foodList:
            return 0
        return 1

    def initDeadEnds(self):
        deadEnds = []
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # Get all enemy side positions and check for dead ends
        for position in self.enemySpace.asList():
            x, y = position

            # If position is not a wall
            if position not in self.walls:
                neighboringPositions = self.getSurroundingNodes(position, True)

                # Checks if those neighboring positions are walls or a path
                wallCount = 0
                for neighbor in neighboringPositions:
                    if neighbor in self.walls:
                        wallCount = wallCount + 1

                # Add dead end to list
                if wallCount == 3:
                    deadEnds.append(position)

        return deadEnds

    def deadEndOrigin(self, position, originsList):
        potentialOrigins = self.getSurroundingNodes(position, True)
        originsList.append(position)
        pathCount = 0
        newOrigin = None
        for neighbor in potentialOrigins:
            if neighbor not in self.walls and neighbor not in originsList and neighbor not in self.deadEnds:
                pathCount += 1
                newOrigin = neighbor
        if pathCount == 1:
            originsList = self.deadEndOrigin(newOrigin, originsList)
        return originsList

    def distFoodHeuristic(self, pos1, pos2):
        # use a simple manhatten distance as heuristic
        clusterMult = 1
        if len(self.enemyFoodClusterValue) > 0:
            for key in self.enemyFoodClusterValue:
                if pos2 == key:
                    clusterMult = 1/(self.enemyFoodClusterValue[pos2] + 1)

        return util.manhattanDistance(pos1, pos2) * clusterMult

    def distHeuristic(self, pos1, pos2):
        return util.manhattanDistance(pos1, pos2)

    def teamHeuristic(self, pos1, pos2):
        return 0 if self.teamSpace.__contains__(pos2) else 10000

    def zeroHeuristic(self, pos1, pos2):
        return 0

    def aStar(self, gameState, goal, heuristic):
        start = getAgentState(self.getCurrentObservation(), self.index).getPosition()

        queue = util.PriorityQueue()
        queue.push((start, []), 0)
        node = []

        while not queue.isEmpty():
            coords, path = queue.pop()
            if coords == goal:
                if len(path) == 0:
                    return Directions.STOP
                return path[0]
            if coords not in node:
                node.append(coords)
                for nodes in self.getSuccessors(coords, gameState):
                    cost = len(path + [nodes[1]]) + heuristic(start, nodes[0])
                    if nodes not in node:
                        queue.update((nodes[0], path + [nodes[1]]), cost)

        return Directions.STOP

    def getSuccessors(self, coordinate, gameState):
        successors = []
        for action in [Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.EAST]:
            x, y = coordinate
            dx, dy = game.Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.walls:
                nextPosition = (nextx, nexty)
                meaningfulEnemyPositions = self.meaningfulEnemyPositions(gameState)
                enemyNextMove = None
                if meaningfulEnemyPositions is not None:
                    for enemyPosition in meaningfulEnemyPositions:
                        enemyNextMove = self.getSurroundingNodes(enemyPosition, True)
                    if nextPosition not in meaningfulEnemyPositions and nextPosition not in enemyNextMove:
                        successors.append((nextPosition, action))
                else:
                    # If we are not concerned about hitting an enemy
                    successors.append((nextPosition, action))
        return successors

    def meaningfulEnemyPositions(self, gameState):
        enemyStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        meaningfulStates = [s for s in enemyStates if
                            s.getPosition() is not None and not s.isPacman and s.scaredTimer == 0]

        if len(meaningfulStates) > 0:
            return [position.getPosition() for position in meaningfulStates]
        else:
            return None

    def meaningfulPacmanPositions(self, gameState):
        meaningfulStates = self.meaningfulPacmanStates(gameState)
        if meaningfulStates is not None:
            return [position.getPosition() for position in meaningfulStates]
        else:
            return None

    def meaningfulPacmanStates(self, gameState):
        enemyStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        meaningfulStates = [s for s in enemyStates if
                            s.getPosition() is not None and s.isPacman]
        if len(meaningfulStates) > 0:
            return meaningfulStates
        else:
            return None

    def isPositionInTeamArea(self, gameState, pos):
        positions = self.getDefendingPositions(gameState).asList()
        return True if pos in positions else False

    def getDefendingPositions(self, gameState):
        teamSpace = capture.halfGrid(capture.Grid(self.width, self.height, True),
                                     red=True if self.index in gameState.getRedTeamIndices() else False)
        for position in teamSpace.asList():
            if gameState.data.layout.isWall(position):
                teamSpace[position[0]][position[1]] = False
        return teamSpace

    def getDefensiveAction(self, gameState):
        agentPos = gameState.getAgentPosition(self.index)
        enemy_positions = self.meaningfulPacmanPositions(gameState)

        if enemy_positions is not None:
            # make a list of distances from observed enemies
            dist = [self.getMazeDistance(agentPos, enemy) for enemy in enemy_positions]
            # get the distance of the closest enemy
            minDist = min(dist)
            # get the index of that distance
            index = dist.index(minDist)
            # use that to find the correct position!
            enemyPosition = enemy_positions[index]

            # Try and eat that pacman
            self.tempGoal = None
            return self.aStar(gameState, enemyPosition, self.distHeuristic)

        if self.detectFoodMissing():
            # Can't see any enemies so lets try to investigate the food location that went missing
            missingFoodList = self.getMissingFood(gameState)
            # print("Missing Food", missingFoodList)
            if missingFoodList is not None:
                missingPosition = self.getClosestFoodToPosition(missingFoodList[0], gameState)
                self.tempGoal = missingPosition
                if agentPos == missingPosition:
                    return self.aStar(gameState, self.getClosestFoodToPosition(agentPos, gameState), self.distHeuristic)
                return self.aStar(gameState, missingPosition, self.distHeuristic)

        # Detect?
        if self.tempGoal is not None:
            if agentPos == self.tempGoal:
                self.tempGoal = None
            return self.aStar(gameState, self.tempGoal, self.distHeuristic)




        # Go to random positions that are near the middle line, using team heuristic that means it good to be on our
        # side of the map
        return self.aStar(gameState, random.choice(self.getMiddlePositions()), self.teamHeuristic)

    def detectFoodMissing(self):
        if self.getPreviousObservation() is not None:
            currentFoodCount = len(self.getFoodYouAreDefending(self.getCurrentObservation()).asList())
            previousFoodCount = len(self.getFoodYouAreDefending(self.getPreviousObservation()).asList())
            check = currentFoodCount != previousFoodCount
            return check
        return False

    def detectFoodEaten(self):
        if self.getPreviousObservation() is not None:
            check = len(self.getFood(self.getCurrentObservation()).asList()) != len(
                self.getFood(self.getPreviousObservation()).asList())
            return check
        return False

    def getClosestFoodToPosition(self, position, gameState):
        dists = [self.getMazeDistance(position, foodPos) for foodPos in self.getFoodYouAreDefending(gameState).asList()]
        index = dists.index(min(dists))

        return self.getFoodYouAreDefending(gameState).asList()[index]


    def getMissingFood(self, gameState):
        currentFoodList = self.getFoodYouAreDefending(gameState).asList()
        if self.getPreviousObservation() is not None:
            previousFoodList = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
            if len(currentFoodList) < len(previousFoodList):
                return list(set(previousFoodList).difference(set(currentFoodList)))
        return None

    def getOffensiveAction(self, gameState):
        enemyPositions = self.meaningfulPacmanPositions(gameState)
        enemyStates = self.meaningfulPacmanStates(gameState)
        enemyIndices = self.getOpponents(gameState)
        agentPos = gameState.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()
        closestTeamPos = self.findClosetTeamPosition(gameState)

        # go back if no food left, originally thought we'd want wins hence why we had it go back if it had collected all but two.
        # however we want Points!
        if len(foodList) <= 0:
            return self.aStar(gameState, closestTeamPos, self.distHeuristic)

        if self.resetGoal and self.isPositionInTeamArea(gameState, agentPos):
            x, y = agentPos
            lowerMid, higherMid = self.middlePosSplit()
            if self.tempGoal is None:
                if y >= self.height / 2:
                    self.tempGoal = random.choice(lowerMid)
                else:
                    self.tempGoal = random.choice(higherMid)
            elif agentPos == self.tempGoal:
                self.tempGoal = None
                self.resetGoal = False

            return self.aStar(gameState, self.tempGoal, self.teamHeuristic)

        # If game is about to end, score points
        if gameState.data.timeleft < self.evacuationTime and getAgentState(gameState, self.index).numCarrying > 0:
            self.goal = closestTeamPos
            return self.aStar(gameState, closestTeamPos, self.distHeuristic)

        for index in enemyIndices:
            if index is not None:
                enemyValue = gameState.getAgentState(index).numCarrying
                enemyPosition = gameState.getAgentState(index).getPosition()

                # If we can see the enemy
                if enemyPosition is not None:
                    closestEnemyMidlinePos = self.getNearestMidlinePoint(enemyPosition)
                    distFromEnemy = self.getMazeDistance(agentPos, enemyPosition)
                    distToMidline = self.getMazeDistance(closestEnemyMidlinePos, enemyPosition)

                    # If the enemy has >1/3rd of our food and we can beat him to the middle
                    if enemyValue >= len(self.getFoodYouAreDefending(
                            gameState).asList()) // 3 and distFromEnemy <= 2 * distToMidline:
                        action = self.aStar(gameState, enemyPosition, self.distHeuristic)
                        return action

                    # If enemy is a ghost
                    if not gameState.getAgentState(index).isPacman:
                        backPos = self.getNearestMidlinePoint(agentPos)
                        scaredTimer = gameState.getAgentState(index).scaredTimer
                        agentFoodDistanceList = [self.getMazeDistance(agentPos, foodPos) for foodPos in self.getFood(gameState).asList()]
                        closestFoodDistance = min(agentFoodDistanceList)
                        closestFoodPos = foodList[agentFoodDistanceList.index(closestFoodDistance)]
                        enemyClosestFoodDistance = self.getMazeDistance(enemyPosition, closestFoodPos)
                        enemyDistanceToAgent = self.getMazeDistance(enemyPosition, gameState.getAgentPosition(self.index))

                        action = self.aStar(gameState, backPos, self.deadEndPosition)

                        # If we can reach the scared ghost before time runs out
                        if scaredTimer > distFromEnemy:
                            if distFromEnemy + 2 < closestFoodDistance:
                                return self.aStar(gameState, enemyPosition, self.deadEndPosition) # Eat Enemy
                            return self.aStar(gameState, closestFoodPos, self.deadEndPosition) # Eat Food

                        # Quick check if power pellet is closer to us than the ghost
                        pellets = self.getCapsules(gameState)

                        if pellets is not None and len(pellets) > 0:
                            apelDists = [self.getMazeDistance(agentPos, pelletPos) for pelletPos in pellets]
                            if apelDists is not None and len(apelDists) > 0:
                                pelDist = min(apelDists)
                                epelDists = [self.getMazeDistance(enemyPosition, pelletPos) for pelletPos in pellets]
                                if epelDists is not None and len(epelDists) > 0:
                                    eDistPel = min(epelDists)
                                    if pelDist < eDistPel:
                                       # print("Racing for power pellet")
                                        return self.aStar(gameState, pellets[apelDists.index(pelDist)], self.distHeuristic)

                        # Get food if it thinks its safe
                        distMultiplyer = 2
                        # If the food is in a dead end, multiply the length to the exit by 2
                        if closestFoodPos in self.deadEnds:
                            closestFoodDistance = len(self.deadEndOrigin(closestFoodPos, []))
                        if closestFoodDistance + 2.5 < enemyClosestFoodDistance and enemyDistanceToAgent > closestFoodDistance + 2.5:
                            return self.aStar(gameState, closestFoodPos, self.distHeuristic)
                        if self.isPositionInTeamArea(gameState, agentPos):
                            self.resetGoal = True
                        return action

        foodPosition = None
        for food in foodList:
            if foodPosition is None:
                foodPosition = food
            if self.getMazeDistance(agentPos, foodPosition) > self.getMazeDistance(agentPos, food):
                foodPosition = food

        return Directions.STOP if foodPosition is None else self.aStar(gameState, foodPosition, self.distFoodHeuristic)

    def findClosetTeamPosition(self, gameState):
        position = gameState.data.agentStates[self.index].start
        positions = self.getDefendingPositions(gameState).asList()
        currentPos = gameState.getAgentPosition(self.index)
        dist = math.inf
        for pos in positions:
            testDist = util.manhattanDistance(pos, currentPos)
            if dist > testDist:
                dist = testDist
                position = pos

        return position

    def getNearestMidlinePoint(self, position):
        midline = self.getMiddlePositions()
        distanceDictionary = {middlePos: self.getMazeDistance(position, middlePos) for middlePos in midline}

        nearest = min(distanceDictionary.values())

        # a method to choose a random point if multiple points have the same distance and are closest to us
        tester = [pos for (pos, dist) in distanceDictionary.items() if dist == nearest]
        if len(tester) > 0:
            return random.choice(tester)
        # print("Something bad happened in getNearestMidlinePoint")
        return None  # Fail Condition

    def checkIfAgentDied(self, index):
        previousState = self.getPreviousObservation()
        currentState = self.getCurrentObservation()

