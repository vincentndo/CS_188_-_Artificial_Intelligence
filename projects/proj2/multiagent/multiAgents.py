# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score = successorGameState.getScore()

        curPos = currentGameState.getPacmanPosition()
        curFood = currentGameState.getFood()
        curFoodList = curFood.asList()

        newFoodList = newFood.asList()
        closestFood = 0
        if len(curFoodList) == len(newFoodList):
          closestFood = min([ manhattanDistance(newPos, food) for food in newFoodList ])
        else:
          score += 5

        newGhostPos = successorGameState.getGhostPosition(1)
        pacmanToGhost = manhattanDistance(newGhostPos, newPos)

        if newScaredTimes[0] > 0:
          return score-closestFood**2+pacmanToGhost

        if pacmanToGhost < 3:
          return score-closestFood**2+pacmanToGhost*4
        elif pacmanToGhost < 6:
          return score-closestFood**2+pacmanToGhost**2
        else:
          return score-closestFood**2+pacmanToGhost

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        numAgents = gameState.getNumAgents()

        def max_action(gameState, depth):
          legalMoves = gameState.getLegalActions(0)
          moveList = []
          for move in legalMoves:
            successorGameState = gameState.generateSuccessor(0, move)
            if successorGameState.isWin() or successorGameState.isLose():
              score = scoreEvaluationFunction(successorGameState)
              moveList.append([move, score])

          remainingMoves = [ [move, action( gameState.generateSuccessor(0, move), 1, depth )[1]]
                              for move in legalMoves if not gameState.generateSuccessor(0, move).isWin() and not gameState.generateSuccessor(0, move).isLose() ]

          if len(remainingMoves) is not 0:
            moveList = moveList + remainingMoves

          return max( [ moveAndScore for moveAndScore in moveList ], key = lambda k: k[1] )
          
        def min_action(gameState, agentIndex, depth):          
          if agentIndex == numAgents - 1:
            nextAgentIndex = 0
            depth -= 1
            if depth == 0:
              return action(gameState, agentIndex, depth)
          else:
            nextAgentIndex = agentIndex + 1

          legalMoves = gameState.getLegalActions(agentIndex)
          moveList = []
          for move in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, move)
            if successorGameState.isWin() or successorGameState.isLose():
              score = scoreEvaluationFunction(successorGameState)
              moveList.append([move, score])

          remainingMoves = [ [move, action( gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth )[1]]
                              for move in legalMoves if not gameState.generateSuccessor(agentIndex, move).isWin() and not gameState.generateSuccessor(agentIndex, move).isLose() ]

          if len(remainingMoves) is not 0:
            moveList = moveList + remainingMoves

          return min( [ moveAndScore for moveAndScore in moveList ], key = lambda k: k[1] )

        def action(gameState, agentIndex, depth):
          if gameState.isWin() or gameState.isLose():
            score = scoreEvaluationFunction(gameState)
            return [Directions.STOP, score]
          if depth == 0 and agentIndex == numAgents - 1:
            legalMoves = gameState.getLegalActions(agentIndex)
            minMove = min( [ move for move in legalMoves],
                             key = lambda k: scoreEvaluationFunction(gameState.generateSuccessor(agentIndex, k)) )
            score = scoreEvaluationFunction( gameState.generateSuccessor(agentIndex, minMove) )
            return [minMove, score]
          else:
            if agentIndex == 0:
              return max_action(gameState, depth)
            else:
              return min_action(gameState, agentIndex, depth)

        return action(gameState, 0, depth)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        numAgents = gameState.getNumAgents()

        def max_action(gameState, depth, alpha, beta):
          score = -999999
          legalMoves = gameState.getLegalActions(0)
          moveList = []
          # for move in legalMoves:
          #   successorGameState = gameState.generateSuccessor(0, move)
          #   if successorGameState.isWin() or successorGameState.isLose():
          #     score = scoreEvaluationFunction(successorGameState)
          #     moveList.append([move, score])

          retMove = Directions.STOP
          for move in legalMoves:
            successorGameState = gameState.generateSuccessor(0, move)
            if successorGameState.isWin() or successorGameState.isLose():
              newScore = scoreEvaluationFunction(successorGameState)
              moveList.append([move, score])
            else:
              newScore = action( successorGameState, 1, depth, alpha, beta )[1]
            if score < newScore:
              score = newScore
              retMove = move
            if score > beta:
              return [ move, score ]
            alpha = max( alpha, score )

          return [ retMove, score ]

          # remainingMoves = [ [move, action( gameState.generateSuccessor(0, move), 1, depth )[1]]
          #                     for move in legalMoves if not gameState.generateSuccessor(0, move).isWin() and not gameState.generateSuccessor(0, move).isLose() ]

          # if len(remainingMoves) is not 0:
          #   moveList = moveList + remainingMoves
          # ret = max( [ moveAndScore for moveAndScore in moveList ], key = lambda k: k[1] )
          # # print(ret[1])
          # return ret

          # return max( [ move for move in legalMoves  ],
          #               key = lambda k: action( gameState.generateSuccessor(0, k), 1, depth )[1] )
                         # scoreEvaluationFunction(gameState.generateSuccessor( 1, action(gameState.generateSuccessor(0, k), 1, depth) )) )
# if not gameState.generateSuccessor(0, move).isWin() and not gameState.generateSuccessor(0, move).isLose()
        def min_action(gameState, agentIndex, depth, alpha, beta):          
          if agentIndex == numAgents - 1:
            nextAgentIndex = 0
            depth -= 1
            if depth == 0:
              return action(gameState, agentIndex, depth, alpha, beta)
          else:
            nextAgentIndex = agentIndex + 1

          score = 999999
          legalMoves = gameState.getLegalActions(agentIndex)
          moveList = []
          retMove = Directions.STOP
          for move in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, move)
            if successorGameState.isWin() or successorGameState.isLose():
              newScore = scoreEvaluationFunction(successorGameState)
              moveList.append([move, score])
            else:
              newScore = action( successorGameState, nextAgentIndex, depth, alpha, beta )[1]
            if score > newScore:
              score = newScore
              retMove = move
            if score < alpha:
              return [ move, score ]
            beta = min( beta, score )

          return [ retMove, score ]

          # remainingMoves = [ [move, action( gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth )[1]]
          #                     for move in legalMoves if not gameState.generateSuccessor(agentIndex, move).isWin() and not gameState.generateSuccessor(agentIndex, move).isLose() ]

          # if len(remainingMoves) is not 0:
          #   moveList = moveList + remainingMoves
          # return min( [ moveAndScore for moveAndScore in moveList ], key = lambda k: k[1] )

          # # return min( [ move for move in legalMoves ],
          # #               key = lambda k: action( gameState.generateSuccessor(agentIndex, k), nextAgentIndex, depth )[1] )
          #                # scoreEvaluationFunction(gameState.generateSuccessor( nextAgentIndex, action(gameState.generateSuccessor(agentIndex, k), nextAgentIndex, depth) )) )

        def action(gameState, agentIndex, depth, alpha, beta):
          if gameState.isWin() or gameState.isLose():
            score = scoreEvaluationFunction(gameState)
            return [Directions.STOP, score]
          if depth == 0 and agentIndex == numAgents - 1:
            score = 999999
            legalMoves = gameState.getLegalActions(agentIndex)
            retMove = Directions.STOP
            # print(alpha)
            for move in legalMoves:
              successorGameState = gameState.generateSuccessor(agentIndex, move)
              newScore = scoreEvaluationFunction( successorGameState )
              if score > newScore:
                score = newScore
                retMove = move
              if score < alpha:
                return [ move, score ]
              beta = min( beta, score )

            return [ retMove, score ]

            # legalMoves = gameState.getLegalActions(agentIndex)
            # minMove = min( [ move for move in legalMoves],
            #                  key = lambda k: scoreEvaluationFunction(gameState.generateSuccessor(agentIndex, k)) )
            # score = scoreEvaluationFunction( gameState.generateSuccessor(agentIndex, minMove) )
            # return [minMove, score]
       
          else:
            if agentIndex == 0:
              return max_action(gameState, depth, alpha, beta)
            else:
              return min_action(gameState, agentIndex, depth, alpha, beta)

        alpha = -999999
        beta = 999999

        return action(gameState, 0, depth, alpha, beta)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        numAgents = gameState.getNumAgents()

        def max_action(gameState, depth):
          legalMoves = gameState.getLegalActions(0)
          moveList = []
          for move in legalMoves:
            successorGameState = gameState.generateSuccessor(0, move)
            if successorGameState.isWin() or successorGameState.isLose():
              score = self.evaluationFunction(successorGameState)
              moveList.append([move, score])
          # print(moveList)
          remainingMoves = [ [move, action( gameState.generateSuccessor(0, move), 1, depth )[1]]
                              for move in legalMoves if not gameState.generateSuccessor(0, move).isWin() and not gameState.generateSuccessor(0, move).isLose() ]

          if len(remainingMoves) is not 0:
            moveList = moveList + remainingMoves
          ret = max( [ moveAndScore for moveAndScore in moveList ], key = lambda k: k[1] )
          # print(ret[1])
          return ret

          # return max( [ move for move in legalMoves  ],
          #               key = lambda k: action( gameState.generateSuccessor(0, k), 1, depth )[1] )
                         # scoreEvaluationFunction(gameState.generateSuccessor( 1, action(gameState.generateSuccessor(0, k), 1, depth) )) )
# if not gameState.generateSuccessor(0, move).isWin() and not gameState.generateSuccessor(0, move).isLose()
        def expect_action(gameState, agentIndex, depth):

          import random

          if agentIndex == numAgents - 1:
            nextAgentIndex = 0
            depth -= 1
            if depth == 0:
              return action(gameState, agentIndex, depth)
          else:
            nextAgentIndex = agentIndex + 1

          legalMoves = gameState.getLegalActions(agentIndex)
          moveList = []
          for move in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, move)
            if successorGameState.isWin() or successorGameState.isLose():
              score = self.evaluationFunction(successorGameState)
              moveList.append([move, score])
          
          remainingMoves = [ [move, action( gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth )[1]]
                              for move in legalMoves if not gameState.generateSuccessor(agentIndex, move).isWin() and not gameState.generateSuccessor(agentIndex, move).isLose() ]

          if len(remainingMoves) is not 0:
            moveList = moveList + remainingMoves
          
          randomIndex = random.randrange(0, len(moveList))
          randomMove = moveList[randomIndex][0]
          expectedScore = sum([ score[1] for score in moveList ]) * 1.0 / (len(moveList) * 1.0)

          return [randomMove, expectedScore]

          # return min( [ move for move in legalMoves ],
          #               key = lambda k: action( gameState.generateSuccessor(agentIndex, k), nextAgentIndex, depth )[1] )
                         # scoreEvaluationFunction(gameState.generateSuccessor( nextAgentIndex, action(gameState.generateSuccessor(agentIndex, k), nextAgentIndex, depth) )) )

        def action(gameState, agentIndex, depth):
          if gameState.isWin() or gameState.isLose():
            score = self.evaluationFunction(gameState)
            return [Directions.STOP, score]
          if depth == 0 and agentIndex == numAgents - 1:
            legalMoves = gameState.getLegalActions(agentIndex)
            minMove = min( [ move for move in legalMoves],
                             key = lambda k: self.evaluationFunction(gameState.generateSuccessor(agentIndex, k)) )
            score = self.evaluationFunction( gameState.generateSuccessor(agentIndex, minMove) )
            return [minMove, score]
          else:
            if agentIndex == 0:
              return max_action(gameState, depth)
            else:
              return expect_action(gameState, agentIndex, depth)

        return action(gameState, 0, depth)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    numAgents = currentGameState.getNumAgents()

    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    capsuleList = currentGameState.getCapsules()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]

    curFoodList = curFood.asList()
    # capsuleList = curCapsules.asList()

    closestCapsule = 0
    if len(capsuleList) != 0:
      closestCapsule = min([ manhattanDistance(curPos, capsule) for capsule in capsuleList ])

    closestFood = 0
    if len(curFoodList) != 0:
      closestFood = min([ manhattanDistance(curPos, food) for food in curFoodList ])
    # else:
      # score += 50
    # t = m
    curGhostPos = [ currentGameState.getGhostPosition(i) for i in range(1, numAgents) ]
    pacmanToGhost = min( [ manhattanDistance(ghost, curPos) for ghost in curGhostPos ] )

    a, b, c, d = 0, 0, 0, 0
    if closestFood != 0:
      a = 1.0/closestFood * 8
      b = 1.0/len(curFoodList) * 8

    if closestCapsule != 0:
      # c = 1.0/closestCapsule * 10
      d = 1.0/len(capsuleList) * 200

    if curScaredTimes[0] > 0:

        return score + a + b + c + d - pacmanToGhost*3

    if pacmanToGhost < 3:
      return score + a + b + c + d + pacmanToGhost*4
    elif pacmanToGhost < 6:
      return score + a + b + c + d + pacmanToGhost**2
    else:
      return score + a + b + c + d + pacmanToGhost

# Abbreviation
better = betterEvaluationFunction
