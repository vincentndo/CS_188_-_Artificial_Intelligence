# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        statesList = self.mdp.getStates()
        copyValues = util.Counter()

        for _ in range(self.iterations):

          for state in statesList:
            if not self.mdp.isTerminal(state):
              actions = self.mdp.getPossibleActions(state)
              qValues = [ self.computeQValueFromValues(state, action) for action in actions ]
              copyValues[state] = max(qValues)

          for state in statesList:
            self.values[state] = copyValues[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
          return 0
        else:
          statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
          ret = 0
          for transState, prob in statesAndProbs:
            if not self.mdp.isTerminal(state):
              reward = self.mdp.getReward(state, action, transState)
              ret += prob * (reward + self.discount * self.values[transState])
          return ret

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
          return ''
        else:
          lst = [ (action, self.computeQValueFromValues(state, action)) for action in self.mdp.getPossibleActions(state) ]

          return max(lst, key = lambda k: k[1])[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        statesList = self.mdp.getStates()
        numStates = len(statesList)
        # copyValues = util.Counter()

        for i in range(self.iterations):
          index = i % numStates
          state = statesList[index]
          if not self.mdp.isTerminal(state):
            actions = self.mdp.getPossibleActions(state)
            qValues = [ self.computeQValueFromValues(state, action) for action in actions ]
            self.values[state] = max(qValues)

          # for state in statesList:
          #   self.values[state] = copyValues[state]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        statesList = self.mdp.getStates()
        precedessors = util.Counter()

        for state in statesList:
          actions = self.mdp.getPossibleActions(state)
          for action in actions:
            statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            for childState, prob in statesAndProbs:
              if prob != 0.0:
                if precedessors[childState] == 0:
                  precedessors[childState] = set()
                else:
                  precedessors[childState].add(state)

        priorityQueue = util.PriorityQueue()

        for state in statesList:
          if not self.mdp.isTerminal(state):
            max_qValues = max( [ self.computeQValueFromValues(state, action) 
                                  for action in self.mdp.getPossibleActions(state) ] )

            diff = abs( self.values[state] - max_qValues )
            priorityQueue.push(state, -diff)

        for i in range(self.iterations):
          if priorityQueue.isEmpty():
            return
          state = priorityQueue.pop()
          self.values[state] = max( [ self.computeQValueFromValues(state, action) 
                                      for action in self.mdp.getPossibleActions(state) ] )
          for p in precedessors[state]:
            max_qValues = max( [ self.computeQValueFromValues(p, action) 
                                  for action in self.mdp.getPossibleActions(p) ] )

            diff = abs( self.values[p] - max_qValues )
            if diff > self.theta:
              priorityQueue.update(p, -diff)


