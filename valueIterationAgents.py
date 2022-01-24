
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
        for i in range(self.iterations):
            values = self.values.copy()
            for s in self.mdp.getStates():
                self.values[s] = -float('inf')

                for a in self.mdp.getPossibleActions(s):
                    tmp = 0
                    for s_prime, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                        tmp += prob * (self.mdp.getReward(s, a, s_prime) + self.discount * values[s_prime])
                    self.values[s] = max(self.values[s], tmp)
                if self.values[s] == -float('inf'):
                    self.values[s] = 0.0

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
        tmp = 0
        for s_prime, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            tmp += prob * (self.mdp.getReward(state, action, s_prime) + self.discount * self.values[s_prime])
        return tmp

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        maxnum, action = -float('inf'), None
        for actions in self.mdp.getPossibleActions(state):
            tmp = self.computeQValueFromValues(state, actions)
            if tmp > maxnum:
                maxnum, action = tmp, actions
        return action

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
        for i in range(self.iterations):

            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            best = self.computeActionFromValues(state)
            if best is None:
                v = 0
            else:
                v = self.computeQValueFromValues(state, best)
            self.values[state] = v

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
        all_states = self.mdp.getStates()

        predecessors = {}
        for state in all_states:
            predecessors[state] = set()
        for state in all_states:
            for action in self.mdp.getPossibleActions(state):
                for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)


        heap = util.PriorityQueue()

        for state in all_states:
            if not self.mdp.isTerminal(state):
                diff = abs(self.getValue(state) - self.computeQvalues(state))
                heap.push(state, -diff)


        for i in range(self.iterations):
            if heap.isEmpty():
                break
            else:
                state = heap.pop()
                if not self.mdp.isTerminal(state):
                    self.values[state] = self.computeQvalues(state)
                for pred in predecessors[state]:
                    if self.mdp.isTerminal(pred):
                        diff = abs(self.getValue(pred))
                    else:
                        diff = abs(self.getValue(pred) - self.computeQvalues(pred))

                    if diff > self.theta:
                        heap.update(pred, -diff)

    def computeQvalues(self, state):
        all_actions = self.mdp.getPossibleActions(state)
        qval= []
        for action in all_actions:
            qval.append(self.getQValue(state, action))

        return max(qval)
