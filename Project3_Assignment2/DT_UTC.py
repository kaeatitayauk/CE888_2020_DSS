# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in
# the UCTPlayGame() function at the bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is player 2 - player 1 has the first move

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 = empty, 1 = player 1, 2 = player 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5  # draw
        return False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
            self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose=False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []:  # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(
                node.playerJustMoved))  # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if verbose:
        print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move  # return the move that was most visited


import pandas as pd
import numpy as np


def UCTPlayGame(model, scaler):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """
    temp = []  # Store formation of board after player 1 plays
    dump = []  # Store formation of board at each game
    c = 0
    wintime = 0
    wintime2 = 0
    state = OXOState()  # uncomment to play OXO
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initialize board 0:empty 1:player1 2:player2
    predict_input = []  # store prediction input, will take from board formation
    while state.GetMoves() != []:
        print(str(state))  # display the current board
        print("Player", state.playerJustMoved, "turn")  # show current player turn
        if state.playerJustMoved == 1:  # player one move fuction
            predict_input = predict_input[0:9]  # take the formation of current board as input data for classifier
            predict_input = np.asarray(predict_input)  # to numpy array
            predict_input = predict_input.reshape(1, -1)  # convert to single sample
            predict_input = scaler.transform(predict_input)  # transform input
            print("Decision Tree Prediction: ", model.predict(predict_input))  # display prediction result
            m = model.predict(predict_input)  # save prediction to parameter m
            pos_move = state.GetMoves()  # store empty slot on the field
            m = int(m)
            if m not in pos_move:  # if classifier predicts on unavailable slot, the Monte Carlo Tree Search will be
                # used
                m = UCT(rootstate=state, itermax=1000, verbose=False)  # play with values for itermax and verbose = True
        else:
            m = UCT(rootstate=state, itermax=100, verbose=False)
        print("Best Move: " + str(m) + "\n")  # display best move of each turn

        state.DoMove(m)
        if state.playerJustMoved == 1:
            data[9] = m  # save the best move of player 1
            for i in data:
                temp.append(i)  # store to temp parameter
            if c > 0:
                dump.append(temp)  # if its not the first turn save the formation to dump
            predict_input = data  # send formation data to predict_input as input for prediction
            data[m] = 1  # player 1 moves will be store in data with at the best move
            temp = []
            c += 1
        else:
            data[m] = 2  # player 2 moves will be store in data with at the best move
        if state.GetResult(state.playerJustMoved) != False:
            print(str(state))
            break
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
        if str(state.playerJustMoved) == '1':
            trainable.append(dump)  # only save data of each move when player 1 wins
            wintime += 1  # count win time of player 1
        if str(state.playerJustMoved) == '2':
            wintime2 += 1  # count win time of player 2
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
        if str(3 - state.playerJustMoved) == '1':
            trainable.append(dump)  # only save data of each move when player 1 wins
            wintime += 1  # count win time of player 1
        if str(state.playerJustMoved) == '2':
            wintime2 += 1  # count win time of player 2
    else:
        print("Nobody wins!")
    win = wintime
    win2 = wintime2
    return trainable, win, win2  # return train data and win count game by game


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import csv

trainable = []
if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    number_of_game = 100
    train = pd.read_csv('traindata.csv')  # read train data
    train = train.drop_duplicates()  # drop duplicate sample
    X = train.iloc[0:, 0:9].values  # train features
    Y = train.iloc[0:, 9].values  # train label
    scaler = StandardScaler().fit(X)  # fit scaler on train feature
    X = scaler.transform(X)  # transform train feature
    clf = DecisionTreeClassifier()  # define DT classifier
    model = clf.fit(X, Y)  # train classifier after a set of match

    win = 0  # win of count player 1 total parameter
    win2 = 0  # win of count player 2 total parameter
    # read train data into list
    with open('traindata.csv', newline='') as f:
        reader = csv.reader(f)
        old_data = list(reader)
    # mearge old data with new match data
    train_data = old_data
    for i in range(number_of_game):  # loop the number of match
        x, w, w2 = UCTPlayGame(model, scaler)  # call main UTC XO and send trained model with transformer
        if x:  # if at current game player 1 win will store the training data of moves
            for k in x:
                for j in k:  # each move breaks into one sample
                    train_data.append(j)  # store each sample
        if w:
            win += 1  # total win count of player 1
        if w2:
            win2 += 1  # total win count of player 2
    print("\nNumber of match:", number_of_game)
    print("Player 1 (Decision Tree) Win Amount:", win)  # display total win count of player 1
    print("Player 2 Win Amount:", win2)  # display total win count of player 2

    myData = train_data
    myFile = open('traindata.csv', 'w', newline='')  # save merged training data
    with myFile:
        writer = csv.writer(myFile)  # save merged training data
        writer.writerows(myData)
