import numpy as np
import time
from tkinter import *
from helper import *
from tkinter import PhotoImage


class PDWorld(Frame):

  def __init__(self, master=None):
    self.pickup_list = []
    self.dropoff_list = []
    self.master = master
    Frame.__init__(self, self.master)
    self.num = 1
    self.agent = ()
    self.master.bind("<space>", lambda e: self.prompt_experiments())

  # Writes output to a TXT file
  def output_(self):
    file1 = open("MyOutput.txt","w")
    print(bank_account)
    print(num_operator)

    file1.write("\n|---------------- Bank account and num operators ----------------|\n")
    file1.write(' '.join(map(str, bank_account)))
    file1.write('\n')
    file1.write(' '.join(map(str, num_operator)))
    file1.write("\n|----------------Running Experiment 1 Run 1 ----------------|\n")
    file1.write("\n|------------------------------ Pick up Q table Final -------------------------------|\n")
    file1.write(' '.join(map(str, pickup_q_table)))
    file1.write("\n|------------------------------ Drop off Q table Final -------------------------------|\n")
    file1.write(' '.join(map(str, dropoff_q_table)))
    file1.write(' '.join("\n|---------------- END ----------------|\n"))


  # --------------------------------- EXPERIMENTS -------------------------------------- #
  def experiment_1_a(self):
    learning_rate = 0.3
    discount_rate = 0.5

    pickup_states = [[1, 3, 0], [2, 1, 0], [2, 2, 0]]
    dropoff_states = [[1, 4, 1], [4, 0, 1], [4, 2, 0]]

    agent.reset()

    clear_lists()
    initialize_Q_table()
    print("\nQ_table initialized!\n")
    populateCells(pickup_states, dropoff_states)
    print("\nCells were succesfully populated!\n")

    print("\n|---------------- RANDOM POLICY ----------------|\n")

    for index in range(10000):
      agent.policy = "PRANDOM"
      Q_learning(learning_rate, discount_rate, agent, pickup_states, dropoff_states)

    print("FINISH")
    self.output_()

  def experiment_1_b(self):
    learning_rate = 0.3
    discount_rate = 0.5

    pickup_states = [[0, 0, 1], [1, 1, 1], [2, 2, 1]]
    dropoff_states = [[1, 4, 1], [4, 0, 1], [4, 2, 1]]

    agent.reset()

    clear_lists()
    initialize_Q_table()
    print("\nQ_table initialized!\n")
    populateCells(pickup_states, dropoff_states)
    print("\nCells were succesfully populated!\n")

    print("\n|---------------- RANDOM POLICY ----------------|\n")

    for index in range(500):
      agent.policy = "PRANDOM"
      Q_learning(learning_rate, discount_rate, agent, pickup_states, dropoff_states)

    print("\n|---------------- GREEDY POLICY ----------------|\n")

    for index in range(9500):
      agent.policy = "PGREEDY"
      Q_learning(learning_rate, discount_rate, agent, pickup_states, dropoff_states)

    print("FINISH")
    self.output_()

  def experiment_1_c(self):
    learning_rate = 0.3
    discount_rate = 0.5

    pickup_states = [[0, 0, 1], [1, 1, 1], [2, 2, 1]]
    dropoff_states = [[1, 4, 1], [4, 0, 1], [4, 2, 1]]

    agent.reset()

    clear_lists()
    initialize_Q_table()
    print("\nQ_table initialized!\n")
    populateCells(pickup_states, dropoff_states)
    print("\nCells were succesfully populated!\n")

    print("\n|---------------- RANDOM POLICY ----------------|\n")

    for index in range(500):
      agent.policy = "PRANDOM"
      Q_learning(learning_rate, discount_rate, agent, pickup_states, dropoff_states)

    print("\n|---------------- EXPLOIT POLICY ----------------|\n")

    for index in range(9500):
      agent.policy = "PEXPLOIT"
      Q_learning(learning_rate, discount_rate, agent, pickup_states, dropoff_states)

    print("FINISH")
    self.output_()


  def experiment_2(self):
    learning_rate = 0.3
    discount_rate = 0.5

    pickup_states = [[0, 0, 1], [1, 1, 1], [2, 2, 1]]
    dropoff_states = [[1, 4, 1], [4, 0, 1], [4, 2, 1]]

    agent.reset()

    clear_lists()
    initialize_Q_table()
    print("\nQ_table initialized!\n")
    populateCells(pickup_states, dropoff_states)
    print("\nCells were succesfully populated!\n")

    print("\n|---------------- RANDOM POLICY ----------------|\n")

    agent.policy = "PRANDOM"
    next_action = SARSA_Q_learning(learning_rate, discount_rate, None, agent, pickup_states, dropoff_states)
    for index in range(500):
      next_action = SARSA_Q_learning(learning_rate, discount_rate, next_action, agent, pickup_states, dropoff_states)

    print("\n|---------------- EXPLOIT POLICY ----------------|\n")

    agent.policy = "PEXPLOIT"
    for index in range(9500):
      next_action = SARSA_Q_learning(learning_rate, discount_rate, next_action, agent, pickup_states, dropoff_states)

    print("FINISH")
    self.output_()

  def experiment_3(self):
    # learning_rate = 0.5
    learning_rate = 0.1
    discount_rate = 0.5

    pickup_states = [[0, 0, 1], [1, 1, 1], [2, 2, 1]]
    dropoff_states = [[1, 4, 1], [4, 0, 1], [4, 2, 1]]

    agent.reset()

    clear_lists()
    initialize_Q_table()
    print("\nQ_table initialized!\n")
    populateCells(pickup_states, dropoff_states)
    print("\nCells were succesfully populated!\n")

    print("\n|---------------- RANDOM POLICY ----------------|\n")

    agent.policy = "PRANDOM"
    next_action = SARSA_Q_learning(learning_rate, discount_rate, None, agent, pickup_states, dropoff_states)
    for index in range(500):
      next_action = SARSA_Q_learning(learning_rate, discount_rate, next_action, agent, pickup_states, dropoff_states)

    print("\n|---------------- EXPLOIT POLICY ----------------|\n")

    agent.policy = "PEXPLOIT"
    for index in range(9500):
      next_action = SARSA_Q_learning(learning_rate, discount_rate, next_action, agent, pickup_states, dropoff_states)

    print("FINISH")
    self.output_()

  def experiment_4(self):
    learning_rate = 0.3
    discount_rate = 0.5

    pickup_states = [[0, 0, 1], [1, 1, 1], [2, 2, 1]]
    dropoff_states = [[1, 4, 1], [4, 0, 1], [4, 2, 1]]

    agent.reset()

    clear_lists()
    initialize_Q_table()
    print("\nQ_table initialized!\n")
    populateCells(pickup_states, dropoff_states)
    print("\nCells were succesfully populated!\n")

    print("\n|---------------- RANDOM POLICY ----------------|\n")

    agent.policy = "PRANDOM"
    next_action = SARSA_Q_learning(learning_rate, discount_rate, None, agent, pickup_states, dropoff_states)
    for index in range(500):
      next_action = SARSA_Q_learning(learning_rate, discount_rate, next_action, agent, pickup_states, dropoff_states)
      if agent.getFinalStates() == 2 and not swapped:
        print("Change pickup locations to (2,3,3) and (1,3,1)")

    print("\n|---------------- EXPLOIT POLICY ----------------|\n")

    agent.policy = "PEXPLOIT"
    for index in range(9500):
      next_action = SARSA_Q_learning(learning_rate, discount_rate, next_action, agent, pickup_states, dropoff_states)
      if agent.getFinalStates() == 2 and not swapped:
        print("Change pickup locations to (2,3,3) and (1,3,1)")

    print("FINISH")
    self.output_()

  def delete_nums(self):
    self.c.delete("nums")

  def prompt_experiments(self):
    experiment_num = int(input("Choose Experiment 1 - 4 by entering the corresponding number -  \n" ))
    if experiment_num == 1:
      case = str(input("For Experiment 1 choose between the cases a - c by typing the corresponding letter -  \n"))
      if case == 'a':
        print("|----------------Running Experiment 1 (a)----------------|\n")
        self.experiment_1_a()
      elif case == 'b':
        print("|----------------Running Experiment 1 (b)----------------|\n")
        self.experiment_1_b()
      elif case == 'c':
        print("|----------------Running Experiment 1 (c)----------------|\n")
        self.experiment_1_c()
    elif experiment_num == 2:
      print("|----------------Running Experiment 2----------------|\n")
      self.experiment_2()
    elif experiment_num == 3:
      print("|----------------Running Experiment 3----------------|\n")
      self.experiment_3()
    elif experiment_num == 4:
      print("|----------------Running Experiment 4----------------|\n")
      self.experiment_4()


class Main(PDWorld):
  def __init__(self):
    master = Tk()
    master.resizable(width=False, height=False)
    app = PDWorld(master)
    app.mainloop()

Main()