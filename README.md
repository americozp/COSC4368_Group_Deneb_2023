# COSC4368_Group_2023
Using Reinforcement Learning  To Discover Paths in a 2-Agent 3D Transportation World

In this project we will use reinforcement to learn and adapt “promising paths” in 2-agent setting. Learning objectives of the 2023 COSC 4368 Group Project include:
•	Understanding basic reinforcement learning concepts such as utilities, policies, learning rates, discount rates and their interactions.
•	Obtain experience in designing agent-based systems that explore and learn in initially unknown environment and which are capable to adapt to changes. 
•	Learning how to conduct experiments that evaluate the performance of reinforcement learning systems and learning to interpret such results. 
•	Development of visualization techniques summarizing how the agents move, how the world and the q-table changes, and the system performance. 
•	Development of path visualization and analysis techniques to interpret and evaluate the behavior of agent-based path-learning systems.
•	Develop and learn coordination strategies for collaborating agents
•	Creating solutions for 3D visualization problems 
•	Learning to develop AI software in a team. 

In particular in this project we will use Q-learning/SARSA  for the 3- dimensional PD-Word assuming a 2 agent setting (http://www2.cs.uh.edu/~ceick/ai/2023-World.pptx), conducting four experiments using different parameters and policies, and summarize and interpret the experimental results. Moreover, you will develop path analysis and visualization techniques that are capable to shed light on what paths the learning system actually has learnt from obtained Q-Tables—we call such paths attractive paths in the remainder of this document. Moreover, you will analyze if the two agents collaborated well by avoiding blockage that occurs if the two agents work on the same path. Finally, you will analyze how close your approach came to an Optimal 2 Agent Policy. 

Two agent named ‘M’ (male) and ‘F (female) are solving the block transportation problem jointly. Agent alternate applying operators to the 3D PD-World, with the female agent acting first. In this world the agents can move east, west, north, south, up and down. Moreover, both agents cannot be in the same position at the same time; consequently, there is a blockage problem, limiting agent mobility and ultimately efficiency in case that both agents work on the same path at the same time. Additionally, there are two cells that are very risky and they return 2 times more negative reward than normal cells. There are two approaches to choose from to implement 2-agent reinforcement learning.
a.	Each agent uses his/her own reinforcement learning strategy and Q-Table. However, we assume that the position the other agent occupies is visible to each agent, and can therefore can be part of the chosen reinforcement learning state space. 
b.	A single reinforcement learning strategy and Q-Table is used which moves both agents, selecting an operator for each agent and then executing the selected two operators.

Extra credit is given to groups who devise and implement both 2-agent learning approaches with all the constraints and compare their results for experiments 2 and 3 (see below) 

In experiments we assume that q values are initialized with 0 at the beginning of the experiment. The following 3 policies will be used in the experiments:

•	PRANDOM: If pickup and dropoff is applicable, choose this operator; otherwise, choose an applicable operator randomly.
•	PEXPLOIT: If pickup and dropoff is applicable, choose this operator; otherwise, apply the applicable operator with the   highest q-value (break ties by rolling a dice for operators with the same q-value) with probability 0.85 and choose a different applicable operator randomly with probability 0.15. 
•	PGREEDY: If pickup and dropoff is applicable, choose this operator; otherwise, apply the applicable operator with the highest q-value (break ties by rolling a dice for operators with the same q-value). 
