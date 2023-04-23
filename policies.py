import random


# Random Policy: If pickup and dropoff is applicable, choose this operator; otherwise, choose an applicable operator randomly.
def PRANDOM(actions):
    return random.choice(actions)

# Exploit Policy: If pickup and dropoff is applicable, choose this operator; otherwise, apply the applicable operator with the 
# highest q-value (break ties by rolling a dice for operators with the same q-value) with probability 0.85 and choose a different 
# applicable operator randomly with probability 0.15. 
def PEXPLOIT(actions, agent, row, col):
    duplicate = []
    q_table = dropoff_q_table if agent.blocked() else pickup_q_table

    # Choose action (operator) with highet q-value 85% of the time
    if random.random() <= 0.85:
        max_operator = actions[0]
        for num in actions:
            q_value = q_table[row][col][num]
            max_q_value = q_table[row][col][max_operator]

            if q_value > max_q_value:
                max_operator = num
                duplicate.clear()
                duplicate.append(num)
            if q_value == max_q_value:
                duplicate.append(num)

        exploit_choice = random.choice(duplicate) if len(duplicate) > 1 else max_operator
    # The rest of the time 15% choose action randomly
    else:
        exploit_choice = random.choice(actions)

    return exploit_choice

# Greedy Policy: If pickup and dropoff is applicable, choose this operator; otherwise, apply the applicable operator with the 
# highest q-value (break ties by rolling a dice for operators with the same q-value). 
def PGREEDY(actions, agent, row, col):
    duplicate = []
    q_table = dropoff_q_table if agent.blocked() else pickup_q_table

    # Choose action (operator) with best q-value 100% of the time
    max_operator = actions[0]
    for num in actions:
        q_value = q_table[row][col][num]
        max_q_value = q_table[row][col][max_operator]

    if q_value > max_q_value:
        max_operator = num
        duplicate.clear()
        duplicate.append(num)
    if q_value == max_q_value:
        duplicate.append(num)

    greedy_choice = random.choice(duplicate) if len(duplicate) > 1 else max_operator

    return greedy_choice