import random
from math import sqrt

import numpy as np

playerName = "myAgent"
nPercepts = 75  #  This is the number of percepts
nActions = 5    #  This is the number of actions

# Train against random for 5 generations, then against self for 1 generations
trainingSchedule = [("hunter", 200)]

# Author : Quinn McCabe - S_ID: 8653644
# This is the class for your creature/agent


class MyCreature:

    """
        This init function wil create the instance of our agent, defining the chromosone as random variables.
        The details on the (vague) behaviour that the chromosone maps to is described below.
    """
    def __init__(self):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values
        self.chromosone = []
        for i in range(0, 11):
            self.chromosone.append(random.uniform(-100, 100))
        # 1 = Hunter - Hunt Enemies much smaller
        # 2 = Aggressive - Hunt enemies smaller or same size
        # 3 = Friendly - Find Friendlies
        # 4 = Whale - Get as big as possible
        # 5 = Scared - Avoid Enemies and Walls
        # 6 = Agitated - Move a lot
        # 7 = Hungry - Seeks out fruit
        # 8 = Pacifist - Runs away from enemies
        # 9 = Safety - First Hugs the walls
        # 10 = SideKick - Follows bigger friendlies
        # 11 = Adjust - Effects starting direction
        # .
        # .

    '''
        This function is passed any location on a precept where there is something noteworthy.
        From here it figures out the direction the agent should move in order to either go towards it,
        or to avoid it.
        This information is then weighed with the chromosone values to determine behaviour.
    '''
    def EnemyDirections(self, enemy):
        direction = -1
        avoid = -1
        if enemy == [2, 2]:
            return
        if(enemy[0] == 2) & (enemy[1] < 2):
            direction = 1  # up
            avoid = 3
        if(enemy[0] == 2) & (enemy[1] > 2):
            direction = 3  # down
            avoid = 1
        if enemy[0] > 2:
            direction = 2  # right
            avoid = 0
        if enemy[0] < 2:
            direction = 0  # left
            avoid = 2

        if direction > -1:
            return [direction, avoid]

    def AgentFunction(self, percepts):

        actions = np.zeros((nActions))

        # Define precepts
        enemies = percepts[:, :, 0]
        board_info = percepts[:, :, 1]
        walls = percepts[:, :, 2]
        our_size = percepts[2][2][0]
        decisions = [1, 1, 1, 1, 1]

        # Lists are defined to hold the directions where enemies, walls, etc, can be found.
        enemy_direction = []
        food_direction = []
        wall_direction = []
        friendly_direction = []
        large_enemy_direction = []
        large_friendly_direction = []

        #Scan precepts for object information
        for enemy, value in np.ndenumerate(enemies):
            if value > 0:
                # friendly unit
                if value < our_size * -1:
                    large_friendly_direction.append((self.EnemyDirections(enemy)))
                friendly_direction.append((self.EnemyDirections(enemy)))
            elif value < 0:
                if value > our_size:
                    # large enemy
                    large_enemy_direction.append(self.EnemyDirections(enemy))
                else:
                    # small enemy
                    enemy_direction.append(self.EnemyDirections(enemy))
        for food, value in np.ndenumerate(board_info):
            if value > 0:
                # food nearby
                food_direction.append(self.EnemyDirections(food))
        for wall, value in np.ndenumerate(walls):
            if value > 0:
                # wall nearby
                wall_direction.append(self.EnemyDirections(wall))
        if percepts[2, 2, 1] == 1:
            # food to eat
            decisions[4] = self.chromosone[6] ** 5
        else:
            decisions[4] = -1000

        # Remove all none values from the lists
        wall_direction = list(filter(None, wall_direction))
        enemy_direction = list(filter(None, enemy_direction))
        large_enemy_direction = list(filter(None, large_enemy_direction))
        friendly_direction = list(filter(None, friendly_direction))
        food_direction = list(filter(None, food_direction))
        # Affect the starting moveement direction
        decisions[int(self.chromosone[10] % 4)] += 1

        # Use precept information to find directions away and towards objects.
        for item in wall_direction:
            decisions[item[0]] += self.chromosone[3]  # towards wall
            decisions[item[1]] += self.chromosone[4]  # away from wall
        for item in enemy_direction:
            decisions[item[0]] += self.chromosone[0] + self.chromosone[1] + self.chromosone[3]  # towards enemy
            decisions[item[1]] += self.chromosone[7] + self.chromosone[8] # away from enemy
        for item in large_enemy_direction:
            decisions[item[0]] += self.chromosone[1]  # towards large enemy
            decisions[item[1]] += self.chromosone[7] + self.chromosone[8] + self.chromosone[4]  # away from large enemy
        for item in friendly_direction:
            decisions[item[0]] += self.chromosone[2]  # towards friendlies
            decisions[item[1]] += self.chromosone[3] + self.chromosone[0] + self.chromosone[8]
        for item in large_friendly_direction:
            decisions[item[0]] += self.chromosone[9]
            decisions[item[1]] += self.chromosone[1]
        for item in food_direction:
            decisions[item[0]] += self.chromosone[0] + self.chromosone[6] + self.chromosone[3] + self.chromosone[1]  # towards food
            decisions[item[1]] += self.chromosone[4]  # away from food


        actions[0] = decisions[0] + self.chromosone[5]
        actions[1] = decisions[1] + self.chromosone[5]
        actions[2] = decisions[2] + self.chromosone[5]
        actions[3] = decisions[3] + self.chromosone[5]
        actions[4] = decisions[4]

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.
        #
        # The 'actions' variable must be returned and it must be a 5-dim numpy vector or a
        # list with 5 numbers.
        #
        # The index of the largest numbers in the 'actions' vector/list is the action taken
        # with the following interpretation:
        # 0 - move left
        # 1 - move up
        # 2 - move right
        # 4 - eat
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        # .
        # .
        # .

        return actions
'''
    This function performs a tournament selection to choose two parents from the population.
    The amount of participants is passed into the function, and from these participants we choose 
    two parents for the new generation based on fitness.
'''
def tournamentSelect(population, participants):
    better_index = -1
    best_index = -1
    best = 0
    for i in range(0, participants):
        loc = random.randint(0, len(population)-1)
        individual = population[loc]
        if(best == 0) or individual > best:
            better = best
            better_index = best_index
            best_index = loc
            best = individual

    return [best_index, better_index]
'''
    This function performs roulette selection to choose two parents from the population.
    Each individual has a chance to be selected, with their chance weighted on their fitness.
'''
def rouletteSelection(old_population, fitness, total_fit):
    relative_fit = [fit/total_fit for fit in fitness]
    probabilities = [sum(relative_fit[:i+1]) for i in range(len(relative_fit))]
    parents_indexes = []
    for n in range(2):
        random_choice = random.uniform(0, 1)
        for i in range(len(old_population)):
            if random_choice <= probabilities[i]:
                parents_indexes.append(i)
    return parents_indexes
def newGeneration(old_population):

    # This function should return a list of 'new_agents' that is of the same length as the
    # list of 'old_agents'.  That is, if previous game was played with N agents, the next game
    # should be played with N agents again.

    # This function should also return average fitness of the old_population
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boiler plate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, creature in enumerate(old_population):

        # creature is an instance of MyCreature that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, the objects has attributes provided by the
        # game engine:
        #
        # creature.alive - boolean, true if creature is alive at the end of the game
        # creature.turn - turn that the creature lived to (last turn if creature survived the entire game)
        # creature.size - size of the creature
        # creature.strawb_eats - how many strawberries the creature ate
        # creature.enemy_eats - how much energy creature gained from eating enemies
        # creature.squares_visited - how many different squares the creature visited
        # creature.bounces - how many times the creature bounced

        # .
        # .
        # .

        # This fitness functions just considers length of survival.  It's probably not a great fitness
        # function - you might want to use information from other stats as well



        fitnessFunc = (creature.alive + 1) * creature.size + (creature.squares_visited/(creature.bounces + 1)) * ((creature.enemy_eats * 2) + creature.strawb_eats)
        fitness[n] = fitnessFunc/100

    # At this point you should sort the agent according to fitness and create new population
    new_population = list()
    for n in range(N):

        # Create new creature
        new_creature = MyCreature()
        total_fit = sum(fitness)

        # Allows for easy choice between tournament selection and roulette selection
        parents = tournamentSelect(fitness, 8)
        #parents = rouletteSelection(old_population, fitness, total_fit)
        parent_1 = old_population[parents[1]]
        parent_2 = old_population[parents[0]]

        # Retrieving two random points for our crossover method
        k_points = random.sample(range(0, len(parent_1.chromosone)+1), 2)
        k_one = min(k_points)
        k_two = max(k_points)

        # 1 in 100 chance of mutation
        mutation_rate = 0.0033

        # Crossover occurs between the two parents.
        # A coin flip is performed to determine which parent becomes parent 1 and which is parent 2.
        # Using the K points determined earlier we select parts of each chromosone for the child.
        coin_flip = random.randint(0, 1)
        if coin_flip == 1:
            new_creature.chromosone[:k_one] = parent_2.chromosone[:k_one]
            new_creature.chromosone[k_one:k_two] = parent_1.chromosone[k_one:k_two]
            new_creature.chromosone[k_two:] = parent_2.chromosone[k_two:]
        else:
            new_creature.chromosone[:k_one] = parent_1.chromosone[:k_one]
            new_creature.chromosone[k_one:k_two] = parent_2.chromosone[k_one:k_two]
            new_creature.chromosone[k_two:] = parent_1.chromosone[k_two:]

        # Generate a random number, if it is below our mutation rate perform a mutation.
        # A mutation will randomly alter a portion of the child chromosone.
        mutate_check = random.uniform(0, 1)
        if mutate_check < mutation_rate:
            chromosone_mutation = random.randint(0, 10)
            new_creature.chromosone[chromosone_mutation] = random.uniform(0, 10)
        # Add the new agent to the new population
        new_population.append(new_creature)

    # using elitism we find the maximum fit individual and keep it in our new generation.
    # This elite individual takes the place of a random individual in our new generation.
    index_max = np.argmax(fitness)
    new_population[random.randint(0, 33)] = old_population[index_max]

    avg_fitness = np.mean(fitness)

    # We write to two CSV files.
    # One holds the average fit of each generation over time.
    # The second holds the max fit of each generation over time.
    with open("stats.csv", "a") as myfile:
        myfile.write(str(avg_fitness))
        myfile.write(',')
        myfile.write('\n')
    myfile.close()


    return (new_population, avg_fitness)

