from charles.charles import Population, Individual
from charles.search import hill_climb, sim_annealing
from copy import deepcopy
from data.ks_data import weights, values, capacity
from charles.selection import fps
from charles.mutation import binary_mutation
from charles.crossover import single_point_co
from random import random
from operator import attrgetter


def get_fitness(self):
    """A function to calculate the total weight of the bag if the capacity is not exceeded
    If the capacity is exceeded, it will return a negative fitness
    Returns:
        int: Total weight
    """
    fitness = 0
    weight = 0
    for bit in range(len(self.representation)):
        if self.representation[bit] == 1:
            fitness += values[bit]
            weight += weights[bit]
    if weight > capacity:
        fitness = capacity - weight
    return fitness


def get_neighbours(self):
    """A neighbourhood function for the knapsack problem,
    for each neighbour, flips the bits
    Returns:
        list: a list of individuals
    """
    n = [deepcopy(self.representation) for i in range(len(self.representation))]

    for index, neighbour in enumerate(n):
        if neighbour[index] == 1:
            neighbour[index] = 0
        elif neighbour[index] == 0:
            neighbour[index] = 1

    n = [Individual(i) for i in n]
    return n


# Monkey Patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours

pop = Population(size=20, optim="max", sol_size=len(values), valid_set=[0, 1], replacement=True)


# Just a test for GA's before implementing them in the library next week

# 100 = generations
for i in range(100):
    new_pop = []
    while len(new_pop) < len(pop):
        parent1, parent2 = fps(pop), fps(pop)

        # XO
        # 0.5 = probability of xo
        if random() < 0.5:
            offspring1, offspring2 = single_point_co(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2

        # Mutation
        # 0.5 = probability of mut
        if random() < 0.5:
            offspring1 = binary_mutation(offspring1)
        if random() < 0.5:
            offspring2 = binary_mutation(offspring2)

        new_pop.append(Individual(representation=offspring1))
        new_pop.append(Individual(representation=offspring2))

    pop.individuals = new_pop
    print(f'Best individual: {max(pop, key=attrgetter("fitness"))}')

