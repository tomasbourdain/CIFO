import numpy as np
from charles.crossover import arithmetic_xo, pmx, single_point_co, cycle_xo
from charles.mutation import inversion_mutation, swap_mutation
from charles.charles import Population, Individual
from charles.search import hill_climb, sim_annealing
from copy import deepcopy
from data.stdp_data import data, nutrients, capacity
from charles.selection import fps, tournament_sel, rank
from random import choice, random, randint, sample
from operator import attrgetter
import matplotlib.pyplot as plt


def get_fitness(self):
    """A function to calculate the total weight of the bag if the capacity and nutrient constraints are not exceeded.
    If the capacity or nutrient constraints are exceeded, it will return a negative fitness.
    Returns:
        int: Total weight
    """
    fitness = 0
    weight = 0
    nutrient_totals = [0] * len(nutrients)

    for bit in range(len(self.representation)):
        if self.representation[bit] == 1:
            fitness += data[bit][2]  # Add the price of the commodity
            weight += data[bit][3]  # Add the weight of the commodity

            # Update nutrient totals
            for nutrient_index in range(len(nutrients)):
                nutrient_totals[nutrient_index] += data[bit][nutrient_index + 3]

    if weight > capacity:
        return -1 * (weight - capacity)  # Return negative fitness if capacity is exceeded

    # Check nutrient constraints
    for nutrient_index in range(len(nutrients)):
        if nutrient_totals[nutrient_index] < nutrients[nutrient_index][1]:
            return -1 * (nutrient_totals[nutrient_index] - nutrients[nutrient_index][1])

    return fitness
   
def get_neighbours(self):
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

# pop = Population(
#     size=20,
#     sol_size=len(data),
#     valid_set=[0,1],
#     replacement=True,
#     optim="max")

# # pop.evolve(gens=100, select=tournament_sel, mutate=inversion_mutation, crossover=pmx,
# #             mut_prob=0.05, xo_prob=0.9, elitism=True)
# results = []
# for i in range(10):
#     result = []
#     pop.evolve(gens=100, select=fps, mutate=swap_mutation, crossover=single_point_co,
#             mut_prob=0.05, xo_prob=0.9, elitism=True)
#     result.append(pop.individuals)
#     results.append(result)
#     print(results)
#     print("\n", results[0])
#     print("\n", results)

# results = np.array(results)
# mean_results = np.mean(results, axis=(1, 2))
# plt.plot(range(10), np.mean(mean_results, axis=1))
# plt.ylabel('Mean Fitness')
# plt.show()

# pop = Population(
#     size=50,
#     sol_size=len(data),
#     valid_set=[0,1],
#     replacement=True,
#     optim="max")

#plot the medium fitness for every generation thoughout 10 iterations 
def plt_mean_fitness(select, mutate, crossover, elitism, iterations):
    pop = Population(
    size=30,
    sol_size=len(data),
    valid_set=[0,1],
    replacement=True,
    optim="max")
    results = []
    for i in range(iterations):
        result = []
        pop.evolve(gens=100, select=select, mutate=mutate, crossover=crossover,
                mut_prob=0.05, xo_prob=0.9, elitism=elitism)
        result.append(pop.individuals[-1].fitness)
        results.append(result)
        print(results)
        print("\n", results[0])
        print("\n", results)

    results = np.array(results)
    return results

iterations = 30
results_rank_1 = plt_mean_fitness(rank, swap_mutation, single_point_co, True, iterations)

results_rank_2 = plt_mean_fitness(rank, inversion_mutation, arithmetic_xo, True, iterations)

results_rank_3 = plt_mean_fitness(rank, swap_mutation, arithmetic_xo, True, iterations)

results_rank_4 = plt_mean_fitness(rank, inversion_mutation, single_point_co, True, iterations)

# results_rank_3 = plt_mean_fitness(fps, swap_mutation, single_point_co, True, iterations)

# results_rank_4 = plt_mean_fitness(fps, inversion_mutation, arithmetic_xo, True, iterations)

# results_rank_5 = plt_mean_fitness(fps, swap_mutation, arithmetic_xo, True, iterations)

# results_rank_6 = plt_mean_fitness(fps, inversion_mutation, single_point_co, True, iterations)

results_rank_7 = plt_mean_fitness(tournament_sel, swap_mutation, single_point_co, True, iterations)

results_rank_8 = plt_mean_fitness(tournament_sel, swap_mutation, arithmetic_xo, True, iterations)

# results_rank_9 = plt_mean_fitness(tournament_sel, inversion_mutation, arithmetic_xo, False, iterations)
results_rank_9 = plt_mean_fitness(tournament_sel, inversion_mutation, arithmetic_xo, True, iterations)

results_rank_10 = plt_mean_fitness(tournament_sel, inversion_mutation, single_point_co, True, iterations)

# results_rank_11 = plt_mean_fitness(tournament_sel, inversion_mutation, cycle_xo, True, iterations)

results_rank_12 = plt_mean_fitness(tournament_sel, inversion_mutation, pmx, True, iterations)


#plot results_rank_1 and results_rank_2 add legend for each results
plt.plot(range(iterations), results_rank_1, label='Rank/Swap/Single') #1
plt.plot(range(iterations), results_rank_2, label='Rank/Inversion/Arithmetic') #2
plt.plot(range(iterations), results_rank_3, label='Rank/Swap/Arithmetic') #3
plt.plot(range(iterations), results_rank_4, label='Rank/Inversion/Single') #4
# plt.plot(range(iterations), results_rank_3, label='FPS/Swap/Single') #3
# plt.plot(range(iterations), results_rank_4, label='FPS/Inversion/Arithmetic') #4
# plt.plot(range(iterations), results_rank_5, label='FPS/Swap/Arithmetic') #5
# plt.plot(range(iterations), results_rank_6, label='FPS/Inversion/Single') #6
plt.plot(range(iterations), results_rank_7, label='Tournament/Swap/Single') #7
plt.plot(range(iterations), results_rank_8, label='Tournament/Swap/Arithmetic') #8
# plt.plot(range(iterations), results_rank_9, label='Tournament/Inversion/Arithmetic') #9
plt.plot(range(iterations), results_rank_9, label='Tournament/Inversion/Arithmetic') #9
plt.plot(range(iterations), results_rank_10, label='Tournament/Inversion/Single') #10
# plt.plot(range(iterations), results_rank_11, label='Tournament/Inversion/Cycle') #11
plt.plot(range(iterations), results_rank_12, label='Tournament/Inversion/PMX') #12

plt.ylabel('Mean Fitness')
plt.xlabel('Iteration')
plt.title('Mean Fitness for 30 iterations')
plt.legend()
plt.show()


