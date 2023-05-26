from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
from matplotlib import pyplot as plt
from data.stdp_data import data, nutrients, capacity


class Individual:
    def __init__(
        self,
        representation=None,
        size=None,
        replacement=True,
        valid_set=None,
    ):
        if representation == None:
            if replacement == True:
                self.representation = [choice(valid_set) for i in range(size)]
            elif replacement == False:
                self.representation = sample(valid_set, size)
        else:
            self.representation = representation
        self.fitness = self.get_fitness()
    
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
    
    def index(self, value):
        return self.representation.index(value)
    
    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        for _ in range(size):
            self.individuals.append(
                Individual(
                    size=kwargs["sol_size"],
                    replacement=kwargs["replacement"],
                    valid_set=kwargs["valid_set"],
                )
            )

    def evolve(self, gens, xo_prob, mut_prob, select, mutate, crossover, elitism):

        for i in range(gens):
            new_pop = []

            if elitism:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)

                if random() < xo_prob:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                if random() < mut_prob:
                    offspring1 = mutate(offspring1)
                if random() < mut_prob:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism:
                if self.optim == "max":
                    worst = min(new_pop, key=attrgetter("fitness"))
                    if elite.fitness > worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)

                elif self.optim == "min":
                    worst = max(new_pop, key=attrgetter("fitness"))
                    if elite.fitness < worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)

            self.individuals = new_pop

            if self.optim == "max":
                print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]
