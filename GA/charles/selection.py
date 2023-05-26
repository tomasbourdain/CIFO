import math
from operator import attrgetter
from random import uniform, choice


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    if population.optim == "max":

        # Sum total fitness
        total_fitness = sum([i.fitness for i in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
        raise NotImplementedError

    else:
        raise Exception("No optimization specified (min or max).")

def tournament_sel(population, size=3):
    """Tournament selection implementation.

    Args:
        population (Population): The population we want to select from.
        size (int): Size of the tournament.

    Returns:
        Individual: The best individual in the tournament.
    """

    # Select individuals based on tournament size
    # with choice, there is a possibility of repetition in the choices,
    # so every individual has a chance of getting selected
    tournament = [choice(population.individuals) for _ in range(size)]

    # with sample, there is no repetition of choices
    # tournament = sample(population.individuals, size)
    if population.optim == "max":
        return max(tournament, key=attrgetter("fitness"))
    if population.optim == "min":
        return min(tournament, key=attrgetter("fitness"))
                       
def rank(population):
  """Rank selection implementation.
  Args:
      population (Population): The population we want to select from.
  Returns:
      Individual: selected individual.
  """
  if population.optim == "max":
      # Sort the population by fitness
      sorted_population = sorted(population, key=attrgetter('fitness'), reverse=True)
      # Get a 'position' on the wheel
      spin = uniform(0, len(population))
      position = 0
      # Find individual in the position of the spin
      for individual in sorted_population:
          position += 1
          if position > spin:
              return individual
  elif population.optim == "min":
      raise NotImplementedError
  else:
      raise Exception("No optimization specified (min or max).")





