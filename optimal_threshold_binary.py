from MLAlgorithms import MLAlgorithms
from operator import add
from random import random, randint, uniform
from functools import reduce


###################### This are Methods for the Genetic algorithm
def individual(length, min, max):
    # Create a member of the population.
    return [uniform(min, max) for x in range(length)]


def population(count, length, min, max):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the min possible value in an individual's list of values
    max: the max possible value in an individual's list of values

    """
    return [individual(length, min, max) for x in range(count)]


def fitness(individual, target, p, y):
    """
    Determine the fitness of an individual. Lower is better.

    individual: the individual to evaluate
    target: the sum of numbers that individuals are aiming for
    """

    miss_classes = hamming_w(individual, p, y)
    return abs(target - miss_classes)

def grade(pop, target, p, y):
    # Find average fitness for a population.
    sum_of_fitnesses = 0
    fitness_result = 0
    min = [50000000000000000, [0,0]]
    for population in pop:

        fitness_result = fitness(population, target, p, y)
        sum_of_fitnesses += fitness_result

        if fitness_result < min[0]:
            min = [fitness_result, population]

    print("Best guess: " + str(min[1]) + ", " + str(min[0]) + " miss classes.")
    return sum_of_fitnesses / (len(pop))




def evolve(pop, target, p, y, retain=0.2, random_select=0.25, mutate=0.1):
    graded = [(fitness(x, target, p, y), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = uniform(
                min(individual), max(individual))

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int (len(male) / 2 )
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents

############################# Here ends the genetic algorithm


def hamming_thresh(t, p, y):
    mis_class = 0  # stores number of miss classification

    class_0_counter = 0
    class_1_counter = 0

    for k in range(0, p.shape[0]):

        entry = p[k]
        # Classify entry k

        if entry[0] >= t:
            y_hat = 0
            class_0_counter += 1

        else:
            y_hat = 1
            class_1_counter += 1

        # Check if entry k has been misclassified

        if y_hat != y[k]:
            mis_class += 1

    # print("Accuracy: " + str(100 - (100 * (mis_class/p.shape[0]))) + "% Threshold: " + str(t)
    #       + "\nMissed Classes: = " + str(mis_class) + " of " + str(p.shape[0]) + " Evaluated entries")
    # print("Counters = [" + str(class_0_counter) + ", " + str(class_1_counter) + "]\n")
    return mis_class



def hamming_w(w, p, y):
    #Add the weight for the fourth class
    class_0_counter = 0
    class_1_counter = 0

    class_0_miss = 0
    class_1_miss = 0



    for k in range(0, p.shape[0]):
        entry = p[k]

        entry[0] = entry[0] * w[0]
        entry[1] = entry[1] * w[1]

        winner = max(entry[0],entry[1])

        if winner == entry[0]:
            y_hat = 0
            class_0_counter += 1
        else:
            y_hat = 1
            class_1_counter += 1


        if y_hat != y[k]:

            if y[k] == 0:
                class_0_miss += 1
            else :
                class_1_miss += 1

    mis_class = class_0_miss + class_1_miss

    #Adjust probabilities
    # print("Accuracy = " + str(100 - (100 * (mis_class / p.shape[0]))) + "% \nWeights used was: " + str(w)
    #       + "\nMissed Classes: = " + str(mis_class) + " of " + str(p.shape[0]) + " Evaluated entries")
    #
    # print("Counters = [" + str(class_0_counter) + ", " + str(class_1_counter) + "]")
    # print("Miss Classes = [" + str(class_0_miss) + ", " + str(class_1_miss) +"]\n")

    return mis_class




models = MLAlgorithms(['Logistic Regression'], scaler='MinMax Scaler', binary=True)
models.autopilot('/Users/cristian/Projects/Paratransit-Project/updated_data.csv')
results = models.models[0]
probabilities = results[3]
y = models.data[['CLASS']].copy().values.ravel()


#
# thresh = 0.005
# for i in range(0, 200):
#     hamming_thresh(thresh, probabilities, y)
#     thresh += 0.005


target = 0
p_count = 10
i_length = 2
i_min = 0.00
i_max = 5.00
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target, probabilities, y)]

for i in range(10):
    print("Iteration: " + str(i))
    p = evolve(p, target, probabilities, y)
    fitness_history.append(grade(p, target, probabilities, y))

for datum in fitness_history:
   print(datum)

