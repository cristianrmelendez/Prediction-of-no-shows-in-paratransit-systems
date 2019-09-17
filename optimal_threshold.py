from MLAlgorithms import MLAlgorithms
from operator import add
from random import random, randint, uniform
from functools import reduce

from lmfit import Parameters, minimize, report_fit




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

    miss_classes = hamming_thresh(individual, p, y)
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


def evolve(pop, target, p, y, retain=0.05, random_select=0.40, mutate=.30):
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


def hamming_thresh(params, p, y):
    mis_class = 0  # stores number of miss classification

    class_1_diff = 0
    class_2_diff = 0
    class_3_diff = 0
    class_4_diff = 0

    class_1_counter = 0
    class_2_counter = 0
    class_3_counter = 0
    class_4_counter = 0

    for k in range(0, p.shape[0]):

        entry = p[k]
        # Classify entry k

        if entry[0] > params[0]:
            class_1_diff = entry[0] - params[0]

        if entry[1] > params[1]:
            class_2_diff = entry[1] - params[1]

        if entry[2] > params[2]:
            class_3_diff = entry[2] - params[2]

        if entry[3] > params[3]:
            class_4_diff = entry[3] - params[3]


        winner = max([class_1_diff, class_2_diff, class_3_diff, class_4_diff])

        if winner == class_1_diff:
            y_hat = 1
            class_1_counter += 1
        elif winner == class_2_diff:
            y_hat = 2
            class_2_counter += 1
        elif winner == class_3_diff:
            y_hat = 3
            class_3_counter += 1
        else:
            y_hat = 4
            class_4_counter += 1

        # Check if entry k has been misclassified
        if y_hat != y[k]:
            mis_class += 1

    print("Accuracy = " + str(100 - (100 * (mis_class/p.shape[0]))) + "% \nThreshold used was: " + str(params)
          + "\nMissed Classes: = " + str(mis_class) + " of " + str(p.shape[0]) + " Evaluated entries")
    print("Counters = [" + str(class_1_counter) + ", " + str(class_2_counter) + ", " + str(class_3_counter) + ", " +
          str(class_4_counter) + "]\n")
    return mis_class




def hamming_w(w, p, y):
    #Add the weight for the fourth class
    class_1_counter = 0
    class_2_counter = 0
    class_3_counter = 0
    class_4_counter = 0
    class_1_miss = 0
    class_2_miss = 0
    class_3_miss = 0
    class_4_miss = 0

    for k in range(0, p.shape[0]):
        entry = p[k]

        entry[0] = entry[0] * w[0]
        entry[1] = entry[1] * w[1]
        entry[2] = entry[2] * w[2]
        entry[3] = entry[3] * w[3]

        winner = max(entry[0],entry[1], entry[2], entry[3])

        if winner == entry[0]:
            y_hat = 1
            class_1_counter += 1
        elif winner == entry[1]:
            y_hat = 2
            class_2_counter += 1
        elif winner == entry[2]:
            y_hat = 3
            class_3_counter += 1
        else:
            y_hat = 4
            class_4_counter += 1

        if y_hat != y[k]:

            if y[k] == 1:
                class_1_miss += 1
            elif y[k] == 2:
                class_2_miss += 1
            elif y[k] == 3:
                class_3_miss += 1
            else:
                class_4_miss += 1

    mis_class = class_1_miss + class_2_miss + class_3_miss + class_4_miss

    #Adjust probabilities
    print("Accuracy = " + str(100 - (100 * (mis_class / p.shape[0]))) + "% \nWeights used was: " + str(w)
          + "\nMissed Classes: = " + str(mis_class) + " of " + str(p.shape[0]) + " Evaluated entries")

    print("Counters = [" + str(class_1_counter) + ", " + str(class_2_counter) + ", " + str(class_3_counter) + ", " +
          str(class_4_counter) + "]")
    print("Miss Classes = [" + str(class_1_miss) + ", " + str(class_2_miss) + ", " + str(class_3_miss) + ", " +
          str(class_4_miss) + "]\n")

    return mis_class

#
#
# testDataFilePath = '/Users/cristian/Projects/Paratransit-Project/updated_data.csv'
# normal_scaler = preprocessing.Normalizer()
# print('Creating scalers...')
#
# print('Reading Data...')
# # Read Data from path
# data = pandas.read_csv(testDataFilePath)
#
# #  [PART 1 of DANIEL INSTRUCTIONS ]
# training_data = data.loc[data['Create_YEAR'].isin([2012, 2013, 2014, 2015])]
# testing_data = data.loc[data['Create_YEAR'].isin([2016, 2017, 2018, 2019])]



#  [PART 2 of DANIEL INSTRUCTIONS]
# Dividing the data frames into sub samples
# 55,000 to divide the data Frame in 10 different data frames
max_rows = 55000
sub_samples = []

# # This will create 10 sub samples of the training data
# while len(training_data) > max_rows:
#     top = training_data[:max_rows]
#     sub_samples.append(top)
#     training_data = training_data[max_rows:]
# else:
#     sub_samples.append(training_data)

#
# [PART 3 of DANIEL INSTRUCTIONS]
# Store the probabilities in array P = Probabilities
#
#
# for k in range(1, len(sub_samples)):
#
#     training_samples = pandas.DataFrame()
#
#     # To get the k - 1 samples as the Training samples
#     for i in range(0, k):
#         training_samples = training_samples.append(sub_samples[i], ignore_index=True)
#
#
#
#     # Divide into Inputs and Outputs
#     (x, y) = divideIO(training_samples)
#
#     # Scale Data
#     x = scaleData(normal_scaler, x)
#
#     try:
#         # Create Models
#         rfModel = RandomForestClassifier()
#
#         # Train Models
#         trainModel(x, y, rfModel)
#
#         # Make Predictions
#         # Now we get to Predict the data of Sub sample K using the k - 1 samples as the training data
#
#         (x_test, y_test) = divideIO(sub_samples[k])
#         x_test = scaleData(normal_scaler, x_test)
#
#         (rfPredictions, rfProbabilities) = makePredictions(x_test, rfModel)
#
#         print("Iteration " + str(k) + " finish")
#
#         # This will be the intial value
#         treshs = np.array([[0.700, 0.3300, 0.600, 0.300]])
#         res = minimize(hamming_thresh, treshs, args=(rfProbabilities, y_test), method='Nelder-Mead',
#         options = {'xtol': 1e-8, 'disp': True})
#
#         print(res)
#     except:
#         print('Error: ')
#         pass


models = MLAlgorithms(['Logistic Regression'], scaler='MinMax Scaler', binary=False)
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
p_count = 20
i_length = 4
i_min = 0.01
i_max = 1.00
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target, probabilities, y)]

for i in range(20):
    print("Iteration: " + str(i))
    p = evolve(p, target, probabilities, y)
    fitness_history.append(grade(p, target, probabilities, y))

for datum in fitness_history:
   print(datum)

