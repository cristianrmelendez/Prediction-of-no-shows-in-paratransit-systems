import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'demandData/Random Forest_Normal Scaler_True_demandData3.csv'
raw_data = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)

# Step 1.1: Concact Dates: Proccesing Data

def comparative_plot(real_demand, expected_demand, predicted_demand):
    time = np.arange(1, len(real_demand) + 1, 1)
    plt.figure(0)
    plt.plot(time, real_demand, label="Real Demand", linewidth=4)
    plt.plot(time, expected_demand, label="Expected Demand", linestyle='--', color="green")
    plt.plot(time, predicted_demand, label="Predicted Demand", linestyle='--', color="red")
    plt.title("System Demand Across Time")
    plt.ylabel("Demand")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig("plot1.png")
    plt.show()
    plt.close(0)


new_dates = []
data =[]
for k in range(len(raw_data[:, 0])):
    day = str(int(raw_data[k, 0]))
    month = str(int(raw_data[k, 1]))
    year = str(int(raw_data[k, 2]))

    if len(month) == 1:
        month = month + "0"
        month[::-1]

    new_dates.append(float(day + month + year))

print(new_dates)
for i in range(0, len(new_dates)):
    data.append([new_dates[i], raw_data[i, 3], raw_data[i, 4], raw_data[i, 5]])

data = np.array(data)



# Step 1.2: Creating a Data-Frame and an Excel Writer.

writer = pd.ExcelWriter("demand_estimation.xlsx")


# Step 2.1: Demand Estimator Function

def demand_estimator(real_outcomes, predicted_classes, performed_prob):
    # Step a: Calculate Real Demand Assuming Performed Code is "1".

    real_demand = np.sum(real_outcomes == 1)

    # Step b: Claculate the Expected Value

    expected_demand = np.sum(performed_prob)

    # Step c: Calculate the Prediction Assuming the Code for Performed is "1"

    predicted_demand = np.sum(predicted_classes == 1)

    # Step d: Calculating the Difference

    if real_demand == 0:  # Percentage of Difference Not Defined.
        if expected_demand == 0:
            diff_1 = 0
        else:
            diff_1 = 100

        if predicted_demand == 0:
            diff_2 = 0
        else:
            diff_2 = 100
    else:
        diff_1 = (np.absolute(
            (expected_demand - real_demand) / real_demand) * 100)  # Difference Between Expected Value and Real Demand
        diff_2 = (np.absolute(
            (predicted_demand - real_demand) / real_demand) * 100)  # Difference Between Predicted Class and Real Demand

    return real_demand, predicted_demand, expected_demand, diff_1, diff_2


# Step 2.2 Demand distribution function for a day
def demand_distribution(num_reserve, performed_prob, num_simul):
    all_sim_demand = []  # empty list that stores simulation demand

    # For each simulation
    for k in range(0, num_simul):
        # Step a. Generate random numbers
        w = np.random.random((num_reserve, 1))
        # Step b. Simulation k of demand on day t
        demand = sum(w < performed_prob)[0]
        # Step c. Add simulated demand to container list
        all_sim_demand.append(demand)

    return all_sim_demand


# Step 3: Estimating the demand

dd = {"Day": [], "Predicted Demand": [], "Expected Demand": [], "Real Demand": [], "Diff 1": [], "Diff 2": []}
unique_days = np.unique(data[:, 0])

for day in unique_days:
    # Step a: Grab "day" data.

    idx = (data[:, 0] == day)
    day_data = data[idx, :]

    # Step b: Estimating Demand

    [real_demand, predicted_demand, expected_demand, diff_1, diff_2] = demand_estimator(day_data[:, 1], day_data[:, 2],
                                                                                        day_data[:, 3])

    # Step c: Saving Data


    dd["Day"].append(day)
    dd["Predicted Demand"].append(predicted_demand)
    dd["Expected Demand"].append(expected_demand)
    dd["Real Demand"].append(real_demand)
    dd["Diff 1"].append(diff_1)
    dd["Diff 2"].append(diff_2)


comparative_plot(dd["Real Demand"], dd["Expected Demand"], dd["Predicted Demand"])
df = pd.DataFrame(dd)
df = df[["Day", "Predicted Demand", "Expected Demand", "Real Demand", "Diff 1", "Diff 2"]]
df.to_excel(writer, sheet_name="Demand Forecasting")
writer.save()

# Step 4. Creating histogram and fitting normal curve to
from scipy.stats import norm
import matplotlib.pyplot as plt



# Step a. Select the probability data of any day
day = [2502016]  # WRITE HERE A specific DAY in the data!!!!
idx = (data[:, 0] == day)
day_data = data[idx, :]
num_reserve = sum(idx)  # number of reservations in a day [DRR]
# Step b. Define number of simmulations
num_simul = 1000
print(num_reserve)

# Step c. Simulate demand instances for day
sim_demand = demand_distribution(num_reserve, day_data[:, 3], num_simul)

# Step d. Build a histogram.
plt.rcParams["patch.force_edgecolor"] = True  # this creates the bar colors
plt.hist(sim_demand, bins=25, density=True, color='g')

# Step e. Fit a normal distribution to the data:
mu, std = norm.fit(sim_demand)

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()

