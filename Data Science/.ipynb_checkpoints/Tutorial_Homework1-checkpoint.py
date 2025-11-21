import random
import matplotlib.pyplot as plt

# -- Global Simulation Parameters ---
numSimulations = 100000
numEmployees = 20
totalWeeklyAttendence = [0] #initialize for 5 days


def simulateOneWeek(numEmployees):
    """Simulates one week and returns a list of 5 daily attendance counts."""
    weeklyAttendence = [0, 0, 0, 0, 0] #initialize for 5 days
    dayIndices = range(5)
    for i in range(numEmployees): #loop through each employee
        officeDays = random.sample(dayIndices, 2) #randomly select 2 days for each employee
        for day in officeDays:
            weeklyAttendence[day] += 1 #increment the count for that day
    return weeklyAttendence

def twoEmpolyeeOverlap(numSimulations):
    """Simulates one week for 2 employees and returns the number of overlapping days."""
    overlapSuccess = 0
    dayIndices = range(5)
    for x in range(numSimulations):
        employeeA = set(random.sample(dayIndices, 2))
        employeeB = set(random.sample(dayIndices, 2))
        if employeeA == employeeB:
            overlapSuccess += 1
    probability = overlapSuccess/numSimulations
    return probability


for week in range(numSimulations): #loop through num of simulations
    attendanceThisWeek = simulateOneWeek(numEmployees)
    totalWeeklyAttendence.extend(attendanceThisWeek) #add the 5 days to the total list

plt.hist(totalWeeklyAttendence, bins=range(numEmployees + 2), align='left', rwidth=0.9, density=True)
plt.xlabel("Number of Employees in Office on a Given Day")
plt.ylabel("Probability")
plt.title(f"Distribution of daily office attendance (from {numSimulations} weeks)")
plt.grid(axis='y', alpha=0.75)
plt.show()

#overlap_probability = twoEmpolyeeOverlap(numSimulations)
#print('Simulated probability of a two-day overlap is:', overlap_probability)