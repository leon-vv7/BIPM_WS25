import nbformat as nbf
import json

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # 1. Title & Objectives
    title_cell = nbf.v4.new_markdown_cell("""# Notebook 0: Python Data Science Foundations

## Objectives
By the end of this notebook, you will be able to:
1.  **Manipulate DataFrames**: Use pandas to load, inspect, and modify datasets.
2.  **Summarize Data**: Calculate summary statistics and aggregate data using `groupby`.
3.  **Visualize Data**: Create basic plots using `matplotlib` and understand the figure/axes structure.
4.  **Apply Best Practices**: Use method chaining and modern plotting libraries for cleaner code.

## Prerequisites
- Basic Python knowledge (variables, functions, lists).
- Familiarity with `pandas` and `numpy` basics.
""")

    # 2. Theory & Logic: Pandas Data Manipulation
    theory_pandas_cell = nbf.v4.new_markdown_cell("""## Part 1: Data Manipulation with Pandas

### Why Pandas?
Pandas is the backbone of data science in Python. It provides the `DataFrame` object, which allows for efficient manipulation of tabular data. Key features include:
-   **Vectorized operations**: Faster than loops.
-   **Label-based indexing**: Access data by column name or index label.
-   **Flexible data handling**: Handles missing data, different types, and time series effortlessly.

### Key Concepts: GroupBy
The `groupby` operation involves three steps:
1.  **Split**: Break the data into groups based on some criteria.
2.  **Apply**: Apply a function to each group independently (e.g., sum, mean, count).
3.  **Combine**: Combine the results into a data structure.

This "Split-Apply-Combine" strategy is essential for summarizing data across categories (e.g., average GDP per continent).
""")

    # 3. Professor's Implementation: Pandas
    prof_pandas_cell = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Load Gapminder data (simulated for this notebook if file not present, or use library)
try:
    from gapminder import gapminder
except ImportError:
    # Fallback if gapminder is not installed
    print("Gapminder not found, creating sample data...")
    data = {
        'country': ['Afghanistan', 'Brazil', 'China', 'Germany', 'United States'] * 3,
        'continent': ['Asia', 'Americas', 'Asia', 'Europe', 'Americas'] * 3,
        'year': [1952]*5 + [1977]*5 + [2007]*5,
        'lifeExp': [28.8, 50.9, 44.0, 67.5, 68.4, 38.4, 61.5, 63.9, 72.5, 73.3, 43.8, 72.4, 72.9, 79.4, 78.2],
        'pop': [8425333, 56602560, 556263527, 69145952, 157553000] * 3,
        'gdpPercap': [779, 2108, 400, 7144, 13990] * 3
    }
    gapminder = pd.DataFrame(data)

# Basic Inspection
print("First 5 rows:")
print(gapminder.head())

# GroupBy Example: Average GDP per continent
print("\\nAverage GDP per continent:")
byContinent = gapminder.groupby("continent")
print(byContinent["gdpPercap"].mean())

# Multiple Aggregations
print("\\nLife Expectancy Stats by Continent:")
print(byContinent["lifeExp"].agg([np.min, np.max, np.mean]))
""")

    # 4. Guided Example: Pandas
    guided_pandas_cell = nbf.v4.new_markdown_cell("""### Guided Example: Analyzing Titanic Data

Let's apply these concepts to the Titanic dataset. We want to understand survival rates by class and gender.

**Step 1: Load Data**
We'll use seaborn to load the dataset.

**Step 2: Grouping**
We will group by `pclass` and `sex`.

**Step 3: Aggregation**
We will calculate the mean of the `survived` column (which gives the survival rate).
""")

    guided_pandas_code_cell = nbf.v4.new_code_cell("""import seaborn as sns

# Load dataset
titanic = sns.load_dataset('titanic')

# Group by Class and Sex
by_class_sex = titanic.groupby(['pclass', 'sex'])

# Calculate survival rate
survival_rates = by_class_sex['survived'].mean()
print("Survival Rates by Class and Sex:")
print(survival_rates)

# Unstack for better readability
print("\\nFormatted Table:")
print(survival_rates.unstack())
""")

    # 5. Theory & Logic: Visualization
    theory_viz_cell = nbf.v4.new_markdown_cell("""## Part 2: Visualization with Matplotlib

### The Figure and Axes
Matplotlib plots are built on a hierarchy:
1.  **Figure**: The overall window or page that everything is drawn on.
2.  **Axes**: The area where data is plotted (what we think of as "a plot"). A Figure can contain multiple Axes.

### Subplots
Creating a grid of subplots allows you to compare different views of data side-by-side. The `plt.subplots()` function is the most convenient way to create a Figure and a set of Axes at once.
""")

    # 6. Professor's Implementation: Subplots
    prof_viz_cell = nbf.v4.new_code_cell("""import matplotlib.pyplot as plt

# Create a grid of 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Generate random data
ran_nums = np.random.standard_normal(400)
ran_walk = ran_nums.cumsum()

# Plot 1: Histogram
ax1.hist(ran_nums, bins=20, color="blue", alpha=0.3)
ax1.set_title("Normal Distribution")
ax1.set_ylabel("Frequency")

# Plot 2: Random Walk
ax2.plot(ran_walk, color="black", linestyle="dashed")
ax2.set_title("Random Walk")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Value")

# Adjust layout to prevent overlap
fig.tight_layout()
plt.show()
""")

    # 7. Exercise
    exercise_cell = nbf.v4.new_markdown_cell("""## Exercises

### Task 1: Gapminder Analysis
Using the `gapminder` dataset:
1.  Filter the data for the year 2007.
2.  Group the data by `continent`.
3.  Calculate the total population (`pop`) for each continent.
4.  Create a bar chart showing the total population per continent.

### Task 2: Titanic Visualization
Using the `titanic` dataset:
1.  Create a figure with 1 row and 2 columns.
2.  In the first subplot, create a histogram of passenger `age`.
3.  In the second subplot, create a bar chart of `survived` counts (how many died vs survived).
    *   *Hint*: You can use `titanic['survived'].value_counts()` to get the data for the bar chart.
""")

    # 8. Solutions (Hidden/Collapsed)
    solution_cell = nbf.v4.new_code_cell("""# --- SOLUTION TASK 1 ---
gapminder_2007 = gapminder[gapminder['year'] == 2007]
pop_by_continent = gapminder_2007.groupby('continent')['pop'].sum()

plt.figure(figsize=(8, 4))
pop_by_continent.plot(kind='bar', color='skyblue')
plt.title("Total Population by Continent (2007)")
plt.ylabel("Population")
plt.show()

# --- SOLUTION TASK 2 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of Age
ax1.hist(titanic['age'].dropna(), bins=20, color='green', alpha=0.7)
ax1.set_title("Passenger Age Distribution")
ax1.set_xlabel("Age")

# Bar chart of Survival
survival_counts = titanic['survived'].value_counts()
ax2.bar(survival_counts.index.astype(str), survival_counts.values, color=['red', 'blue'])
ax2.set_title("Survival Counts (0=Died, 1=Survived)")
ax2.set_xlabel("Outcome")

plt.tight_layout()
plt.show()
""")
    solution_cell.metadata = {"collapsed": True, "jupyter": {"source_hidden": True}}


    # 9. Industry Best Practices
    best_practices_cell = nbf.v4.new_markdown_cell("""## Industry Best Practices

### Modern Pandas: Method Chaining
Instead of saving intermediate variables, fluent code often uses "method chaining".
```python
# Traditional
df_2007 = gapminder[gapminder['year'] == 2007]
grouped = df_2007.groupby('continent')
result = grouped['pop'].sum()

# Method Chaining
result = (
    gapminder
    .query("year == 2007")
    .groupby("continent")["pop"]
    .sum()
)
```

### Visualization Libraries
While `matplotlib` is the foundation, modern libraries offer higher-level abstractions:
*   **Seaborn**: Built on matplotlib, provides beautiful default styles and complex statistical plots (e.g., `sns.boxplot`, `sns.heatmap`) with less code.
*   **Plotly**: Creates interactive, web-ready plots where users can zoom and hover to see data values.

### Tidy Data
Ensure your data is "tidy":
1.  Each variable forms a column.
2.  Each observation forms a row.
3.  Each type of observational unit forms a table.
Pandas is optimized for tidy data. If your data is "wide" (e.g., years as columns), use `pd.melt()` to reshape it.
""")

    # Add cells to notebook
    nb.cells = [
        title_cell,
        theory_pandas_cell,
        prof_pandas_cell,
        guided_pandas_cell,
        guided_pandas_code_cell,
        theory_viz_cell,
        prof_viz_cell,
        exercise_cell,
        solution_cell,
        best_practices_cell
    ]

    # Save notebook
    with open('/Users/robertgreene/Code/Python/BIPM_WS25/Lectures-main/Notebook_0_Foundations.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
