import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from deap import algorithms, base, creator, tools

# Load stock price data
df_aapl = pd.read_csv("AAPL.csv")
df_amzn = pd.read_csv("AMZN.csv")
df_goog = pd.read_csv("GOOG.csv")
df_meta = pd.read_csv("META.csv")
df_ndaq = pd.read_csv("NDAQ.csv")

# Concatenate all dataframes into a single dataframe
df = pd.concat([df_aapl['Close'], df_amzn['Close'], df_goog['Close'], df_meta['Close'], df_ndaq['Close']], axis=1)
df.columns = ['AAPL', 'AMZN', 'GOOG', 'META', 'NDAQ']

# Preprocess data
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Define target variable and features
target = scaled_df['AAPL']
features = scaled_df.drop('AAPL', axis=1)

# Feature selection
selector = SelectKBest(score_func=f_regression, k=2)
selected_features = selector.fit_transform(features, target)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=42)

# Define evaluation function
def evaluate(individual):
    weights = np.array(individual)
    portfolio_return = np.dot(X_train, weights)
    mse = mean_squared_error(y_train, portfolio_return)
    return mse,

# Define genetic algorithm parameters
POPULATION_SIZE = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 10

# Create a fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Define attributes and population
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=selected_features.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create initial population
population = toolbox.population(n=POPULATION_SIZE)

# Create hall of fame to store best individuals
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Perform the evolution
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

# Perform the evolution
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS, stats=stats, halloffame=hall_of_fame,
                                          verbose=True)

# Get the best individual
best_individual = hall_of_fame[0]
best_weights = np.array(best_individual)

# Evaluate performance on test set
portfolio_return_test = np.dot(X_test, best_weights)
mse_test = mean_squared_error(y_test, portfolio_return_test)

# Print results
print("Best Weights:", best_weights)
print("MSE on Test Set:", mse_test)
