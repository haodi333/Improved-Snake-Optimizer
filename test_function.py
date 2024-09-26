from SO import snake_optimization
from ISO import improved_snake_optimization
import numpy as np
import matplotlib.pyplot as plt

def F(x):
    print(x)
    sum_total=0
    x=np.array(x)
    for i in range(0,len(x[0])):
        sum_total+=x[0][i]**2-10*np.cos(2*np.pi*x[0][i])+10
    return sum_total

def sphere_model(x):
    x = np.array(x)[0]
    return np.sum(x**2)

def generalized_rastrigins_function(x, A=10):
    x = np.array(x)[0]
    n = len(x)
    sum_term = sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])
    return A * n + sum_term

def six_hump_camel_back(X):
    x = np.array(X)[0][0]
    y = np.array(X)[0][1]
    term1 = (4 - 2.1 * x ** 2 + (x ** 4) / 3) * (x ** 2)
    term2 = x * y
    term3 = (-4 + 4 * y ** 2) * (y ** 2)
    return term1 + term2 + term3

def ackleys_function(x):
    x = np.array(x)[0]
    dimension = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / dimension)) - np.exp(sum2 / dimension) + 20 + np.exp(1)

def generalized_griewank_function(x, a=1, k=2):
    x = np.array(x)[0]
    dimension = len(x)
    term1 = 1 / a * np.sum(x**2)
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, dimension + 1))))
    return term1 - term2 + a

def quartic_function_with_noise(x):
    x = np.array(x)[0]
    a, b, c, d, e, noise_std_dev = 1, -2, 1.5, 3, 0, 0.1
    noise = np.random.normal(0, noise_std_dev, x.shape)
    return np.abs(np.sum(a * x**4 + b * x**3 + c * x**2 + d * x + e + noise))

def Fb(x):
    return np.argmin(x)

dim = 30
max_iter = 100
population_size = 50
solution_bound = [-100,100]
ID = 0

func_list = {0:sphere_model, 1:generalized_rastrigins_function, 2:six_hump_camel_back, 3: ackleys_function, 4:generalized_griewank_function, 5:quartic_function_with_noise}
theoretical_best_value_list = {sphere_model:0, generalized_rastrigins_function:0, six_hump_camel_back:-1.0316285, ackleys_function:0, generalized_griewank_function:0, quartic_function_with_noise:0}
func = func_list[ID]

theoretical_best_value = theoretical_best_value_list[func]
food, global_fitness, gene_best_fitness= snake_optimization(func, Fb, population_size, max_iter, dim, solution_bound)
gene_best_fitness=np.array(gene_best_fitness)
print("----SO----")
print("Best solution: ", food)
print("Best fitness：", global_fitness)
plt.plot(range(len(gene_best_fitness)),gene_best_fitness,label='Best fitness (SO)')

food, global_fitness, gene_best_fitness= improved_snake_optimization(func, Fb, population_size, max_iter, dim, solution_bound)
gene_best_fitness=np.array(gene_best_fitness)
print("----ISO----")
print("Best solution: ", food)
print("Best fitness：", global_fitness)
plt.plot(range(len(gene_best_fitness)),gene_best_fitness,label='Best fitness (ISO)')

plt.plot(range(len(gene_best_fitness)),[theoretical_best_value,]*len(gene_best_fitness),label='Theoretical best value')

plt.legend()
plt.title('Fitness curve on ' + func.__name__)