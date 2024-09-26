import numpy as np

def g(x):
    return 1+x+0.5*x**2

def improved_snake_optimization(F, Fb, population_size, max_iter, dim, solution_bound):
    # Initialization
    vec_flag = [1, -1]  # Indicates whether c2 and c3 should be positive or negative
    food_threshold = 0.25  # Threshold for having food
    temp_threshold = 0.6  # Threshold for temperature suitability for matingd
    c1 = 0.5  # Constant c1 used for calculating food quality
    model_threshold = 0.6  # Model threshold for switching between modes
    c2 = 0.5  # Constant c2 used for position updates
    c3 = 2  # Constant c3 used for combat and mating
    p = 0.3
    X = solution_bound[0] + np.random.random_sample((population_size, dim)) * (solution_bound[1] - solution_bound[0])
    X = np.matrix(X)
    fitness = [0 for i in range(population_size)]
    for i in range(population_size):
        fitness[i] = F(X[i])
    g_best = Fb(fitness)
    gy_best = fitness[g_best]
    food = X[g_best, :]
    # Divide the population into male and female parts
    male_number = int(np.round(population_size / 2))
    female_number = population_size - male_number
    male = X[0:male_number, :]
    female = X[male_number:population_size, :]
    male_individual_fitness = fitness[0:male_number]
    female_individual_fitness = fitness[male_number:population_size]
    male_fitness_best_index = Fb(male_individual_fitness)
    male_fitness_best_value = male_individual_fitness[male_fitness_best_index]
    male_best_fitness_solution = male[male_fitness_best_index, :]
    female_fitness_best_index = Fb(female_individual_fitness)
    female_fitness_best_value = female_individual_fitness[female_fitness_best_index]
    female_best_fitness_solution = female[male_fitness_best_index, :]
    new_male = np.matrix(np.zeros((male_number, dim)))
    new_female = np.matrix(np.zeros((female_number, dim)))
    gene_best_fitness = [0 for i in range(max_iter)]

    for t in range(max_iter):
        #print("Epoch:%s"%t)
        temp = g(-(t / max_iter))
        quantity = c1 * g((t - max_iter) / max_iter)
        if quantity > 1:
            quantity = 1
        if quantity < food_threshold:   # Exploring stage
            for i in range(male_number):
                for j in range(dim):
                    rand_leader_index = np.random.randint(0, male_number)
                    rand_male = male[rand_leader_index, :]
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    am = g(
                        -(male_individual_fitness[rand_leader_index] / (male_individual_fitness[i] + np.spacing(1))))
                   # In exploring stage, new individuals are generated around the current best solution.
                    if np.random.random() > 0.8:
                        new_male[i, j] = food[0, j] + flag * c2 * am * (
                                (solution_bound[1] - solution_bound[0]) * np.random.random() + solution_bound[0])
                    else:
                        new_male[i, j] = rand_male[0, j] + flag * c2 * am * (
                                (solution_bound[1] - solution_bound[0]) * np.random.random() + solution_bound[0])
            for i in range(female_number):
                for j in range(dim):
                    rand_leader_index = np.random.randint(0, female_number)
                    rand_female = female[rand_leader_index, :]
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    am = g(-(female_individual_fitness[rand_leader_index] / (
                            female_individual_fitness[i] + np.spacing(1))))
                    # In exploring stage, new individuals are generated around the current best solution.
                    if np.random.random() > 0.8:
                        new_female[i, j] = food[0, j] + flag * c2 * am * (
                                (solution_bound[1] - food[0,j]) * np.random.random() + (food[0, j]-solution_bound[0])* np.random.random())
                    else:
                        new_female[i, j] = rand_female[0, j] + flag * c2 * am * (
                                (solution_bound[1] - solution_bound[0]) * np.random.random() + solution_bound[0])
        else:  #Food finding stage
            if temp > temp_threshold:
                for i in range(male_number):
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    for j in range(dim):
                        new_male[i, j] = food[0, j] + flag * c3 * temp * np.random.random() * (food[0, j] - male[i, j])
                for i in range(female_number):
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    for j in range(dim):
                        new_female[i, j] = food[0, j] + flag * c3 * temp * np.random.random() * (food[0, j] - female[i, j])
            else:  #Fighting/Mating stage
                model = np.random.random()
                if model < model_threshold:
                    for i in range(male_number):
                        for j in range(dim):
                            fm = g(-female_fitness_best_value / (male_individual_fitness[i] + np.spacing(1)))
                            new_male[i, j] = male[i, j] + c3 * fm * np.random.random() * (
                                    quantity * male_best_fitness_solution[0, j] - male[i, j])
                    for i in range(female_number):
                        for j in range(dim):
                            ff = g(-male_fitness_best_value / (female_individual_fitness[i] + np.spacing(1)))
                            new_female[i, j] = female[i, j] + c3 * ff * np.random.random() * (
                                    quantity * female_best_fitness_solution[0, j] - female[i, j])
                else:
                    for i in range(male_number):
                        for j in range(dim):
                            mm = g(-female_individual_fitness[i] / (male_individual_fitness[i] + np.spacing(1)))
                            new_male[i, j] = male[i, j] + c3 * np.random.random() * mm * (
                                    quantity * female[i, j] - male[i, j])
                    for i in range(female_number):
                        for j in range(dim):
                            mf = g(-male_individual_fitness[i] / (female_individual_fitness[i] + np.spacing(1)))
                            new_female[i, j] = female[i, j] + c3 * np.random.random() * mf * (
                                    quantity * male[i, j] - female[i, j])
                    negative_or_positive = np.random.randint(0, 2)
                    egg = vec_flag[negative_or_positive]
                    if egg == 1:
                        male_best_fitness_index = np.argmax(male_individual_fitness)
                        sorted_indices = np.argsort(male_individual_fitness)
                        weeded_incices = sorted_indices[-1*int(p*len(male)):]
                        best_male_index = sorted_indices[0]
                        # New individuals are generated around the best solution.
                        for i in range(0,len(weeded_incices)):
                                if np.random.random()>0.8:
                                    new_male[i,:] = male[best_male_index,:] + np.random.random() * (
                                            solution_bound[1] - male[best_male_index,:]) - np.random.random() *(male[best_male_index,:] - solution_bound[0])
                                else:
                                    new_male[i,:] = solution_bound[0]+ np.random.random() * (
                                            solution_bound[1] - solution_bound[0])
                        sorted_indices = np.argsort(female_individual_fitness)
                        weeded_incices = sorted_indices[-1*int(p*len(female)):]
                        best_female_index = sorted_indices[0]
                        for i in range(0,len(weeded_incices)):
                                if np.random.random()>0.8:
                                    new_female[i,:] = female[best_female_index,:] + np.random.random() * (
                                            solution_bound[1] - female[best_female_index,:]) - np.random.random() *(female[best_female_index,:] - solution_bound[0])
                                else:
                                    new_female[i,:] = solution_bound[0] + np.random.random() * (
                                            solution_bound[1] - solution_bound[0])
        for j in range(male_number):
            flag_low = new_male[j, :] < solution_bound[0]
            flag_high = new_male[j, :] > solution_bound[1]
            new_male[j, :] = (np.multiply(new_male[j, :], ~(flag_low + flag_high))) + np.multiply(solution_bound[1], flag_high) +np.multiply(solution_bound[0], flag_low)
            y = F(new_male[j, :])
            if y < male_individual_fitness[j]:
                male_individual_fitness[j] = y
                male[j, :] = new_male[j, :]
        male_current_best_fitness_index = Fb(male_individual_fitness)
        male_current_best_fitness = male_individual_fitness[male_current_best_fitness_index]
        
        for j in range(female_number):
            flag_low = new_female[j, :] < solution_bound[0]
            flag_high = new_female[j, :] > solution_bound[1]
            new_female[j, :] = (np.multiply(new_female[j, :], ~(flag_low + flag_high))) + np.multiply(solution_bound[1], flag_high) +np.multiply(solution_bound[0], flag_low)
            y = F(new_female[j, :])
            if y < female_individual_fitness[j]:
                female_individual_fitness[j] = y
                female[j, :] = new_female[j, :]
        female_current_best_fitness_index = Fb(female_individual_fitness)
        female_current_best_fitness = male_individual_fitness[female_current_best_fitness_index]
        if male_current_best_fitness < male_fitness_best_value:
            male_best_fitness_solution = male[male_current_best_fitness_index, :]
            male_fitness_best_value = male_current_best_fitness
        if female_current_best_fitness < female_fitness_best_value:
            female_best_fitness_solution = female[female_current_best_fitness_index, :]
            female_fitness_best_value = female_current_best_fitness
        if male_current_best_fitness < female_current_best_fitness:
            gene_best_fitness[t] = male_current_best_fitness
        else:
            gene_best_fitness[t] = female_current_best_fitness
        if male_fitness_best_value < female_fitness_best_value:
            gy_best = male_fitness_best_value
            food = male_best_fitness_solution 
        else:
            gy_best = female_fitness_best_value
            food = female_best_fitness_solution
    return food, gy_best, gene_best_fitness
