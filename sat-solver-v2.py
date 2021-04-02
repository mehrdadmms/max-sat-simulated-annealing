from random import randint, random, choice
import json
import numpy as np
import time

# this code has max_iterations and max_restart so if algorithm stucks in local optima it will try this much 
# and it takes about 40 seconds at worst on my computer which is under the 60 seconds limit 

class Solver:
    max_restart = 10 
    max_iterations = 50000
    t0 = 5
    tf = 0.001
    clauses = []
    best_satisfaction = 0
    best_satisfaction_set = None

    def __init__(self):
        pass

    def solve(self, filename):
        with open(filename, 'r') as file:
            all_lines = file.readlines()

        for i in range(0, len(all_lines)):
            line = all_lines[i].split()
            if i == 0:
                self.n_vars, self.n_clause = line 
                self.n_vars = int(self.n_vars)
                self.best_satisfaction_set = [0]*self.n_vars
                self.n_clause = int(self.n_clause)
                continue
            self.clauses.append([self.to_tuple(x) for x in line[:-1]])
        initial_solutions = [self.initial_solution() for _ in range(self.max_restart)]
        x = time.time()
        mean, std = self.simmulated_annealing(self.clauses, initial_solutions)
        print("SA: {} +- {}".format(mean, std))

        print('\n\n')
        print('no answer')
        print(str(self.best_satisfaction))
        print('best : ' + str(self.best_satisfaction) )
        # print(str(self.best_satisfaction_set))
        print('best set: ' + str(self.best_satisfaction_set))
        print('ran in ' + str(time.time() - x))
        
    @staticmethod
    def to_tuple(n_str):
        n = int(n_str)
        return (abs(n)-1, n >= 0)

    def next_temperature(self, i):
        return (self.t0 - self.tf)/(np.cosh(self.max_restart*i/self.max_iterations)) + self.tf

    def initial_solution(self):
        return [choice([True, False]) for _ in range(self.n_vars)]

    @staticmethod
    def evaluate(clause, solution):
        return any([solution[position] == value for position, value in clause])

    def evaluate_all(self, clauses, solution):
        evaluation = np.sum([self.evaluate(clause, solution) for clause in clauses])
        self.check_satisfaction(clauses, evaluation)
        return evaluation

    def check_satisfaction(self, clauses, evaluation):
        if self.best_satisfaction < evaluation:
            self.best_satisfaction = evaluation
            satisfaction_set = [0]*self.n_vars
            for clause in clauses:
                for c in clause:
                    index, bit = c
                    satisfaction_set[index] = 1 if bit else 0
            self.best_satisfaction_set = satisfaction_set
        if self.best_satisfaction == self.n_clause:
            print('\n\nsatisfied')
            print('best satisfaction count: ' + str(self.best_satisfaction))
            print('best satisfaction set: ' + ''.join([str(x) for x in self.best_satisfaction_set]))
            # print(str(self.best_satisfaction))
            # print(''.join([str(x) for x in self.best_satisfaction_set]))
            quit()

    @staticmethod
    def disturb(solution):
        return [1 - var if random() < 0.01 else var for var in solution]    

    def simmulated_annealing(self, clauses, initial_solutions):
        all_scores = []

        for i in range(self.max_restart):
            solution = initial_solutions[i]
            score = self.evaluate_all(clauses, solution)
            temperature = self.t0
            iterations = 0
            scores = []

            while iterations < self.max_iterations:
                new_solution = self.disturb(solution)
                new_score = self.evaluate_all(clauses, new_solution)
                delta = score - new_score  # Equivalent to E(new_solution) - E(solution)
                if delta <= 0 or random() < np.exp(-delta/temperature):
                    solution = new_solution
                    score = new_score
                iterations += 1
                scores.append(score)
                temperature = self.next_temperature(iterations)
            print(np.array(scores).max())
            all_scores.append(scores)

        all_scores = np.array(all_scores)
        best_scores = all_scores.max(axis=1)

        return best_scores.mean(), best_scores.std()





if __name__ == '__main__':
    # filename = './samples/Q1_20_80.txt'
    filename = './samples/Q1_20_90.txt'
    # filename = './samples/ez.txt'
    solver = Solver()
    solver.solve(filename)
