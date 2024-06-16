import pygad.kerasga
from sklearn.preprocessing import normalize
import pygad
import time
import numpy as np
from Parameters import Parameters
from NeuralNetwork import NeuralNetwork
from Bot import Bot
from Data_preparer import Data_preparer
import subprocess
import matplotlib.pylab as plt


class Learning:

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Learning, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load_instance = Parameters.load_instance
        self.name_instance = Parameters.name_instance
        self.parameters_GA = Parameters.parameters_GA
        self.bot = Bot()
        self.NN = NeuralNetwork()
        self.data_preparer = Data_preparer()
        self.best_balance = 0
        self.best_avr_balance = 0
        self.generations = 0
        self.fitness_cache = {}

    def fill_signals_dll(self, number_of_piece, parameters):
        # individual for our strategy
        set_count = len(parameters)

        data = self.data_preparer.get_piece(number_of_piece)
        data_2, data_3, data_15 = self.data_preparer.get_piece_for_frames(number_of_piece)
        fist_candles = self.data_preparer.get_fist_candles(number_of_piece)

        # Сохраняем
        np.savetxt('buffer/data.txt', data[:, :6])
        np.savetxt('buffer/data_2.txt', data_2)
        np.savetxt('buffer/data_3.txt', data_3)
        np.savetxt('buffer/data_15.txt', data_15)
        np.savetxt('buffer/parameters.txt', parameters)

        # Запускаем внешний exe-файл и передаем ему аргументы
        result = subprocess.run(["engine.exe",
                                 str(self.bot.get_frames()[0]),
                                 str(self.bot.get_frames()[1]),
                                 str(self.bot.get_frames()[2]),
                                 str(self.bot.get_frames()[3]),
                                 str(fist_candles[0]),
                                 str(fist_candles[1]),
                                 str(fist_candles[2]),
                                 str(self.data_preparer.size_of_piece())
                                 ], capture_output=True)

        # Открываем бинарный файл
        with open('buffer/signals.bin', 'rb') as f:
            # Считываем данные
            data = f.read()
            # Преобразуем данные в матрицу numpy
            signals = np.frombuffer(data, dtype=np.float64).reshape(set_count, 13)

        return signals

    def calcDecisions(self, number_of_piece):
        init_data = self.data_preparer.get_piece_of_training_set_open(number_of_piece)
        parameters_buy, parameters_sell = self.NN.perdict_definition(normalize(init_data, axis=1))

        signals_buy = self.fill_signals_dll(number_of_piece, parameters_buy)
        signals_sell = self.fill_signals_dll(number_of_piece, parameters_sell)

        decisions_buy = self.NN.perdict_decisions_buy(signals_buy)
        decisions_sell = self.NN.perdict_decisions_sell(signals_sell)

        return decisions_buy, decisions_sell

    def trade(self, piece):

        decisions_buy, decisions_sell = self.calcDecisions(piece)

        # simulation trade
        balanceB = 1.000
        balanceA = 0.000
        x = 1  # сколько B крутить
        up = 0
        balance_old = balanceB
        count = 0

        for i in range(Parameters.NUMBER_OF_CANDLES - 1, self.data_preparer.size_of_piece()):
            if up == 0:  # если не апнуто
                if decisions_buy[i - (Parameters.NUMBER_OF_CANDLES - 1)] > 0.9:
                    balance_old = balanceB
                    balanceA = x / self.data_preparer.get_piece(piece)[i, 1]
                    balanceB -= x * (1 + 0.00075)
                    up = 1
                    count += 1
                    continue

            if up == 1:  # если апнуто
                if decisions_sell[i - (Parameters.NUMBER_OF_CANDLES - 1)] > 0.9:
                    balanceB += balanceA * self.data_preparer.get_piece(piece)[i, 1] * (1 - 0.00075)
                    balanceA = 0
                    up = 0
        if up == 1:
            balanceB = balance_old
            if count == 1: balanceB = -1
        if balanceB < 1: balanceB = -100

        return balanceB

    def fitness_func(self, solution, sol_idx):

        if sol_idx in self.fitness_cache:
            return self.fitness_cache[sol_idx]

        self.NN.fill_weights(solution)

        balances = np.zeros(Parameters.number_of_pieces)
        for piece in range(0, Parameters.number_of_pieces):
            print("Generations " + str(self.generations) + "/" + str(Parameters.num_generations) + ", " +
                  "Bot " + str(sol_idx+1) + "/" + str(Parameters.num_bot) + ", " +
                  "Piece " + str(piece+1) + "/" + str(Parameters.number_of_pieces))
            balances[piece] = self.trade(piece)
        avg_balance = np.average(balances)

        if avg_balance > self.best_avr_balance:
            self.best_avr_balance = avg_balance
            self.best_balance = max(balances)

        self.fitness_cache[sol_idx] = avg_balance

        return avg_balance

    def evolution(self):

        def on_generation(ga_instance):
            self.generations = ga_instance.generations_completed
            print("Generation = {generation}".format(generation=self.generations))
            print("Best average balance = {fitness}".format(fitness=self.best_avr_balance))
            # Returning the details of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            # сохраняем промежуточный ген
            np.savetxt(Parameters.bot_name + '_temp_result.gen', solution)
            self.fitness_cache = {}

        self.bot.set_params_for_NN()
        self.bot.read_frames_from_xml()

        self.NN.create_networks(self.bot)

        self.data_preparer.set_data()
        self.data_preparer.create_training_set()
        self.data_preparer.set_data_for_frames(self.bot)

        time_start_evolution = time.time()

        if self.load_instance == 1:
            ga_instance = pygad.load(filename=self.name_instance)
        else:
            ga_instance = pygad.GA(num_generations=self.parameters_GA['num_generations'],
                                   num_parents_mating=self.parameters_GA['num_parents_mating'],
                                   initial_population=self.NN.init_population_builder(self.parameters_GA['num_bot']),
                                   fitness_func=lambda x, y: self.fitness_func(x, y),
                                   parent_selection_type=self.parameters_GA['parent_selection_type'],
                                   crossover_type=self.parameters_GA['crossover_type'],
                                   mutation_type=self.parameters_GA['mutation_type'],
                                   mutation_percent_genes=self.parameters_GA['mutation_percent_genes'],
                                   keep_parents=self.parameters_GA['keep_parents'],
                                   on_generation=on_generation)

        ga_instance.run()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        # сохраняем лучший ген
        np.savetxt(Parameters.bot_name + '_best_result.gen', solution)

        print("time " + str(np.round((time.time() - time_start_evolution), 3)) + " sec")

        print("Средний баланс лучшего бота " + str(self.best_avr_balance))
        print("Лучший баланс на одном промежутке " + str(self.best_balance))

        ga_instance.plot_fitness()

    def trade_for_use(self, piece):

        decisions_buy, decisions_sell = self.calcDecisions(piece)

        # simulation trade
        balanceB = 1.000
        balanceA = 0.000
        x = 1  # сколько B крутить
        up = 0
        balance_old = balanceB
        count = 0

        pooint_x_buy, pooint_y_buy = [], []
        pooint_x_sell, pooint_y_sell = [], []

        for i in range(Parameters.NUMBER_OF_CANDLES, self.data_preparer.size_of_piece()):
            if up == 0:  # если не апнуто
                if decisions_buy[i - (Parameters.NUMBER_OF_CANDLES - 1)] > 0.9:
                    balance_old = balanceB
                    balanceA = x / self.data_preparer.get_piece(piece)[i, 1]
                    balanceB -= x * (1 + 0.00075)
                    pooint_x_buy.append(i)
                    pooint_y_buy.append(self.data_preparer.get_piece(piece)[i, 1])
                    up = 1
                    count += 1
                    continue

            if up == 1:  # если апнуто
                if decisions_sell[i - (Parameters.NUMBER_OF_CANDLES - 1)] > 0.9:
                    balanceB += balanceA * self.data_preparer.get_piece(piece)[i, 1] * (1 - 0.00075)
                    balanceA = 0
                    pooint_x_sell.append(i)
                    pooint_y_sell.append(self.data_preparer.get_piece(piece)[i, 1])
                    up = 0

        if up == 1:
            balanceB = balance_old

        return balanceB, pooint_x_buy, pooint_y_buy, pooint_x_sell, pooint_y_sell

    def use(self):

        self.bot.set_params_for_NN()
        self.bot.read_frames_from_xml()

        self.NN.create_networks(self.bot)

        self.data_preparer.set_data()
        self.data_preparer.create_training_set()
        self.data_preparer.set_data_for_frames(self.bot)

        weights = np.loadtxt(Parameters.name_best_use)
        self.NN.fill_weights(weights)

        balanceB, pooint_x_buy, pooint_y_buy, pooint_x_sell, pooint_y_sell= self.trade_for_use(Parameters.number_of_piece)

        print('Баланс: ' + str(balanceB))

        if Parameters.build_plot == 1:
            mid = (self.data_preparer.get_piece(Parameters.number_of_piece)[:, 2] +
                   self.data_preparer.get_piece(Parameters.number_of_piece)[:, 3]) / 2

            fig = plt.figure()
            ax = plt.subplot()

            ax.plot(mid, linestyle='-', color="black", label='average')
            ax.plot(pooint_x_buy, pooint_y_buy, linestyle='', marker='^', color="green", label='buy', markersize=12)
            ax.plot(pooint_x_sell, pooint_y_sell, linestyle='', marker='v', color="red", label='sell', markersize=12)
            ax.legend()
            ax.grid(True, linestyle='-', color='0.75')
            plt.show()

