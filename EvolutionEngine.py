# ------------------ IMPORTS ------------------
import pygame
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import cv2
from datetime import datetime
import os
from Colors import Color
from Car import Car
import numpy as np
from utils import *

import statistics

from copy import deepcopy

import random


from Individual import Individual
from Computation import Computation
from Crossover import Crossover
from Mutation import Mutation
from Selection import Selection
from DNA import DNA









class EvolutionEngine:

    def __init__(self, _MAX_GENERATIONS,_POPULATION_SIZE, _ELITE_FRACTION, _DNA_INITIALIZATION, _DNA_DECODER, _COMPUTATION, _SELECTION, _CROSSOVER, _MUTATION):


        # some execution params, arent in research, can be constant
        self.FPS = 60000


        self.READ_FROM_FILE = False

        self.USE_MAP = True
        self.USE_VAL_MAP = True
        self.LOAD_POS = True
        self.LOAD_VAL_POS = True
        self.COLLISION_SURFACE_COLOR = Color.GREEN

        self.LOAD_MODEL = False

        self.DRAW_SIMULATION = False

        # get evolution params from main
        self.MAX_GENERATIONS = _MAX_GENERATIONS
        self.POPULATION_SIZE = _POPULATION_SIZE
        self.ELITE_FRACTION = _ELITE_FRACTION

        self.DNA_init = _DNA_INITIALIZATION
        self.DNA_decode = _DNA_DECODER
        self.compute = _COMPUTATION
        self.selection = _SELECTION
        self.crossover = _CROSSOVER
        self.mutate = _MUTATION



        # some iniitialization of variables
        pygame.init()

        info = pygame.display.Info()

        self.WINDOW_WIDTH = int(info.current_w * 0.9)
        self.WINDOW_HEIGHT = int(info.current_h * 0.9)

        self.CAR_X = 0
        self.CAR_Y = 0
        self.CAR_A = 0

        self.CAR_VAL_X = 0
        self.CAR_VAL_Y = 0
        self.CAR_VAL_A = 0

        self.elite_amount = int(self.POPULATION_SIZE * self.ELITE_FRACTION)

        self.TIME = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

        self.output_path = "evolutionOutput/" + self.TIME

        os.makedirs(self.output_path, exist_ok=True)

        self.WINDOW_MID_POINT = (self.WINDOW_WIDTH / 2, self.WINDOW_HEIGHT / 2)

        self.population: list[Individual] = []
        self.new_population : list[Individual]= []

        self.generationsList = [0]
        self.bestFitnessList = [0]
        self.medianFitnessList = [0]
        self.testFitnessList = [0]

        self.fig, self.ax = plt.subplots(1, 1)

        self.TITLE = "Ewolucja pojazdów"
        pygame.display.set_caption(self.TITLE)


        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.screen.fill(self.COLLISION_SURFACE_COLOR)

        self.track = None

        self.STARTING_POINT: list[float] = []
        self.STARTING_ANGLE = None

        self.valTrack = None

        self.STARTING_VAL_POINT: list[float] = []
        self.STARTING_VAL_ANGLE = None

        self.TIMING_CLOCK = pygame.time.Clock()

        self.generationNumber = 1

        self.CAR_WIDTH = self.WINDOW_WIDTH * 0.01
        self.CAR_HEIGHT = self.CAR_WIDTH * 0.6

        self.collision_map = None
        self.collision_val_map = None

        self.best_indv_val_fitness = 0




    def drawTrack(self):
        drawInstr = "Lewy przycisk myszy by rysować, prawy by usuwać, kółkiem myszy steruje się grubością pędzla, spacja by przejść dalej."
        pygame.display.set_caption(self.TITLE + "     " + drawInstr)

        if self.USE_MAP:
            self.track = pygame.image.load("assets/track.png")
            self.track = pygame.transform.scale(self.track, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            self.screen.blit(self.track, (0, 0))
            pygame.image.save(self.track, self.output_path + "/track.png")
        else:
            running = True
            drawing = False
            erasing = False
            brushSize = 50
            self.track = self.screen.copy()
            while running:

                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        exit()

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            drawing = True
                        elif event.button == 3:
                            erasing = True
                        elif event.button == 4:
                            if brushSize < 1000:
                                brushSize += 1
                        elif event.button == 5:
                            if brushSize > 2:
                                brushSize -= 1

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            drawing = False
                        elif event.button == 3:
                            erasing = False

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            pygame.image.save(self.track, self.output_path + "/track.png")
                            running = False

                if drawing:
                    pygame.draw.circle(self.track, Color.BLACK, pygame.mouse.get_pos(), brushSize)
                if erasing:
                    pygame.draw.circle(self.track, self.COLLISION_SURFACE_COLOR, pygame.mouse.get_pos(), brushSize)

                pygame.draw.circle(self.track, Color.RED, self.WINDOW_MID_POINT, 2)

                self.screen.blit(self.track, (0, 0))

                if running:
                    pygame.draw.circle(self.screen, Color.BLACK, pygame.mouse.get_pos(), brushSize, 1)

                pygame.display.update()
                self.TIMING_CLOCK.tick(250)

        track_pixels = pygame.surfarray.pixels3d(self.track)
        self.collision_map = np.all(track_pixels == self.COLLISION_SURFACE_COLOR, axis=2)
        del track_pixels

    def drawValTrack(self):
        drawInstr = "Lewy przycisk myszy by rysować, prawy by usuwać, kółkiem myszy steruje się grubością pędzla, spacja by przejść dalej."
        pygame.display.set_caption(self.TITLE + "     " + drawInstr)

        if self.USE_VAL_MAP:
            self.valTrack = pygame.image.load("assets/val_track.png")
            self.valTrack = pygame.transform.scale(self.valTrack, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.image.save(self.valTrack, self.output_path + "/val_track.png")
        else:
            running = True
            drawing = False
            erasing = False
            brushSize = 50
            self.valTrack = self.screen.copy()
            self.valTrack.fill(self.COLLISION_SURFACE_COLOR)
            while running:

                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        exit()

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            drawing = True
                        elif event.button == 3:
                            erasing = True
                        elif event.button == 4:
                            if brushSize < 1000:
                                brushSize += 1
                        elif event.button == 5:
                            if brushSize > 2:
                                brushSize -= 1

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            drawing = False
                        elif event.button == 3:
                            erasing = False

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            pygame.image.save(self.valTrack, self.output_path + "/val_track.png")
                            running = False

                if drawing:
                    pygame.draw.circle(self.valTrack, Color.BLACK, pygame.mouse.get_pos(), brushSize)
                if erasing:
                    pygame.draw.circle(self.valTrack, self.COLLISION_SURFACE_COLOR, pygame.mouse.get_pos(), brushSize)

                pygame.draw.circle(self.valTrack, Color.RED, self.WINDOW_MID_POINT, 2)

                self.screen.blit(self.valTrack, (0, 0))

                if running:
                    pygame.draw.circle(self.screen, Color.BLACK, pygame.mouse.get_pos(), brushSize, 1)

                pygame.display.update()
                self.TIMING_CLOCK.tick(250)

        val_track_pixels = pygame.surfarray.pixels3d(self.valTrack)
        self.collision_val_map = np.all(val_track_pixels == self.COLLISION_SURFACE_COLOR, axis=2)
        del val_track_pixels

    def placeCar(self):

        carInstr = "Kursorem myszy ustaw pojazd, lewy przycisk myszy by położyć pojazd, lewa i prawa strzałka by obrócić pojazd."
        pygame.display.set_caption(self.TITLE + "     " + carInstr)


        if self.LOAD_POS:
            self.loadPos()
            self.STARTING_POINT = [self.CAR_X, self.CAR_Y]
            self.STARTING_ANGLE = self.CAR_A
        else:

            baseSprite = pygame.image.load("assets/car_sprite.png").convert_alpha()
            baseSprite = pygame.transform.scale(baseSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))
            drawSprite = None

            carCenterX = 0
            carCenterY = 0

            carAngle = 0

            rotatingRight = False
            rotatingLeft = False

            rotationSpeed = 0.8

            # car = Car(self.CAR_WIDTH, self.CAR_HEIGHT, self.MINIMUM_SPEED, self.TURN_SPEED, self.ACCELERATION,
            #     self.COLLISION_SURFACE_COLOR, self.DRAW_SENSORS, self.SENSORS_DRAW_DISTANCE, self.DATA_MODEL,
            #           [pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]], self.STARTING_ANGLE, self.PATH_TO_FOLDER + "assets/",
            #     self.track, self.SENSOR_ANGLE, self.USE_CROSSOVER, None)

            running = True
            while running:

                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        exit()

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            rotatingLeft = True
                        elif event.key == pygame.K_RIGHT:
                            rotatingRight = True

                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT:
                            rotatingLeft = False
                        elif event.key == pygame.K_RIGHT:
                            rotatingRight = False

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            running = False

                self.screen.blit(self.track, (0, 0))

                if rotatingLeft:
                    carAngle += rotationSpeed

                if rotatingRight:
                    carAngle -= rotationSpeed

                carAngle = (carAngle + 360) % 360

                carCenterX: float = pygame.mouse.get_pos()[0]
                carCenterY: float = pygame.mouse.get_pos()[1]
                drawSprite = pygame.transform.rotate(baseSprite, carAngle)
                self.screen.blit(drawSprite,(carCenterX - drawSprite.get_width() / 2, carCenterY - drawSprite.get_height() / 2))

                pygame.display.update()
                self.TIMING_CLOCK.tick(self.FPS)



                # car.position = [carCenterX, carCenterY]
                # car.updateFitness(0)
                # print(car.fitness)


            self.STARTING_POINT = [carCenterX, carCenterY]
            self.STARTING_ANGLE = carAngle

    def placeValCar(self):

        carInstr = "Kursorem myszy ustaw pojazd, lewy przycisk myszy by położyć pojazd, lewa i prawa strzałka by obrócić pojazd."
        pygame.display.set_caption(self.TITLE + "     " + carInstr)

        if self.LOAD_VAL_POS:
            self.loadValPos()
            self.STARTING_VAL_POINT = [self.CAR_VAL_X, self.CAR_VAL_Y]
            self.STARTING_VAL_ANGLE = self.CAR_VAL_A
        else:

            baseSprite = pygame.image.load("assets/car_sprite.png").convert_alpha()
            baseSprite = pygame.transform.scale(baseSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))
            drawSprite = None

            carCenterX = 0
            carCenterY = 0

            carAngle = 0

            rotatingRight = False
            rotatingLeft = False

            rotationSpeed = 0.8

            # car = Car(self.CAR_WIDTH, self.CAR_HEIGHT, self.MINIMUM_SPEED, self.TURN_SPEED, self.ACCELERATION,
            #     self.COLLISION_SURFACE_COLOR, self.DRAW_SENSORS, self.SENSORS_DRAW_DISTANCE, self.DATA_MODEL,
            #           [pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]], self.STARTING_ANGLE, self.PATH_TO_FOLDER + "assets/",
            #     self.track, self.SENSOR_ANGLE, self.USE_CROSSOVER, None)

            running = True
            while running:

                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        exit()

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            rotatingLeft = True
                        elif event.key == pygame.K_RIGHT:
                            rotatingRight = True

                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT:
                            rotatingLeft = False
                        elif event.key == pygame.K_RIGHT:
                            rotatingRight = False

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            running = False

                self.screen.blit(self.valTrack, (0, 0))

                if rotatingLeft:
                    carAngle += rotationSpeed

                if rotatingRight:
                    carAngle -= rotationSpeed

                carAngle = (carAngle + 360) % 360

                carCenterX: float = pygame.mouse.get_pos()[0]
                carCenterY: float = pygame.mouse.get_pos()[1]
                drawSprite = pygame.transform.rotate(baseSprite, carAngle)
                self.screen.blit(drawSprite, (carCenterX - drawSprite.get_width() / 2, carCenterY - drawSprite.get_height() / 2))

                pygame.display.update()
                self.TIMING_CLOCK.tick(self.FPS)

                # car.position = [carCenterX, carCenterY]
                # car.updateFitness(0)
                # print(car.fitness)

            self.STARTING_VAL_POINT = [carCenterX, carCenterY]
            self.STARTING_VAL_ANGLE = carAngle

    def createPopulation(self):
        self.new_population.clear()
        for _ in range(self.POPULATION_SIZE):
            indv = Individual()
            indv.dnaType = self.DNA_init()
            self.new_population.append(indv)
        if self.LOAD_MODEL:
            fullDataPath = "assets/bestDNA.txt"
            with open(fullDataPath, 'r') as file:
                for line in file:
                    line = line.strip()
                    dna = []
                    for char in line:
                        dna.append(int(char))
                    self.new_population[0].dnaType.DNA = dna



    def dekodeDNAToNetwork(self):
        for i in range(self.POPULATION_SIZE):
            self.population[i].nn = self.DNA_decode(self.population[i].dnaType)

    def evaluate_fitness(self):
        CarPopulation = []
        for i in range(0, self.POPULATION_SIZE):
            CarPopulation.append(Car(self.STARTING_POINT, self.STARTING_ANGLE, self.track, self.population[i].nn, self.compute, self.collision_map, (self.CAR_WIDTH, self.CAR_HEIGHT)))
        runningGeneration = True
        remainingCars = self.POPULATION_SIZE
        timer = 0
        bestFitness = 0
        while runningGeneration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            if self.DRAW_SIMULATION:
                self.screen.blit(self.track, (0, 0))
            for i in range(0, self.POPULATION_SIZE):
                if not self.population[i].new and CarPopulation[i].alive:
                    CarPopulation[i].alive = False
                    CarPopulation[i].fitness = self.population[i].fitness
                    remainingCars -= 1
                if CarPopulation[i].alive:
                    CarPopulation[i].update(timer)
                    if not CarPopulation[i].alive:
                        remainingCars -= 1
                    if CarPopulation[i].fitness > bestFitness:
                        bestFitness = CarPopulation[i].fitness
                        bestIndex = i
                if self.DRAW_SIMULATION:
                    CarPopulation[i].draw(self.screen)
            # self.drawNetwork(self.population[bestIndex].nn)
            if remainingCars == 0:
                break
            if timer % 5 == 0:
                t = "Generacja: " + str(self.generationNumber)
                t2 = "Żywych: " + str(remainingCars)
                t3 = "Czas: " + str(round(timer / 120, 2)) + "s"
                t4 = "Obecny najlepszy wynik generacji: " + str(round(bestFitness, 2))
                pygame.display.set_caption(self.TITLE + " - " + t + " - " + t2 + " - " + t3 + " - " + t4)
            if self.DRAW_SIMULATION:
                pygame.display.update()
            self.TIMING_CLOCK.tick(self.FPS)
            timer += 1

        for i in range(0, self.POPULATION_SIZE):
            self.population[i].fitness = CarPopulation[i].fitness

    def validateBest(self):

        best_individual = Car(self.STARTING_VAL_POINT, self.STARTING_VAL_ANGLE, self.valTrack, self.population[0].nn, self.compute, self.collision_val_map, (self.CAR_WIDTH, self.CAR_HEIGHT))
        runningGeneration = True
        timer = 0
        bestFitness = 0

        if not self.population[0].new:
            best_individual.fitness = self.population[0].testFitness
        else:
            while runningGeneration:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                if self.DRAW_SIMULATION:
                    self.screen.blit(self.valTrack, (0, 0))

                best_individual.update(timer)
                if self.DRAW_SIMULATION:
                    best_individual.draw(self.screen)
                bestFitness = best_individual.fitness
                self.population[0].testFitness = best_individual.fitness
                # self.drawNetwork(self.population[bestIndex].nn)
                if not best_individual.alive:
                    break
                if timer % 5 == 0:
                    t = "Generacja: " + str(self.generationNumber)
                    t2 = "Żywych: " + str(1)
                    t3 = "Czas: " + str(round(timer / 120, 2)) + "s"
                    t4 = "Obecny najlepszy wynik generacji: " + str(round(bestFitness, 2))
                    pygame.display.set_caption(self.TITLE + " - " + t + " - " + t2 + " - " + t3 + " - " + t4)
                if self.DRAW_SIMULATION:
                    pygame.display.update()
                self.TIMING_CLOCK.tick(self.FPS)
                timer += 1

        self.best_indv_val_fitness = best_individual.fitness


    def runEvolution(self):

        self.createPopulation()

        while self.generationNumber <= self.MAX_GENERATIONS:

            self.population = deepcopy(self.new_population)

            # dekodowanie
            self.dekodeDNAToNetwork()

            #ewaluowanie
            self.evaluate_fitness()
            self.population.sort(key=Individual.sortKey, reverse=True)
            self.validateBest()

            for i in range(self.POPULATION_SIZE):
                self.population[i].new = False

            # elity
            for i in range(self.elite_amount):
                self.new_population[i] = deepcopy(self.population[i])

            # nowi
            for i in range(self.elite_amount):
                indv = Individual()
                indv.dnaType = self.DNA_init()
                self.new_population[i + self.elite_amount] = indv

            # selekcja rodzicow, krzyżowanie, dzieci, mutacje
            for i in range(self.POPULATION_SIZE - self.elite_amount * 2):
                parent1 = self.selection(self.population)
                parent2 = self.selection(self.population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                child.new = True
                child.fitness = 0
                child.testFitness = 0
                child.nn = None
                self.new_population[i + self.elite_amount * 2] = child


            print("Generacja nr.", self.generationNumber)
            self.generationNumber += 1
            self.save_fitness_results()
            self.graph()

    def run(self) -> None:
        self.drawTrack()
        self.drawValTrack()
        self.placeCar()
        self.placeValCar()
        self.saveParams()
        self.runEvolution()

    def drawNetwork(self, neuralNet):
        WIDTH = 300
        HEIGHT = 200
        WHITE = (255, 255, 255)
        BLUE = (0, 102, 204)
        RED = (204, 0, 0)
        BLACK = (0, 0, 0)
        # self.screen.fill(WHITE, {0,0,WIDTH, HEIGHT})
        # Load font
        font = pygame.font.Font(None, 30)

        action = ["LEWO", "PRAWO", "SZYBCIEJ", "WOLNIEJ"]

        layers = neuralNet.layerSizes
        maxNeurons = max(layers)

        x_spacing = WIDTH // (len(layers) + 1)
        y_spacing = HEIGHT // (maxNeurons + 1)

        for i in range(5):
            y = (i - 1) * y_spacing + HEIGHT // 2 - 50
            text_surface = font.render(str((i - 2) * self.SENSOR_ANGLE) + "°", True, BLACK)
            self.screen.blit(text_surface, (5, y - 10))

        for i in range(4):
            y = (i - 0.5) * y_spacing + HEIGHT // 2 - 50
            text_surface = font.render(action[i], True, BLACK)
            self.screen.blit(text_surface, (WIDTH - x_spacing / 2, y - 10))

        back_neuron_positions = []
        neuron_positions = []
        for i, layer_size in enumerate(layers):
            x = (i + 1) * x_spacing + 10
            layer_positions = []
            for j in range(layer_size):
                y = (j + 1) * y_spacing + HEIGHT // 2 - y_spacing * (layer_size - 1) // 2 - 50
                pygame.draw.circle(self.screen, WHITE, (x, y), 10)
                layer_positions.append((x, y))
            back_neuron_positions.extend(layer_positions)
            neuron_positions.append(layer_positions)

        flattenedNN = np.array([weight for layer in neuralNet.neuralNetwork for row in layer for weight in row])
        maxWeight = np.max(np.abs(flattenedNN))
        for i, layer in enumerate(neuralNet.neuralNetwork):
            for j, weights in enumerate(layer):
                if j == 0:
                    continue
                for k, neuron in enumerate(weights):
                    if neuron != 0:
                        start_pos = back_neuron_positions[j - 1]
                        end_pos = neuron_positions[i + 1][k]
                        color = RED if weights[k] > 0 else BLUE
                        lineWidth = abs(5 * weights[k] / maxWeight)
                        pygame.draw.line(self.screen, color, start_pos, end_pos, int(lineWidth) + 1)

        pygame.display.flip()



    def save_fitness_results(self):

        fullPath = self.output_path + "/bestDNA.txt"
        with open(fullPath, 'w') as file:
            stringDNA = "".join(str(b) for b in self.population[0].dnaType.DNA)
            file.write(stringDNA)

        bestFitness = self.population[0].fitness

        # collect all fitness values
        fitness_values = [ind.fitness for ind in self.population]

        avg = np.mean(fitness_values)

        # compute median fitness
        medianFitness = statistics.median(fitness_values)

        print("Najlepszy wynik generacji: " + format(bestFitness, ".3f"))
        print("Srednie przystosowanie: " + format(avg, ".3f"))
        print("Testowe przystosowanie: " + format(self.best_indv_val_fitness, ".3f"))
        print("")

        self.bestFitnessList.append(bestFitness)
        self.medianFitnessList.append(avg)
        self.testFitnessList.append(self.best_indv_val_fitness)
        self.generationsList.append(len(self.generationsList))

        fullDataPath = self.output_path + "/graphData.txt"
        with open(fullDataPath, 'w') as file:
            for i in range(0, len(self.bestFitnessList)):
                file.write(str(self.generationsList[i]))
                file.write(" ")
                file.write(str(self.bestFitnessList[i]))
                file.write(" ")
                file.write(str(self.medianFitnessList[i]))
                file.write(" ")
                file.write(str(self.testFitnessList[i]))
                file.write("\n")


    def saveParams(self):
        fullDataPath = self.output_path + "/parameters.txt"
        with open(fullDataPath, 'w') as file:
            file.write("FPS  " + str(self.FPS) + "\n")
            file.write("WINDOW_WIDTH  " + str(self.WINDOW_WIDTH) + "\n")
            file.write("WINDOW_HEIGHT  " + str(self.WINDOW_HEIGHT) + "\n")
            file.write("MAX_GENERATIONS  " + str(self.MAX_GENERATIONS) + "\n")
            file.write("POPULATION_SIZE  " + str(self.POPULATION_SIZE) + "\n")
            file.write("READ_FROM_FILE  " + str(self.READ_FROM_FILE) + "\n")
            file.write("USE_MAP  " + str(self.USE_MAP) + "\n")
            file.write("CAR_X  " + str(self.STARTING_POINT[0]) + "\n")
            file.write("CAR_Y  " + str(self.STARTING_POINT[1]) + "\n")
            file.write("CAR_A  " + str(self.STARTING_ANGLE) + "\n")
            file.write("CAR_VAL_X  " + str(self.STARTING_VAL_POINT[0]) + "\n")
            file.write("CAR_VAL_Y  " + str(self.STARTING_VAL_POINT[1]) + "\n")
            file.write("CAR_VAL_A  " + str(self.STARTING_VAL_ANGLE) + "\n")

    def loadParams(self):
        fullDataPath = "evolutionOutput/parameters.txt"
        parameters = {}
        with open(fullDataPath, 'r') as file:
            for line in file:

                line = line.strip()

                key, value = line.split(maxsplit=1)

                if value.isdigit():
                    parameters[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    parameters[key] = float(value)
                elif value.lower() in ("true", "false"):
                    parameters[key] = value.lower() == "true"
                else:
                    parameters[key] = value

            self.FPS = parameters["FPS"]
            self.WINDOW_WIDTH = parameters["WINDOW_WIDTH"]
            self.WINDOW_HEIGHT = parameters["WINDOW_HEIGHT"]
            self.DATA_MODEL = parameters["DATA_MODEL"]
            self.MAX_GENERATIONS = parameters["MAX_GENERATIONS"]
            self.POPULATION_SIZE = parameters["POPULATION_SIZE"]
            self.SURVIVAL_RATE = parameters["SURVIVAL_RATE"]
            self.MIN_DURATION = parameters["MIN_DURATION"]
            self.MAX_DURATION = parameters["MAX_DURATION"]
            self.READ_FROM_FILE = parameters["READ_FROM_FILE"]
            self.USE_CROSSOVER = parameters["USE_CROSSOVER"]
            self.CAR_WIDTH = parameters["CAR_WIDTH"]
            self.CAR_HEIGHT = parameters["CAR_HEIGHT"]
            self.MINIMUM_SPEED = parameters["MINIMUM_SPEED"]
            self.TURN_SPEED = parameters["TURN_SPEED"]
            self.ACCELERATION = parameters["ACCELERATION"]
            self.DRAW_SENSORS = parameters["DRAW_SENSORS"]
            self.SENSORS_DRAW_DISTANCE = parameters["SENSORS_DRAW_DISTANCE"]
            self.SENSOR_ANGLE = parameters["SENSOR_ANGLE"]
            self.USE_MAP = parameters["USE_MAP"]
            self.CAR_X = parameters["CAR_X"]
            self.CAR_Y = parameters["CAR_Y"]
            self.CAR_A = parameters["CAR_A"]

    def loadPos(self):
        fullDataPath = "assets/parameters.txt"
        parameters = {}
        with open(fullDataPath, 'r') as file:
            for line in file:

                line = line.strip()

                key, value = line.split(maxsplit=1)

                if value.isdigit():
                    parameters[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    parameters[key] = float(value)
                elif value.lower() in ("true", "false"):
                    parameters[key] = value.lower() == "true"
                else:
                    parameters[key] = value

            self.CAR_X = parameters["CAR_X"]
            self.CAR_Y = parameters["CAR_Y"]
            self.CAR_A = parameters["CAR_A"]

    def loadValPos(self):
        fullDataPath = "assets/parameters.txt"
        parameters = {}
        with open(fullDataPath, 'r') as file:
            for line in file:

                line = line.strip()

                key, value = line.split(maxsplit=1)

                if value.isdigit():
                    parameters[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    parameters[key] = float(value)
                elif value.lower() in ("true", "false"):
                    parameters[key] = value.lower() == "true"
                else:
                    parameters[key] = value

            self.CAR_VAL_X = parameters["CAR_VAL_X"]
            self.CAR_VAL_Y = parameters["CAR_VAL_Y"]
            self.CAR_VAL_A = parameters["CAR_VAL_A"]

    def graph(self):
        tick_spacing = int(len(self.generationsList) / 10 + 1)

        self.ax.clear()
        self.ax.plot(self.generationsList, self.bestFitnessList, label="Najlepsze dopasowanie trasy treningowej", color="blue")
        self.ax.plot(self.generationsList, self.medianFitnessList, label="Średnie dopasowanie trasy treningowej", color="red")
        self.ax.plot(self.generationsList, self.testFitnessList, label="Dopasowanie trasy testowej najlepszego osobnika", color="green")
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        self.ax.set_xlabel("Generacje")
        self.ax.set_ylabel("Przystosowanie")
        self.fig.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

        plt.tight_layout()

        # self.ax.set_ylim(bottom=0)
        fullPath = self.output_path + "/bestFitnessGraph.png"
        self.ax.grid(True, alpha=0.3)

        plt.savefig(fullPath, bbox_inches="tight")
        img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("Wykres najlepszego wyniku", img)



