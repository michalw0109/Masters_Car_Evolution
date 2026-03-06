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


from copy import deepcopy

import random


from Individual import Individual
from Computation import Computation
from Crossover import Crossover
from Mutation import Mutation
from Selection import Selection
from DNA import DNA









class EvolutionEngine:

    def __init__(self, _MAX_GENERATIONS,_POPULATION_SIZE, _ELITE_FRACTION,
                                _DNA_INITIALIZATION, _DNA_DECODER, _COMPUTATION, _SELECTION, _CROSSOVER, _MUTATION):


        # some execution params, arent in research, can be constant
        self.FPS = 12000
        self.WINDOW_WIDTH = 2200
        self.WINDOW_HEIGHT = 1300

        self.READ_FROM_FILE = False

        self.USE_MAP = True
        self.LOAD_POS = True
        self.COLLISION_SURFACE_COLOR = Color.GREEN

        self.LOAD_MODEL = False

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
        self.CAR_X = 0
        self.CAR_Y = 0
        self.CAR_A = 0

        self.elite_amount = int(self.POPULATION_SIZE * self.ELITE_FRACTION)

        self.TIME = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

        self.output_path = "evolutionOutput/" + self.TIME

        os.makedirs(self.output_path, exist_ok=True)

        self.WINDOW_MID_POINT = (self.WINDOW_WIDTH / 2, self.WINDOW_HEIGHT / 2)

        self.population: list[Individual] = []
        self.new_population : list[Individual]= []

        self.generationsList = [0]
        self.bestFitnessList = [0]

        self.fig, self.ax = plt.subplots(1, 1)

        self.TITLE = "Ewolucja pojazdów"
        pygame.init()
        pygame.display.set_caption(self.TITLE)
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.screen.fill(self.COLLISION_SURFACE_COLOR)

        self.track = None

        self.STARTING_POINT: list[float] = []
        self.STARTING_ANGLE = None

        self.TIMING_CLOCK = pygame.time.Clock()

        self.generationNumber = 1

        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 11

        self.collision_map = None



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
            CarPopulation.append(Car(self.STARTING_POINT, self.STARTING_ANGLE, self.track, self.population[i].nn, self.compute, self.collision_map))
        runningGeneration = True
        remainingCars = self.POPULATION_SIZE
        timer = 0
        bestFitness = 0
        while runningGeneration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            self.screen.blit(self.track, (0, 0))
            for i in range(0, self.POPULATION_SIZE):
                if CarPopulation[i].alive:
                    CarPopulation[i].update(timer)
                    if not CarPopulation[i].alive:
                        remainingCars -= 1
                    if CarPopulation[i].fitness > bestFitness:
                        bestFitness = CarPopulation[i].fitness
                        bestIndex = i
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
            pygame.display.update()
            #self.TIMING_CLOCK.tick(self.FPS)
            timer += 1

        for i in range(0, self.POPULATION_SIZE):
            self.population[i].fitness = CarPopulation[i].fitness

    def runEvolution(self):

        self.createPopulation()

        while self.generationNumber <= self.MAX_GENERATIONS:

            self.population = deepcopy(self.new_population)

            # dekodowanie
            self.dekodeDNAToNetwork()

            #ewaluowanie
            self.evaluate_fitness()
            self.population.sort(key=Individual.sortKey, reverse=True)

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
                self.new_population[i + self.elite_amount * 2] = child


            print("Generacja nr.", self.generationNumber)
            self.generationNumber += 1
            self.save_fitness_results()
            self.graph()

    def run(self) -> None:
        self.drawTrack()
        self.placeCar()
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
        print("Najlepszy wynik generacji: " + format(bestFitness, ".3f"))
        print("")
        self.bestFitnessList.extend([bestFitness])
        self.generationsList.extend([len(self.generationsList)])

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

    def graph(self):
        tick_spacing = int(len(self.generationsList) / 10 + 1)

        self.ax.plot(self.generationsList, self.bestFitnessList)
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # self.ax.set_ylim(bottom=0)
        fullPath = self.output_path + "/bestFitnessGraph.png"

        plt.savefig(fullPath)
        img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("Wykres najlepszego wyniku", img)

        fullDataPath = self.output_path + "/graphData.txt"
        with open(fullDataPath, 'w') as file:
            for i in range(0, len(self.bestFitnessList)):
                file.write(str(self.generationsList[i]))
                file.write(" ")
                file.write(str(self.bestFitnessList[i]))
                file.write("\n")

