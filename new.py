











# ------------------ IMPORTS ------------------
from EvolutionEngine import EvolutionEngine
import random


# ------------------ IMPORTS ------------------
import numpy as np
from Colors import Color

# ------------------ GLOBAL VARIABLES ------------------

FPS = 120

WINDOW_WIDTH = 1700
WINDOW_HEIGHT = 900

PATH_TO_FOLDER = R"C:\Users\micha\Desktop\studia\magisterka\Car_evolution/"

#------------EVOLUTION------------


# 0 - dataModelDiscrete
# 1 - dataModelHalfDiscrete
# 2 - dataModelContinuous
# 3 - dataModelHalfContinuous
DATA_MODEL = 3

MAX_GENERATIONS = 100

POPULATION_SIZE = 100

SURVIVAL_RATE = 0.3

MIN_DURATION = 100
MAX_DURATION = 100

READ_FROM_FILE = False

#------------CAR------------

CAR_WIDTH = 20
CAR_HEIGHT = 11

MINIMUM_SPEED = 0

TURN_SPEED = 3

ACCELERATION = 2

SENSOR_ANGLE = 25

COLLISION_SURFACE_COLOR = Color.GREEN

DRAW_SENSORS = True

SENSORS_DRAW_DISTANCE = 1920

USE_CROSSOVER = False

USE_MAP = False
CAR_X = 1100
CAR_Y = 130
CAR_A = 0

LOAD_PARAMS = False

def main() -> None:
    np.random.seed(1)
    random.seed(1)
    window = EvolutionEngine(FPS, WINDOW_WIDTH, WINDOW_HEIGHT, PATH_TO_FOLDER,
                 DATA_MODEL, MAX_GENERATIONS, POPULATION_SIZE, SURVIVAL_RATE, MIN_DURATION, MAX_DURATION, READ_FROM_FILE,
                 CAR_WIDTH, CAR_HEIGHT, MINIMUM_SPEED, TURN_SPEED, ACCELERATION,
                 COLLISION_SURFACE_COLOR, DRAW_SENSORS, SENSORS_DRAW_DISTANCE, SENSOR_ANGLE, USE_CROSSOVER, LOAD_PARAMS,
                 USE_MAP, CAR_X, CAR_Y, CAR_A)
    window.run()

if __name__ == "__main__":
    main()


