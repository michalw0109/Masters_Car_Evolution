from collections import defaultdict

import pygame
import math
from Colors import Color
from NeuralNetwork import NeuralNetwork
import numpy as np



class Action:
    TURN_LEFT = 0
    TURN_RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3


class Car:
    def __init__(self, _STARTING_POINT, _STARTING_ANGLE, _TRACK, _nn, _computeNetwork, _COLLISIONS_MASK):

        # some car params, arent in research, can be constant
        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 11

        self.MINIMUM_SPEED = 0
        self.TURN_SPEED = 3
        self.ACCELERATION = 2

        self.COLLISION_SURFACE_COLOR = Color.GREEN
        self.DRAW_SENSORS = False
        self.SENSORS_DRAW_DISTANCE = 5000
        self.SENSOR_ANGLES = 25
        self.CAR_SPRITE_PATH = "assets/"
        self.MAX_NR_OF_LAPS = 2

        # get evolution params from main
        self.STARTING_POINT = _STARTING_POINT
        self.STARTING_ANGLE = _STARTING_ANGLE
        self.TRACK = _TRACK
        self.nn = _nn
        self.computeNetwork = _computeNetwork
        self.COLLISIONS_MASK = _COLLISIONS_MASK


        # some iniitialization of variables
        self.WINDOW_MID_POINT = (self.TRACK.get_width() / 2, self.TRACK.get_height() / 2)

        self.baseSprite = pygame.image.load(self.CAR_SPRITE_PATH + "car_sprite.png").convert_alpha()
        self.baseSprite = pygame.transform.scale(self.baseSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))

        self.deadSprite = pygame.image.load(self.CAR_SPRITE_PATH + "car_dead_sprite.png").convert_alpha()
        self.deadSprite = pygame.transform.scale(self.deadSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))

        self.sprite = self.baseSprite.copy()

        self.trackMask = pygame.mask.from_threshold(self.TRACK, self.COLLISION_SURFACE_COLOR, (1, 1, 1, 255))  # Mask for green areas

        self.fitness = 0
        self.speed = self.MINIMUM_SPEED
        self.position = self.STARTING_POINT.copy()
        self.angle = self.STARTING_ANGLE

        self.sensorValues = []
        self.sensorHits = []
        self.inputVector = []

        self.alive = True

        self.startingTrackProgression = math.atan((self.position[1] - self.WINDOW_MID_POINT[1]) / (self.position[0] - self.WINDOW_MID_POINT[0])) + math.pi / 2
        self.trackProgression = self.startingTrackProgression
        self.maxTrackProgression = self.trackProgression
        self.halfLaps = 0

        self.deadFlag = False

        self.lastFitnessProgress = 0
        self.lastFastSpeed = 0



    def reset(self):
        self.sprite = self.baseSprite.copy()
        self.fitness = 0
        self.speed = self.MINIMUM_SPEED
        self.position = self.STARTING_POINT.copy()
        self.angle = self.STARTING_ANGLE
        self.sensorValues = []
        self.sensorHits = []
        self.alive = True
        self.trackProgression = self.startingTrackProgression
        self.maxTrackProgression = self.trackProgression
        self.halfLaps = 0
        self.deadFlag = False
        self.deadSprite = pygame.image.load(self.CAR_SPRITE_PATH + "car_dead_sprite.png").convert_alpha()
        self.deadSprite = pygame.transform.scale(self.deadSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))

    def updateSensors(self):
        #trackWidth = self.TRACK.get_width()
        #trackHeight = self.TRACK.get_height()

        collision_map = self.COLLISIONS_MASK
        step = 5
        self.sensorValues.clear()
        self.sensorHits.clear()
        for i in range(0, 5):
            sensorAngle = self.angle + (i - 2) * self.SENSOR_ANGLES
            radians = math.radians((360 - sensorAngle))
            xStep = math.cos(radians) * step
            yStep = math.sin(radians) * step
            length = 0

            sensorX: float = self.position[0]
            sensorY: float = self.position[1]


            # While the collision surface is not reached, increment the length of the sensor
            while not collision_map[int(sensorX)][int(sensorY)]:
                sensorX += xStep
                sensorY += yStep
                length += step

            distance = length
            self.sensorValues.append(distance)
            self.sensorHits.append([sensorX, sensorY])


    def computeReaction(self):
        self.inputVector.clear()
        self.inputVector.extend(self.sensorValues)
        self.inputVector.append(self.speed)

        output =  self.computeNetwork(self.nn, self.inputVector)

        def normalize_pair(a, b):
            if a>0 and b>0:
                s = a + b
                return a / s, b / s
            elif a<0 and b<0:
                a*=-1
                b*=-1
                (a,b) = (b,a)
                s = a + b
                return a / s, b / s

            a-=min(a, b)
            b-=min(a, b)
            s = a + b
            return (a / s, b / s) if s != 0 else (0, 0)

        tl, tr = normalize_pair(output[Action.TURN_LEFT], output[Action.TURN_RIGHT])
        acc, brk = normalize_pair(output[Action.ACCELERATE], output[Action.BRAKE])

        self.turn_left(tl)
        self.turn_right(tr)
        self.accelerate(acc)
        self.brake(brk)

    def updatePosition(self):
        # if self.speed == 0:
        #     self.alive = False
        radians = math.radians(360 - self.angle)
        self.position[0] += math.cos(radians) * self.speed
        self.position[1] += math.sin(radians) * self.speed

    def checkCollision(self):

        # if self.position[0] < 0 or self.position[1] < 0 or self.position[0] > self.TRACK.get_width() or self.position[
        #     1] > self.TRACK.get_height():
        #     self.alive = False
        #     return
        self.sprite = pygame.transform.rotate(self.baseSprite, self.angle)
        spriteMask = pygame.mask.from_surface(self.sprite)

        collisionPoint = self.trackMask.overlap(spriteMask, (self.position[0] - self.sprite.get_width() / 2,
                                                             self.position[1] - self.sprite.get_height() / 2))
        if collisionPoint:
            self.alive = False

    def updateFitness(self, time):
        newLapThreshold = 2.6

        if abs(self.position[0] - self.WINDOW_MID_POINT[0]) > 0.00001:

            newTrackProgression = math.atan((self.position[1] - self.WINDOW_MID_POINT[1]) / (self.position[0] - self.WINDOW_MID_POINT[0])) + math.pi / 2

            if (self.trackProgression - newTrackProgression) > newLapThreshold:
                self.halfLaps += 1

            if (newTrackProgression - self.trackProgression) > newLapThreshold:
                self.halfLaps -= 1

            self.trackProgression = newTrackProgression

            if self.trackProgression + self.halfLaps * math.pi > self.maxTrackProgression:
                self.lastFitnessProgress = time
                self.maxTrackProgression = self.trackProgression + self.halfLaps * math.pi

        if time - self.lastFitnessProgress > 1200:
            self.alive = False

        if (self.maxTrackProgression - self.startingTrackProgression) / (math.pi * 2) > self.MAX_NR_OF_LAPS:
            self.alive = False
            self.fitness = math.pi * 200 * self.MAX_NR_OF_LAPS + 6000.0 / (time / 120 + 1)
        else:
            self.fitness = (self.maxTrackProgression - self.startingTrackProgression) * 100


        if self.speed > 0.3:
            self.lastFastSpeed = time

        if time - self.lastFastSpeed > 1200:
            self.alive = False

    def update(self, time):
        self.updateSensors()
        self.computeReaction()
        self.updatePosition()
        self.checkCollision()
        if self.alive:
            self.updateFitness(time)

    def draw(self, screen) -> None:
        # Draw the car on the track (and its sensors if enabled)

        if self.alive:
            screen.blit(self.sprite, ((self.position[0] - self.sprite.get_width() / 2, self.position[1] - self.sprite.get_height() / 2)))
        else:
            if not self.deadFlag:
                self.deadFlag = True
                self.deadSprite = pygame.transform.rotate(self.deadSprite, self.angle)
            screen.blit(self.deadSprite, (
            (self.position[0] - self.deadSprite.get_width() / 2, self.position[1] - self.deadSprite.get_height() / 2)))

        if self.DRAW_SENSORS and self.alive:
            for position in self.sensorHits:
                pygame.draw.line(screen, Color.WHITE, self.position, position, 2)
                pygame.draw.circle(screen, Color.MAGENTA, position, 4)

    def accelerate(self, force):
        self.speed += self.ACCELERATION * force

    def brake(self, force):
        self.speed -= self.ACCELERATION * force
        if self.speed < self.MINIMUM_SPEED:
            self.speed = self.MINIMUM_SPEED

    def turn_left(self, force):
        self.angle += self.TURN_SPEED * force

    def turn_right(self, force):
        self.angle -= self.TURN_SPEED * force

