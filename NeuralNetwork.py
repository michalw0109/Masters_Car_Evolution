import numpy as np
import re


class NeuralNetwork:
    
    def __init__ (self):
        
        ### reproduction parameters ###
        #_____________________________#
                                    
        # create new link during simulation
        self.probOfNewConnection = 0.8
        
        # value probOfNewConnection is multiplied by on every iteration
        self.probOfNewConnectionMult = 0.95
        
        self.minProbOfNewConnection = 0.01

        # delete a connection during simulation
        self.probOfDelConnection = 0.8
        
        # value probOfDelConnection is multiplied by on every iteration
        self.probOfDelConnectionMult = 0.95        
        
        self.minProbOfDelConnection = 0.01

        # create new neuron
        self.probOfNewNeuron = 0.8
        
        # value probOfNewNeuron is multiplied by on every iteration
        self.probOfNewNeuronMult = 0.99
        
        self.minProbOfNewNeuron = 0.05

        
        
        # max standard deviation of values added to weights when creating offspring - 
        # if reproductionStdDev goes to min it goes back up to max
        self.maxReproductionStdDev = 0.5
        
        # standard deviation of values added to weights when creating offspring - changes every reproduction based on mult
        self.reproductionStdDev = self.maxReproductionStdDev

        # value std dev is multiplied by on every iteration
        self.stdDevMult = 0.92
        
        # min standard deviation of values added to weights when creating offspring
        self.minReproductionStdDev = 0.001
        
        # value min is multiplied by on every StdDev reset to max 
        self.minReproductionStdDevMul = 0.7
        
        #_____________________________#

        ###    network parameters   ###
        #_____________________________#
                                    
        # non zero weight at start - linking input and output
        self.probOfStartConnection = 0.5
        
        # number of tries for creating new neuron at start
        self.startingMutationMagnitude = 2
        
        # standard deviation of normal distr for weights
        self.startingStdDev = 1
        
        # structure of network
        self.nrOfInputs = 5
        self.nrOfHiddenLayers = 3
        self.nrOfOutputs = 4
        
        # number of layers in network, input layer not included
        self.networkSize = self.nrOfHiddenLayers + 1

        # array of sizes of layers (number of neurons for given layer)
        self.layerSizes = np.zeros(self.nrOfHiddenLayers + 2, dtype = int)

        # array of numbers of weights in each neuron for given layer
        self.layerWeightsSizes = np.zeros(self.nrOfHiddenLayers + 2, dtype = int)
        
        # maximum abs walue of a weight
        self.maxAbsWeightVal = 32
        
        #_____________________________#
        
        # core structure of the network
        self.neuralNetwork = []
        
        self.chromosome = ""
        #_____________________________#
        
        # make list of layer sizes
        self.layerSizes[0] = self.nrOfInputs
        self.layerSizes[self.layerSizes.size - 1] = self.nrOfOutputs
        
        # make a list of number of weights in neurons for every layer
        for i in range(1, self.layerWeightsSizes.size):
            self.layerWeightsSizes[i] += 1
            for j in range(0, i):
                self.layerWeightsSizes[i] += self.layerSizes[j]
                
        # make whole network structure for weights
        for i in range(1, self.layerSizes.size):
            self.neuralNetwork.append(np.zeros((self.layerSizes[i], self.layerWeightsSizes[i])).T)
            
        # check for some new neuron mutations
        for i in range(0, self.startingMutationMagnitude):
            if np.random.random() < self.probOfNewNeuron:
                self.addNeuron(int(np.random.random() * self.nrOfHiddenLayers) + 1)
            
        # get the neurons random weights
        for layer in range(0, self.networkSize):
            for neuron in range(0, self.layerSizes[layer + 1]):
                for weight in range(0, self.layerWeightsSizes[layer + 1]):
                    if np.random.random() < self.probOfStartConnection:
                        self.neuralNetwork[layer][weight][neuron] = np.random.normal(0, self.startingStdDev)
                        
        self.setChromosome()

    def activationFunction(self, x):
        x_clamped = np.clip(x, -709, 709)  # Prevent overflow in exp
        return 1 / (1 + np.exp(-x_clamped))

    def activationFunction2(self, x):
        return 2 * x

    def printNetwork(self):
        for i in range(0, self.networkSize):
            print(self.neuralNetwork[i])
            print(" ")
            
    def compute(self, input):
        input.insert(0, 1)
        for i in range(0, self.networkSize - 1):
            output = self.activationFunction(np.dot(input, self.neuralNetwork[i]))
            input.extend(output)
        output = self.activationFunction(np.dot(input, self.neuralNetwork[self.networkSize - 1]))
        return output
            
    def addNeuron(self, layerNr):
        newLayerSizes = self.layerSizes.copy()
        
        # update list of layer sizes
        newLayerSizes[layerNr] += 1
        
        # make a list of number of weights in neurons for every layer
        newLayerWeightsSizes = np.zeros(newLayerSizes.size, dtype = int)
        for i in range(1, newLayerWeightsSizes.size):
            newLayerWeightsSizes[i] += 1
            for j in range(0, i):
                newLayerWeightsSizes[i] += newLayerSizes[j]
        
        # make whole network structure for weights
        newNeuralNetwork = []
        for i in range(1, newLayerSizes.size):
            newNeuralNetwork.append(np.zeros((newLayerSizes[i], newLayerWeightsSizes[i])).T)

        # copy the weights
        for layer in range(0, self.networkSize):
            for neuron in range(0, self.layerSizes[layer + 1]):
                newNeuron = 0
                for weight in range(0, self.layerWeightsSizes[layer + 1]):
                    if weight == self.layerWeightsSizes[layerNr+1]:
                        newNeuron = 1
                    newNeuralNetwork[layer][weight + newNeuron][neuron] = self.neuralNetwork[layer][weight][neuron]
                    
        # copy results
        self.layerSizes = newLayerSizes
        self.layerWeightsSizes = newLayerWeightsSizes
        self.neuralNetwork = newNeuralNetwork
        

    def networkCopy(self):
        newNetwork = []
        for i in range(1, self.networkSize + 1):
            newNetwork.append(np.zeros((self.layerSizes[i], self.layerWeightsSizes[i])).T)
        for layer in range(0, self.networkSize):
            for neuron in range(0, self.layerSizes[layer + 1]):
                for weight in range(0, self.layerWeightsSizes[layer + 1]):
                    newNetwork[layer][weight][neuron] = self.neuralNetwork[layer][weight][neuron]
                    if(newNetwork[layer][weight][neuron] > self.maxAbsWeightVal):
                        newNetwork[layer][weight][neuron] = self.maxAbsWeightVal
                    if(newNetwork[layer][weight][neuron] < -self.maxAbsWeightVal):
                        newNetwork[layer][weight][neuron] = -self.maxAbsWeightVal
        return newNetwork

    def copy(self):
        newNeuralNetwork = NeuralNetwork()
        newNeuralNetwork.layerSizes = self.layerSizes.copy()
        newNeuralNetwork.layerWeightsSizes = self.layerWeightsSizes.copy()
        return newNeuralNetwork
        
    def reproduce(self, firstParentFitness, otherParent, useCrossover):
        
        if useCrossover:
            
            # copy the parent
            otherParentFitness = otherParent.fitness        
            
            if firstParentFitness > otherParentFitness:
                child = self.copy()
                child.chromosome = self.chromosome
            else:
                child = otherParent.nn.copy()
                child.chromosome = otherParent.nn.chromosome
            
            firstParentChromosome = self.chromosome
            otherParentChromosome = otherParent.nn.chromosome
            
            firstParentRatio = firstParentFitness / (firstParentFitness + otherParentFitness)
            #otherParentRatio = otherParentFitness / (firstParentFitness + otherParentFitness)

            tempChromosome = list(child.chromosome)
            

            for i in range(0, min(len(firstParentChromosome), len(otherParentChromosome))):
                
                if tempChromosome[i] == '0' or tempChromosome[i] == '1':
                    if np.random.random() < firstParentRatio:
                        tempChromosome[i] = firstParentChromosome[i]
                    else:
                        tempChromosome[i] = otherParentChromosome[i]

            child.chromosome = ''.join(tempChromosome)  # Convert back to a string


            child.neuralNetwork = self.createWeightsFromChromosome(child)
            
            
            # check new neuron mutation
            if np.random.random() < self.probOfNewNeuron:
                child.addNeuron(int(np.random.random() * child.nrOfHiddenLayers) + 1)
            
            # change the weights, make or delete connections
            for layer in range(0, child.networkSize):
                for neuron in range(0, child.layerSizes[layer + 1]):
                    for weight in range(0, child.layerWeightsSizes[layer + 1]):
                        if(child.neuralNetwork[layer][weight][neuron] != 0):
                            if np.random.random() < self.probOfDelConnection:
                                child.neuralNetwork[layer][weight][neuron] = 0
                            else:
                                child.neuralNetwork[layer][weight][neuron] += np.random.normal(0, self.reproductionStdDev)
                        else:
                            if np.random.random() < self.probOfNewConnection:
                                child.neuralNetwork[layer][weight][neuron] = np.random.normal(0, self.reproductionStdDev)
            
            # update new parameters for reproduction
            self.probOfNewConnection = max(self.minProbOfNewConnection, self.probOfNewConnection * self.probOfNewConnectionMult)
            self.probOfDelConnection = max(self.minProbOfDelConnection, self.probOfDelConnection * self.probOfDelConnectionMult)
            self.probOfNewNeuron = max(self.minProbOfNewNeuron, self.probOfNewNeuron * self.probOfNewNeuronMult)
            self.reproductionStdDev *= self.stdDevMult
            if self.reproductionStdDev < self.minReproductionStdDev:
                self.reproductionStdDev = self.maxReproductionStdDev
                self.minReproductionStdDev *= self.minReproductionStdDevMul
                
            otherParent.nn.probOfNewConnection = max(otherParent.nn.minProbOfNewConnection, otherParent.nn.probOfNewConnection * otherParent.nn.probOfNewConnectionMult)
            otherParent.nn.probOfDelConnection = max(otherParent.nn.minProbOfDelConnection, otherParent.nn.probOfDelConnection * otherParent.nn.probOfDelConnectionMult)
            otherParent.nn.probOfNewNeuron = max(otherParent.nn.minProbOfNewNeuron, otherParent.nn.probOfNewNeuron * otherParent.nn.probOfNewNeuronMult)
            otherParent.nn.reproductionStdDev *= otherParent.nn.stdDevMult
            if otherParent.nn.reproductionStdDev < otherParent.nn.minReproductionStdDev:
                otherParent.nn.reproductionStdDev = otherParent.nn.maxReproductionStdDev
                otherParent.nn.minReproductionStdDev *= otherParent.nn.minReproductionStdDevMul
            
            child.setChromosome()
            
            return child
        else:
            # copy the parent
            child = self.copy()
            child.neuralNetwork = self.networkCopy()
            # check new neuron mutation
            if np.random.random() < self.probOfNewNeuron:
                child.addNeuron(int(np.random.random() * child.nrOfHiddenLayers) + 1)
            
            # change the weights, make or delete connections
            for layer in range(0, child.networkSize):
                for neuron in range(0, child.layerSizes[layer + 1]):
                    for weight in range(0, child.layerWeightsSizes[layer + 1]):
                        if(child.neuralNetwork[layer][weight][neuron] != 0):
                            if np.random.random() < self.probOfDelConnection:
                                child.neuralNetwork[layer][weight][neuron] = 0
                            else:
                                child.neuralNetwork[layer][weight][neuron] += np.random.normal(0, self.reproductionStdDev)
                        else:
                            if np.random.random() < self.probOfNewConnection:
                                child.neuralNetwork[layer][weight][neuron] = np.random.normal(0, self.reproductionStdDev)
            
            # update new parameters for reproduction
            self.probOfNewConnection = max(self.minProbOfNewConnection, self.probOfNewConnection * self.probOfNewConnectionMult)
            self.probOfDelConnection = max(self.minProbOfDelConnection, self.probOfDelConnection * self.probOfDelConnectionMult)
            self.probOfNewNeuron = max(self.minProbOfNewNeuron, self.probOfNewNeuron * self.probOfNewNeuronMult)
            #self.probOfDelNeuron *= self.probOfDelNeuronMult
            self.reproductionStdDev *= self.stdDevMult
            if self.reproductionStdDev < self.minReproductionStdDev:
                self.reproductionStdDev = self.maxReproductionStdDev
                self.minReproductionStdDev *= self.minReproductionStdDevMul
            
            
            return child
        
    
    
    def test():

        newNetwork = NeuralNetwork()

        # some print
        print(newNetwork.layerSizes)
        print(newNetwork.layerWeightsSizes)
        newNetwork.printNetwork()
        flattenedNN = np.array([weight for layer in newNetwork.neuralNetwork for row in layer for weight in row])
        print(flattenedNN)
        


    
    def setChromosome(self):
        flattenedNN = np.array([weight for layer in self.neuralNetwork for row in layer for weight in row])
        self.chromosome = ""
        for weight in flattenedNN:
            self.chromosome+=self.floatToBinary(weight)
        #print(self.chromosome)

    def createWeightsFromChromosome(self, network):
        newNetwork = []
        counter = 0
        binaryNumber = r'[np][01]+\.[01]{10}'  # [01]+ matches the integer part, \.[01]{2} matches the fractional part

        # Find all matches
        binaryNumbers = re.findall(binaryNumber, network.chromosome)
        
        #print(network.layerSizes)
        #print(len(binaryNumbers))    
        
            
        for i in range(1, network.networkSize + 1):
            newNetwork.append(np.zeros((network.layerSizes[i], network.layerWeightsSizes[i])).T)
        for layer in range(0, network.networkSize):
            for neuron in range(0, network.layerSizes[layer + 1]):
                for weight in range(0, network.layerWeightsSizes[layer + 1]):
                    newNetwork[layer][weight][neuron] = self.binaryToFloat(binaryNumbers[counter])
                    counter += 1
                    #print(counter)
                    if(newNetwork[layer][weight][neuron] > network.maxAbsWeightVal):
                        newNetwork[layer][weight][neuron] = network.maxAbsWeightVal
                    if(newNetwork[layer][weight][neuron] < -network.maxAbsWeightVal):
                        newNetwork[layer][weight][neuron] = -network.maxAbsWeightVal
        return newNetwork
    
    def saveToFile(self, filePath):
        with open(filePath, 'w') as file:
        # Write some values to the file
            file.write("size = \n")
            for i in range(0, self.layerSizes.size):
                file.write(str(self.layerSizes[i]))
                file.write(" ")
            file.write("\n")
            for layer in range(0, self.networkSize):
                file.write("layer nr. ")
                file.write(str(layer + 1))
                file.write("\n")
                if self.layerSizes[layer + 1] == 0:
                        file.write("no neurons \n")
                else:
                    for weight in range(0, self.layerWeightsSizes[layer + 1]):
                        for neuron in range(0, self.layerSizes[layer + 1]):
                            file.write("{:.8f}".format(self.neuralNetwork[layer][weight][neuron], 10))
                            file.write("  ")
                        file.write("\n")
                
                
    def readFromFile(self, filePath):
        with open(filePath, 'r') as file:
            lines = file.readlines()
            # Split the second line into float values
            sizeValues = [float(val) for val in lines[1].split()]
            for i in range(0, len(sizeValues)):
                self.layerSizes[i] = sizeValues[i]
            
            
            self.layerWeightsSizes = np.zeros(self.nrOfHiddenLayers + 2, dtype = int)
            # make a list of number of weights in neurons for every layer
            for i in range(1, self.layerWeightsSizes.size):
                self.layerWeightsSizes[i] += 1
                for j in range(0, i):
                    self.layerWeightsSizes[i] += self.layerSizes[j]
                    
            # make whole network structure for weights
            self.neuralNetwork = []
            for i in range(1, self.layerSizes.size):
                self.neuralNetwork.append(np.zeros((self.layerSizes[i], self.layerWeightsSizes[i])).T)
             
            iterator = 3
            # get the neurons random weights
            for layer in range(0, self.layerSizes.size - 1):
                if self.layerSizes[layer + 1] > 0:
                    for weight in range(0, self.layerWeightsSizes[layer + 1]):
                        weightValues = [float(val) for val in lines[iterator].split()]
                        for neuron in range(0, self.layerSizes[layer + 1]):
                            self.neuralNetwork[layer][weight][neuron] = weightValues[neuron]
                        iterator += 1
                else:
                    iterator += 1
                iterator += 1
        self.setChromosome()
            
                
    def floatToBinary(self, num, precision=10):
        # Check for negativity
        is_negative = num < 0
        num = abs(num)  # Work with the absolute value

        # Separate the integral and fractional parts
        integral_part = int(num)
        fractional_part = num - integral_part

        # Convert the integral part to binary
        integral_binary = bin(integral_part).replace("0b", "")
        if len(integral_binary) > 6:
            integral_binary = integral_binary[:6]  # Trim to first 6 characters
        elif len(integral_binary) < 6:
            integral_binary = integral_binary.zfill(6)  # Pad with '0' to 6 characters

        # Convert the fractional part to binary
        fractional_binary = []
        while len(fractional_binary) < precision:
            fractional_part *= 2
            bit = int(fractional_part)
            fractional_binary.append(str(bit))
            fractional_part -= bit

        # Combine the results
        binary_result = f"{integral_binary}.{''.join(fractional_binary)}"

        # Add the negative sign if necessary
        return f"n{binary_result}" if is_negative else f"p{binary_result}"



    def binaryToFloat(self, binary_str):
        # Check for negative sign
        is_negative = binary_str.startswith('n')
        binary_str = binary_str[1:]  # Remove the sign for processing

        # Split into integral and fractional parts
        if '.' in binary_str:
            integral_str, fractional_str = binary_str.split('.')
        else:
            integral_str, fractional_str = binary_str, ''

        # Convert the integral part
        integral_part = int(integral_str, 2)

        # Convert the fractional part
        fractional_part = 0
        for i, bit in enumerate(fractional_str):
            fractional_part += int(bit) * (2 ** -(i + 1))

        # Combine both parts
        result = integral_part + fractional_part

        # Apply the negative sign if necessary
        return -result if is_negative else result


    



