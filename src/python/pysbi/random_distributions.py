## This module provides functionality to generate random numbers with arbitrary
## distributions.
##
## See http://code-spot.co.za/2008/09/21/generating-random-numbers-with-arbitrary-distributions/
##
## @author Herman Tulleken (herman.tulleken@gmail.com)
##


from __future__ import division

from math import exp

from random import seed
from random import random

newSampleCount = 1000
THRESHOLD = 0.001

def lerp(value, inputMin, inputMax, outputMin, outputMax):
    if value >= inputMax:
        return outputMax

    return ramp(value, inputMin, inputMax, outputMin, outputMax)


def sigmoid(value, inputMin, inputMax, outputMin, outputMax):
    w = exp((-2 * value + (inputMax + inputMin))/ (inputMax - inputMin))

    return (outputMax - outputMin) / (1 + w) + outputMin;


def ramp(value, inputMin, inputMax, outputMin, outputMax):
    if value <= inputMin:
        return outputMin

    return line(value, inputMin, inputMax, outputMin, outputMax)

def line(value, inputMin, inputMax, outputMin, outputMax):
    return outputMin + ((value - inputMin) * (outputMax - outputMin) / (inputMax - inputMin))

##This class is described in AI Programming Wisdom 1,
#"The Beauty of Response Curves", by Bob Alexander.
#Essentailly, this class provides a look-up table with
#linear interpolation for arbitrary functions.
#@param n
#	Number of output samples.
#@param T
#	The number type of the input and output, usually float or double.

class ResponseCurve:

##	Constructs a new TransferFunction.
    #
    #	@param inputMin
    #		The minimum value an input can be.
    #	@param inputMax
    #		The maximum value an input can be.
    #	@param outputSamples
    #		Samples of outputs.

    def __init__(self, inputMin, inputMax, outputSamples):
        self.inputMin = inputMin
        self.inputMax = inputMax
        self.count = len(outputSamples)
        self.period = (inputMax - inputMin) / (self.count - 1)

        self.outputSamples = [0] * self.count

        for i in range(self.count):
            self.outputSamples[i] = outputSamples[i]


    #	If the input is below the inputMin given in the constructor,
    #	the output is clamped to the first output sample.
    #
    #	If the input is above the inputMax given in the constructor,
    #	the output is clamped to the last output sample.
    #
    #	Otherwise an index is calculated, and the output is interpolated
    #	between outputSample[index] and outputSample[index + 1].
    #
    #	@param input
    #		The input for which output is sought.
    def __call__ (self, input):
        if input <= self.inputMin:
            return self.outputSamples[0]

        if input >= self.inputMax:
            return self.outputSamples[-1]


        index = int((input - self.inputMin)/(self.period))
        inputSampleMin = self.inputMin + self.period * index

        return lerp(input, inputSampleMin, inputSampleMin + self.period, self.outputSamples[index], self.outputSamples[index + 1])

    def getInputMin(self):
        return self.inputMin

    def getInputMax(self):
        return self.inputMax

##	Similar to ResponseCurve, but allows sample points to be unevenly spaced.
#
#	This curve is slower than the ordinary ResponseCurve. However, it is useful
#	for generating the inverse of a monotonic function. For rapid access, this
#	curve should be sampled into a ordinary ResponseCurve.
class XYResponseCurve:
    ##
    #	Construct a new XYResponse curve from input and output samples
    #
    #	@param inputSamples
    #		The input values for this response curve. Must be strictly increasing.
    #	@param outputSamples
    #		The output vlaues for this curve.
    def __init__(self, inputSamples, outputSamples):
        self.count = len(inputSamples)

        if self.count != len(outputSamples):
            raise Exception('Number of input samples does not match number of output samples')

        self.inputSamples = [0] * self.count
        self.outputSamples = [0] * self.count

        for i in range(self.count):
            self.inputSamples[i] = inputSamples[i];
            self.outputSamples[i] = outputSamples[i];


    ##	If the input is below the inputMin given in the constructor,
    #	the output is clamped to the first output sample.

    #	If the input is above the inputMax given in the constructor,
    #	the output is clamped to the last output sample.

    #	Otherwise an index is calculated, and the output is interpolated
    #	between outputSample[index] and outputSample[index + 1].

    #	@param input
    #		The input for which output is sought.
    def __call__(self, input):
        if input <= self.inputSamples[0]:
            return self.outputSamples[0];


        if input >= self.inputSamples[-1]:
            return self.outputSamples[- 1]

        index = self.findInputIndex(input)


        x1 = self.inputSamples[index + 1]
        x0 = self.inputSamples[index]

        tau = (input - x0) / (x1 - x0)
        y1 = self.outputSamples[index + 1]
        y0 = self.outputSamples[index]

        return (y1 - y0) * tau + y0

    #Makes this XYResponseCurve into its inverse function.
    def makeInverse(self):
        tmp = self.inputSamples
        self.inputSamples = self.outputSamples
        self.outputSamples = tmp


    ##	@private
    # Test which input sample lies to the left of the given input.
    def findInputIndex(self, input):
        min = 0
        max = self.count

        while max > min + 1:
            mid = (max + min) // 2

            if input < self.inputSamples[mid]:
                max = mid
            else:
                min = mid

        return min


## This class wraps a ResponseCurve, and map inputs from [0 1] to the correct
#range for the ResponseCurve.
class NormalisedInputCurve:
    def __init__ (self, curve):
        self.curve = curve

    ## @param input
    #    is a value between 0 and 1.
    def __call__(self, input):
    #Step 4. Map random value to the appropriate input value for the response curve
        return self.curve(input*(self.curve.inputMax - self.curve.inputMin) + self.curve.inputMin)

## Makes a curve that accepts numbers in the range [0 1], and returns a result so
# that when the input is uniformely distributed random numbers, the result is
# random numbers distributed according to the samples with which this function is
# called.
def make_distribution_curve(inputSamples, outputSamples):
    newInputMin = outputSamples[0]
    newInputMax = sum(outputSamples)
    newOutputMax = inputSamples[-1]
    newOutputMin = inputSamples[0]

    oldSampleCount = len(inputSamples)
    accumulativeOutputSamples = [0] * oldSampleCount

    # Step 1. Calculate accumulative output

    accumulativeOutputSamples[0] = outputSamples[0]

    for i in range(oldSampleCount):
        accumulativeOutputSamples[i] = accumulativeOutputSamples[i - 1] + outputSamples[i]

    # Step2. Load inverse into XY response curve
    xyCurve = XYResponseCurve(accumulativeOutputSamples, inputSamples)

    newOutputSamples = [0] * newSampleCount

    # Step 3. Gather samples for ordinary reponse curve
    for i in range(newSampleCount):
        input = (i / (newSampleCount - 1)) * (newInputMax - newInputMin) + newInputMin
        newOutputSamples[i] = xyCurve(input)

    #Used for debugging.
    #printf("%f %f\n", input, newOutputSamples[i]);


    # Construct ordinary response curve from samples.
    curve = ResponseCurve(newInputMin, newInputMax, newOutputSamples)

    #Construct a curve that accepts normalised input
    curve = NormalisedInputCurve(curve)

    return curve

def demo():
    seed(0)	#Fixed seed for testing

    # Some arbitrary distribution curve
    inputSamples = [-20, -10, 0, 10, 20, 30, 40, 50, 60]
    outputSamples = [2, 10, 80, 75, 60, 30, 10, 5, 1]

    #newInputMin = outputSamples[0]
    #newInputMax = sum(outputSamples)

    newOutputMax = inputSamples[-1]
    newOutputMin = inputSamples[0]

    curve = make_distribution_curve(inputSamples, outputSamples)

    #test the distribution

    testSampleCount = 100;
    count = [0] * testSampleCount

    # generate 10 000 random numbers, and check count occurrences in distribution bands.
    for i in range(10000):
        uniformRandVal = random() #random value between 0 and 1

        #The random value that follows the desired distribution.
        curvedRandVal = curve(uniformRandVal)

        # Calculate the distribution band of the random value.
        countIndex = int((curvedRandVal - newOutputMin) / (newOutputMax - newOutputMin) * testSampleCount)

        if (countIndex < testSampleCount):
            count[countIndex] += 1
        else: # a rare occurence, but possible. Just bundle with the last distribution band.
            count[testSampleCount - 1] += 1



    # Print out, so that the values can be pasted into Excel or Calc.
    for i in range(testSampleCount):
        print '%d\t%d' % (i, count[i])
