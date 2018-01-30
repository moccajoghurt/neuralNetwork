#include <iostream>
#include <vector>
#include <cstdlib> // exit
#include <cmath>
#include "neuralNetwork.h"

using namespace std;

/////////////////////////////////////////// class Neuron
Neuron::Neuron() {
    bias = 1.;
    learningRate = 0.5;
}
float Neuron::getOutput(vector<float> input) {
    float result = 0;
    if (input.size() != weights.size()) {
        cout << "Neuron::getOutput(): inputNum != weightNum" << endl;
        exit(1);
    }
    for (int i = 0; i < weights.size(); i++) {
        result += input[i] * weights[i];
    }
    // add bias
    result += (bias * biasWeight);
    lastCalculatedWeightedInputSum = result;
    lastCalculatedOutput = logisticFunc(result);
    lastInput = input;
    return lastCalculatedOutput;
}
void Neuron::initWeights(int inputCount) {
    for (int i = 0; i < inputCount; i++) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        weights.push_back(r);
    }
    biasWeight = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void Neuron::updateWeight(int index, float newVal) {
    if (index < 0 || index >= weights.size()) {
        cout << "Neuron::updateWeights(): invalid index" << endl;
        exit(1);
    }
    weights[index] = newVal;
}

float Neuron::simplifiedSigmoidFunc(float val) {
    float buf = val / (1 + abs(val));
    return buf;
}

float Neuron::logisticFunc(float val) {
    float buf = 1. / (1 + exp(-val));
    return buf;
}

// for testing
void Neuron::setWeights(vector<float> weightsParam) {
    if (weightsParam.size() != weights.size()) {
        cout << "Neuron::setWeights(): inputNum != weightNum" << endl;
        exit(1);
    }
    for (int i = 0; i < weightsParam.size(); i++) {
        weights[i] = weightsParam[i];
    }
}

/////////////////////////////////////////// class Network
void Network::createLayer(int neuronCount, int neuronInputsCount) {
    vector<Neuron> neuronLayer;
    for (int i = 0; i < neuronCount; i++) {
        Neuron n;
        n.initWeights(neuronInputsCount);
        neuronLayer.push_back(n);
    }
    neuralNetwork.push_back(neuronLayer);
}
vector<vector<Neuron> >& Network::forwardPropagate(vector<float> input) {
    if (input.size() % neuralNetwork[0].size() != 0) {
        cout << "The number of input values must be a multiple of the nodes in the input layer." << endl;
        exit(1);
    }
    vector<vector<Neuron> > updatedNeurons;
    vector<vector<float> > layerResults;
    int layerCount = 0;
    for (int i = 0; i < neuralNetwork.size(); i++) {
        if (i == 0) {
            int neuronCount = neuralNetwork[0].size();
            vector<float> neuronResults;
            for (int n = 0; n < neuronCount; n++) {
                neuronResults.push_back(neuralNetwork[0][n].getOutput(input));
            }
            layerResults.push_back(neuronResults);
            updatedNeurons.push_back(neuralNetwork[i]);
            layerCount++;
            continue;
        }
        int neuronCount = neuralNetwork[i].size();
        vector<float> neuronResults;
        for (int n = 0; n < neuronCount; n++) {
            // It's a fully connected Network, so feed each Neuron with all results of the previous layer
            float neuronOutput = neuralNetwork[i][n].getOutput(layerResults[layerCount - 1]);
            neuronResults.push_back(neuronOutput);
        }
        layerResults.push_back(neuronResults);
        updatedNeurons.push_back(neuralNetwork[i]);
        layerCount++;
    }
    neuralNetwork = updatedNeurons;
    return neuralNetwork;
}

float Network::derivateSimplifiedSigmoidFunc(float value) {
    float buf = (1. / ((abs(value) + 1) * (abs(value) + 1)) );
    return buf;
}

float Network::derivateLogisticFunc(float value) {
    float buf = value * (1 - value);
    return buf;
}

// implementation is based upon this: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
void Network::backPropagate(vector<float> trainingValues) {
    if (neuralNetwork.back().size() != trainingValues.size()) {
        cout << "error: Training data must have the same number of values as the neurons in the output layer." << endl;
        exit(1);
    }
    for (int i = neuralNetwork.size() - 1; i > 0; i--) {
        vector<float> totalErrorWrtOutputs;
        vector<float> outputWrtTotalNetInputs;
        if (i == neuralNetwork.size() - 1) {
            // we are in the Output Layer
            float totalError = 0;
            // calculate the total error
            for (int n = 0; n < neuralNetwork[i].size(); n++) {
                float deltaSum = trainingValues[n] - neuralNetwork[i][n].getLastCalculatedOutput();
                //float deltaOutputSum = derivateSigmoidFunc(neuralNetwork[i][n].getLastSum()) * deltaSum;
                float localError = (0.5 * deltaSum) * (0.5 * deltaSum);
                totalError += localError;
            }
            for (int n = 0; n < neuralNetwork[i].size(); n++) {
                vector<float> lastInputs = neuralNetwork[i][n].getLastInput();
                float totalErrorWrtOutput = -(trainingValues[n] - neuralNetwork[i][n].getLastCalculatedOutput()); // Wrt = with respect to
                float outputWrtTotalNetInput = derivateLogisticFunc(neuralNetwork[i][n].getLastCalculatedOutput());
                // cout << totalErrorWrtOutput << " " << outputWrtTotalNetInput << endl;
                totalErrorWrtOutputs.push_back(totalErrorWrtOutput);
                outputWrtTotalNetInputs.push_back(outputWrtTotalNetInput);
                vector<float> neuronWeights = neuralNetwork[i][n].getWeights();
                for (int x = 0; x < neuronWeights.size(); x++) {
                    float totalErrorWrtWeight = totalErrorWrtOutput * outputWrtTotalNetInput * lastInputs[x];
                    float newWeight = neuronWeights[x] - neuralNetwork[i][n].getLearningRate() * totalErrorWrtWeight;
                    // cout << newWeight << endl;
                }
            }
        }
        // we are in a hidden layer

        // iterate over neurons
        for (int n = 0; n < neuralNetwork[i].size(); n++) {
            // iterate over weights
            for (int m = 0; m < neuralNetwork[i][n].getWeights().size(); m++) {
                float totalErrorWrtWeight;
                float totalErrorWrtNeuronOutput;
                // iterate over the neurons of the layer to the right (leftmost = inputlayer, rightmost = outputlayer)
                for (int x = 0; x < neuralNetwork[i - 1].size(); x++) {
                    
                }
            }
        }
    }
}

// purely for testing
void Network::setWeights(int layerIndex, vector<vector<float> > weightsParam) {
    if (weightsParam.size() != neuralNetwork[layerIndex].size()) {
        cout << "Network::setWeights(): inputNum != weightNum" << endl;
        exit(1);
    }
    for (int i = 0; i < weightsParam.size(); i++) {
        neuralNetwork[layerIndex][i].setWeights(weightsParam[i]);
    }
}