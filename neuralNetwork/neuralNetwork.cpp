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

void Neuron::setWeights(vector<float> weightsParam) {
    if (weightsParam.size() != weights.size() && weights.size() != 0) {
        cout << "Neuron::setWeights(): inputNum != weightNum" << endl;
        exit(1);
    }
    weights = weightsParam;
}

/////////////////////////////////////////// class Network
void Network::createLayer(int neuronCount, int neuronInputsCount) {
    
    if (neuronInputsCount < 0 && neuralNetwork.size() == 0) {
        cout << "The first layer must define the number of inputs!" << endl;
        exit(1);
    } else if (neuronInputsCount < 0) {
        neuronInputsCount = neuralNetwork[neuralNetwork.size() - 1].size();
    } else if (neuronInputsCount > 0 && neuralNetwork.size() != 0) {
        cout << "The neural network is fully connected, so only the first layer can have a manual neuron-input-size." << endl;
        exit(1);
    }

    vector<Neuron> neuronLayer;
    for (int i = 0; i < neuronCount; i++) {
        Neuron n;
        n.initWeights(neuronInputsCount);
        neuronLayer.push_back(n);
    }
    neuralNetwork.push_back(neuronLayer);
}
void Network::forwardPropagate(vector<float> input) {
    vector<vector<Neuron> > updatedNetwork;
    vector<vector<float> > layerResults;
    int layerCount = 0;
    for (int i = 0; i < neuralNetwork.size(); i++) {
        int neuronCount = neuralNetwork[i].size();
        if (i == 0) {
            // we are in the first layer
            vector<float> neuronResults;
            for (int n = 0; n < neuronCount; n++) {
                neuronResults.push_back(neuralNetwork[i][n].getOutput(input));
            }
            layerResults.push_back(neuronResults);
            updatedNetwork.push_back(neuralNetwork[i]);
            layerCount++;
            continue;
        }
        
        vector<float> neuronResults;
        for (int n = 0; n < neuronCount; n++) {
            // It's a fully connected Network, so feed each Neuron with all results of the previous layer
            float neuronOutput = neuralNetwork[i][n].getOutput(layerResults[layerCount - 1]);
            neuronResults.push_back(neuronOutput);
        }
        layerResults.push_back(neuronResults);
        updatedNetwork.push_back(neuralNetwork[i]);
        layerCount++;
    }
    neuralNetwork = updatedNetwork;
    // return layerResults[layerResults.size() - 1];
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
// todo: currently the bias is not getting updated. check out if it's necessary
void Network::backPropagate(vector<float> trainingValues) {
    if (neuralNetwork.back().size() != trainingValues.size()) {
        cout << "error: Training data must have the same number of values as the neurons in the output layer." << endl;
        exit(1);
    }

    vector<vector<Neuron> > updatedNetwork;

    vector<float> totalErrorWrtOutputs;
    vector<float> outputWrtTotalNetInputs;
    // iterate over layers
    for (int i = neuralNetwork.size() - 1; i >= 0; i--) {
        vector<Neuron> newNeurons;

        vector<float> totalErrorWrtOutputsBuf;
        vector<float> outputWrtTotalNetInputsBuf;
        if (i == neuralNetwork.size() - 1) {
            // we are in the Output Layer
            float totalError = 0;
            // calculate the total error
            for (int n = 0; n < neuralNetwork[i].size(); n++) {
                float deltaSum = trainingValues[n] - neuralNetwork[i][n].getLastCalculatedOutput();
                float localError = (0.5 * deltaSum) * (0.5 * deltaSum);
                totalError += localError;
            }
            //iterate over neurons
            for (int n = 0; n < neuralNetwork[i].size(); n++) {
                Neuron newNeuron;

                vector<float> lastInputs = neuralNetwork[i][n].getLastInput();
                float totalErrorWrtOutput = -(trainingValues[n] - neuralNetwork[i][n].getLastCalculatedOutput()); // Wrt = with respect to
                float outputWrtTotalNetInput = derivateLogisticFunc(neuralNetwork[i][n].getLastCalculatedOutput());
                totalErrorWrtOutputsBuf.push_back(totalErrorWrtOutput);
                outputWrtTotalNetInputsBuf.push_back(outputWrtTotalNetInput);
                vector<float> neuronWeights = neuralNetwork[i][n].getWeights();
                // iterate over weights
                vector<float> newWeights;
                for (int x = 0; x < neuronWeights.size(); x++) {
                    float totalErrorWrtWeight = totalErrorWrtOutput * outputWrtTotalNetInput * lastInputs[x];
                    float newWeight = neuronWeights[x] - neuralNetwork[i][n].getLearningRate() * totalErrorWrtWeight;
                    newWeights.push_back(newWeight);
                }
                newNeuron.setWeights(newWeights);
                newNeurons.push_back(newNeuron);
            }
            vector<vector<Neuron> >::iterator it = updatedNetwork.begin();
            updatedNetwork.insert(it, newNeurons);
            totalErrorWrtOutputs = totalErrorWrtOutputsBuf;
            outputWrtTotalNetInputs = outputWrtTotalNetInputsBuf;
            continue;
        }
        // we are in a hidden layer

        // iterate over neurons
        for (int n = 0; n < neuralNetwork[i].size(); n++) {
            Neuron newNeuron;

            float totalErrorWrtHiddenOutput = 0;
            // iterate over the neurons of the layer to the right (leftmost = inputlayer, rightmost = outputlayer)
            for (int x = 0; x < neuralNetwork[i + 1].size(); x++) {
                float errorOutputWrtNetOutput = totalErrorWrtOutputs[x] * outputWrtTotalNetInputs[x];
                float netOutputWrtHiddenNeuronOutput = neuralNetwork[i + 1][x].getWeights()[n];
                float errorOutputWrtHiddenNeuronOutput = netOutputWrtHiddenNeuronOutput * errorOutputWrtNetOutput;
                totalErrorWrtHiddenOutput += errorOutputWrtHiddenNeuronOutput;
            }
            float lastCalcBuf = neuralNetwork[i][n].getLastCalculatedOutput();
            float hiddenOutputWrtHiddenNet = lastCalcBuf * (1 - lastCalcBuf); // if lastCalcBuf is 1 (which happens of the input to the neurons is too high, the new weight will be multiplied by 0)
            hiddenOutputWrtHiddenNet = hiddenOutputWrtHiddenNet == 0 ? 0.00001 : hiddenOutputWrtHiddenNet;
            totalErrorWrtOutputsBuf.push_back(totalErrorWrtHiddenOutput);
            outputWrtTotalNetInputsBuf.push_back(hiddenOutputWrtHiddenNet);
            // iterate over weights
            vector<float> newWeights;
            for (int y = 0; y < neuralNetwork[i][n].getWeights().size(); y++) {
                float hiddenNetWrtWeight = neuralNetwork[i][n].getLastInput()[y];
                float totalErrorWrtWeight = totalErrorWrtHiddenOutput * hiddenOutputWrtHiddenNet * hiddenNetWrtWeight;
                float newWeight = neuralNetwork[i][n].getWeights()[y] - neuralNetwork[i][n].getLearningRate() * totalErrorWrtWeight;
                newWeights.push_back(newWeight);
            }
            newNeuron.setWeights(newWeights);
            newNeurons.push_back(newNeuron);
        }
        vector<vector<Neuron> >::iterator it = updatedNetwork.begin();
        updatedNetwork.insert(it, newNeurons);
        totalErrorWrtOutputs = totalErrorWrtOutputsBuf;
        outputWrtTotalNetInputs = outputWrtTotalNetInputsBuf;
    }
    neuralNetwork = updatedNetwork;
}

void Network::setWeights(int layerIndex, vector<vector<float> > weightsParam) {
    if (weightsParam.size() != neuralNetwork[layerIndex].size()) {
        cout << "Network::setWeights(): inputNum != weightNum" << endl;
        exit(1);
    }
    for (int i = 0; i < weightsParam.size(); i++) {
        neuralNetwork[layerIndex][i].setWeights(weightsParam[i]);
    }
}

vector<float> Network::getOutputLayerResults() {
    vector<float> outputLayerResults;
    int layerCount = neuralNetwork.size();
    // iterate over neurons
    for (int i = 0; i < neuralNetwork[layerCount - 1].size(); i++) {
        outputLayerResults.push_back(neuralNetwork[layerCount - 1][i].getLastCalculatedOutput());
    }
    return outputLayerResults;
}