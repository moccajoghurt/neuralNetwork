#include <iostream>
#include <cmath>
#include <windows.h>
#include "neuralNetwork.h"

using namespace std;

void layerCreationTest() {
    Network n;
    n.createLayer(4, 5);
    n.createLayer(6, 7);
    n.createLayer(8, 9);

    if (n.getNetwork().size() != 3) {
        cout << "layerCreationTest() failed: incorrect layer count" << endl;
    }
    int expectedWeightCount = 4*5 + 6*7 + 8*9;
    int weightCount = 0;
    for (vector<Neuron> l : n.getNetwork()) {
        for (Neuron n : l) {
            weightCount += n.getWeights().size();
        }
    }
    if (weightCount != expectedWeightCount) {
        cout << "layerCreationTest() failed: incorrect weight count" << endl;
    } else {
        cout << "layerCreationTest() success" << endl;
    }
}

bool correctNeuronValueLogistic(Network& n, float f1, float f2) {
    float weight1 = n.getNetwork()[0][0].getWeights()[0];
    float weight2 = n.getNetwork()[0][0].getWeights()[1];
    float bias = n.getNetwork()[0][0].getBias();
    float biasWeight = n.getNetwork()[0][0].getBiasWeight();
    float expectedResult = (weight1 * f1) + (weight2 * f2) + (bias * biasWeight);
    expectedResult = 1. / (1 + exp(-expectedResult)); // apply logistic func
    vector<float> testInput;
    testInput.push_back(f1);
    testInput.push_back(f2);
    float neuronResult = n.getNetwork()[0][0].getOutput(testInput);
    if (abs(neuronResult - expectedResult) < 0.001) {
        return true;
    } else {
        return false;
    }
}

void neuronCalculatesCorrectOutputLogisticTest() {
    Network n;
    n.createLayer(1, 2);
    for (int i = 0; i < 1000; i++) {
        float rand1 = static_cast<float>(rand());
        float rand2 = static_cast<float>(rand());
        if (!correctNeuronValueLogistic(n, -rand1, rand2)) {
            cout << "neuronCalculatesCorrectOutputTest() failed" << endl;
            return;
        }
    }
    cout << "neuronCalculatesCorrectOutputTest() success" << endl;
}

void forwardPropagationCalculatesCorrectValueLogisticTest() {
    Network n;
    n.createLayer(2, 4);
    n.createLayer(2, 2);

    vector<float> testInput;
    testInput.push_back(1.3);
    testInput.push_back(0.4);
    testInput.push_back(-5.7);
    testInput.push_back(0.2);

    float layer1Node1Output = n.getNetwork()[0][0].getOutput(testInput);
    float layer1Node2Output = n.getNetwork()[0][1].getOutput(testInput);

    vector<float> layer2InputBuf;
    layer2InputBuf.push_back(layer1Node1Output);
    layer2InputBuf.push_back(layer1Node2Output);
    float layer2Node1Output = n.getNetwork()[1][0].getOutput(layer2InputBuf);
    float layer2Node2Output = n.getNetwork()[1][1].getOutput(layer2InputBuf);

    vector<vector<Neuron> > propagationResult = n.forwardPropagate(testInput);
    if (propagationResult[0][0].getLastCalculatedOutput() != layer1Node1Output ||
        propagationResult[0][1].getLastCalculatedOutput() != layer1Node2Output ||
        propagationResult[1][0].getLastCalculatedOutput() != layer2Node1Output ||
        propagationResult[1][1].getLastCalculatedOutput() != layer2Node2Output) {

        cout << "forwardPropagationCalculatesCorrectValueTest() failed" << endl;

        cout << propagationResult[0][0].getLastCalculatedOutput() << " " << layer1Node1Output << endl;
        cout << propagationResult[0][1].getLastCalculatedOutput() << " " << layer1Node2Output << endl;
        cout << propagationResult[1][0].getLastCalculatedOutput() << " " << layer2Node1Output << endl;
        cout << propagationResult[1][1].getLastCalculatedOutput() << " " << layer2Node2Output << endl;
    } else {
        cout << "forwardPropagationCalculatesCorrectValueTest() success" << endl;
    }
}

void printNeuralNetwork(vector<vector<Neuron> > network) {
    int layerCount = 1;
    for (vector<Neuron> nv : network) {
        cout << "Layer " << layerCount << " outputs: ";
        for (Neuron ne : nv) {
            cout << ne.getLastCalculatedOutput() << "\t";
        }
        cout << endl;
        layerCount++;
    }
}

void backpropagationTest() {
    Network n;
    n.createLayer(2,2);
    n.createLayer(2,2);

    vector<vector<float> > weightsLayer1;
    vector<float> testWeightsNeuron1;
    testWeightsNeuron1.push_back(0.15);
    testWeightsNeuron1.push_back(0.2);
    vector<float> testWeightsNeuron2;
    testWeightsNeuron2.push_back(0.25);
    testWeightsNeuron2.push_back(0.3);

    vector<vector<float> > weightsLayer2;
    vector<float> testWeightsNeuron1_2;
    testWeightsNeuron1_2.push_back(0.4);
    testWeightsNeuron1_2.push_back(0.45);
    vector<float> testWeightsNeuron2_2;
    testWeightsNeuron2_2.push_back(0.5);
    testWeightsNeuron2_2.push_back(0.55);

    weightsLayer1.push_back(testWeightsNeuron1);
    weightsLayer1.push_back(testWeightsNeuron2);

    weightsLayer2.push_back(testWeightsNeuron1_2);
    weightsLayer2.push_back(testWeightsNeuron2_2);

    n.setWeights(0, weightsLayer1);
    n.setWeights(1, weightsLayer2);

    vector<vector<Neuron> >& neurons = n.getNetwork();
    
    vector<float> inputLayer;
    inputLayer.push_back(0.05);
    inputLayer.push_back(0.1);

    neurons[0][0].setBiasWeight(0.35);
    neurons[0][1].setBiasWeight(0.35);
    neurons[1][0].setBiasWeight(0.6);
    neurons[1][1].setBiasWeight(0.6);

    n.forwardPropagate(inputLayer);
    neurons = n.getNetwork();
    // printNeuralNetwork(neurons);

    vector<float> trainData;
    trainData.push_back(0.01);
    trainData.push_back(0.99);
    n.backPropagate(trainData);
}



int main() {
    layerCreationTest();
    neuronCalculatesCorrectOutputLogisticTest();
    forwardPropagationCalculatesCorrectValueLogisticTest();

    backpropagationTest();
    cout.flush();
    //todo: fowardpropagation mit festen Gewichten wie in backpropagation testen (mit Tutorial-Ausgaben vergleichen)
}