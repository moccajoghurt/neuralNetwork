#include <iostream>
#include <cmath>
#include <windows.h>
#include "neuralNetwork.h"

using namespace std;

void layerCreationTest() {
    Network n;
    n.createLayer(4);
    n.createLayer(6);
    n.createLayer(8);

    if (n.getNetwork().size() != 3) {
        cout << "layerCreationTest() failed: incorrect layer count" << endl;
    }
    int expectedWeightCount = 4*4 + 6*4 + 8*6;
    int weightCount = 0;
    for (vector<Neuron> l : n.getNetwork()) {
        for (Neuron n : l) {
            weightCount += n.getWeights().size();
        }
    }
    if (weightCount != expectedWeightCount) {
        cout << weightCount << " " << 4*4 + 6*4 + 8*6 << endl;
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
    n.createLayer(2);

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

    n.forwardPropagate(testInput);
    vector<vector<Neuron> > propagationResult = n.getNetwork();
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

void printNeuralNetworkOutputs(vector<vector<Neuron> > network) {
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

void printNeuralNetwork(vector<vector<Neuron> > network) {
    cout << "#################################### printing neural network weights:" << endl;
    int layerCount = 0;
    for (vector<Neuron> nv : network) {
        int neuronCount = 0;
        cout << "Layer_" << layerCount << endl;
        for (Neuron ne : nv) {
            cout << "Neuron_" << neuronCount<< "_weights: ";
            for (int n = 0; n < ne.getWeights().size(); n++) {
                cout << ne.getWeights()[n] << "\t";
            }
            cout << endl;
            neuronCount++;
        }
        cout << endl;
        layerCount++;
    }
    cout.flush();
}

void backpropagationCalculatesCorrectValueTest() {
    Network n;
    n.createLayer(2);
    n.createLayer(2);

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

    vector<float> trainData;
    trainData.push_back(0.01);
    trainData.push_back(0.99);
    n.forwardPropagate(inputLayer);
    n.backPropagate(trainData);

    if (abs(neurons[0][0].getWeights()[0] - 0.149) < 0.001 &&
        abs(neurons[0][0].getWeights()[1] - 0.199) < 0.001 &&
        abs(neurons[0][1].getWeights()[0] - 0.249) < 0.001 &&
        abs(neurons[0][1].getWeights()[1] - 0.299) < 0.001 &&
        abs(neurons[1][0].getWeights()[0] - 0.358) < 0.001 &&
        abs(neurons[1][0].getWeights()[1] - 0.408) < 0.001 &&
        abs(neurons[1][1].getWeights()[0] - 0.511) < 0.001 &&
        abs(neurons[1][1].getWeights()[1] - 0.561) < 0.001) {
        cout << "backpropagationCalculatesCorrectValueTest() success" << endl;
    } else {
        cout << "backpropagationCalculatesCorrectValueTest() failed" << endl;
    }
}

void backpropagationIsImprovingNetworkOutputTest() {
    int layerNum = rand() % 10 + 1;
    // Network n;
    for (int i = 0; i < layerNum; i++) {
        // int 
    }
    Network n;
    n.createLayer(3);
    n.createLayer(3);
    n.createLayer(1);

    vector<float> testInput0;
    testInput0.push_back(0);
    testInput0.push_back(0);
    testInput0.push_back(0);
    vector<float> train0;
    train0.push_back(0);

    vector<float> testInput1;
    testInput1.push_back(1);
    testInput1.push_back(0);
    testInput1.push_back(0);
    vector<float> train1;
    train1.push_back(1);

    vector<float> testInput2;
    testInput2.push_back(1);
    testInput2.push_back(1);
    testInput2.push_back(0);
    vector<float> train2;
    train2.push_back(0);

    vector<float> testInput3;
    testInput3.push_back(1);
    testInput3.push_back(1);
    testInput3.push_back(1);
    vector<float> train3;
    train3.push_back(1);

    for (int i = 0; i < 2000; i++) {
        n.forwardPropagate(testInput0);
        n.backPropagate(train0);
        n.forwardPropagate(testInput1);
        n.backPropagate(train1);
        n.forwardPropagate(testInput2);
        n.backPropagate(train2);
        n.forwardPropagate(testInput3);
        n.backPropagate(train3);
    }
    n.forwardPropagate(testInput0);
    printNeuralNetworkOutputs(n.getNetwork());

}

int main() {
    layerCreationTest();
    neuronCalculatesCorrectOutputLogisticTest();
    forwardPropagationCalculatesCorrectValueLogisticTest();
    backpropagationCalculatesCorrectValueTest();
    backpropagationIsImprovingNetworkOutputTest();
    //todo: fowardpropagation mit festen Gewichten wie in backpropagation testen (mit Tutorial-Ausgaben vergleichen)
}