#include <iostream>
#include <cmath>
#include <windows.h>
#include "neuralNetwork.h"

using namespace std;

void layerCreationTest() {
    Network n;
    n.createLayer(4, 4);
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

bool correctNeuronValue(Network& n, float f1, float f2) {
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

void neuronCalculatesCorrectOutputTest() {
    Network n;
    n.createLayer(1, 2);
    for (int i = 0; i < 1000; i++) {
        float rand1 = static_cast<float>(rand());
        float rand2 = static_cast<float>(rand());
        if (!correctNeuronValue(n, -rand1, rand2)) {
            cout << "neuronCalculatesCorrectOutputTest() failed" << endl;
            return;
        }
    }
    cout << "neuronCalculatesCorrectOutputTest() success" << endl;
}

void forwardPropagationCalculatesCorrectValueTest() {
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
    n.createLayer(2, 2);
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

bool backpropagationImprovesNetworkOutputTestProc() {
    Network n;
    int layerNum = rand() % 3 + 1;
    int inputValueCount = rand() % 100 + 1;
    n.createLayer(rand() % 50, inputValueCount); // the first layer must define the number of inputs
    for (int i = 0; i < layerNum; i++) {
        int neuronNum = rand() % 50;
        n.createLayer(neuronNum);
    }
    vector<float> inputValues;
    for (int i = 0; i < inputValueCount; i++) {
        float randInput = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        inputValues.push_back(randInput);
    }
    vector<float> trainValues;
    int networkSize = n.getNetwork().size();
    for (int i = 0; i < n.getNetwork()[networkSize-1].size(); i++) {
        float randInput = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        trainValues.push_back(randInput);
    }

    vector<float> untrainedOutput = n.forwardPropagate(inputValues);
    for (int i = 0; i < 10; i++) {
        n.backPropagate(trainValues);
        n.forwardPropagate(inputValues);
    }
    vector<float> trainedOutput = n.forwardPropagate(inputValues);
    bool improved = false;
    for (int i = 0; i < untrainedOutput.size(); i++) {
        if (abs(trainValues[i] - untrainedOutput[i]) > abs(trainValues[i] - trainedOutput[i])) {
            improved = true;
        }
    }
    if (improved) {
        return true;
    } else {
        return false;
    }
}

void backpropagationImprovesNetworkOutputTest() {
    // if the output doesn't improve 10 times in a row, we might have a problem. Otherwise it's probably RNG
    bool updated = false;
    for (int i = 0; i < 10; i++) {
        if (backpropagationImprovesNetworkOutputTestProc()) {
            updated = true;
        }
    }
    if (updated) {
        cout << "backpropagationIsImprovingNetworkOutputTest() success" << endl;
    } else {
        cout << "backpropagationIsImprovingNetworkOutputTest() failed" << endl;
    }
}

bool backpropagationUpdatesEachLayerTestProc() {
    Network n;
    int layerNum = rand() % 3 + 1;
    int inputValueCount = rand() % 1000 + 1;
    n.createLayer(rand() % 15, inputValueCount); // the first layer must define the number of inputs
    for (int i = 0; i < layerNum; i++) {
        int neuronNum = rand() % 15;
        n.createLayer(neuronNum);
    }
    vector<float> inputValues0;
    vector<float> inputValues1;
    vector<float> inputValues2;
    for (int i = 0; i < inputValueCount; i++) {
        float randInput0 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        float randInput1 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        float randInput2 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        inputValues0.push_back(randInput0);
        inputValues1.push_back(randInput1);
        inputValues2.push_back(randInput2);
    }
    vector<float> trainValues0;
    vector<float> trainValues1;
    vector<float> trainValues2;
    int networkSize = n.getNetwork().size();
    for (int i = 0; i < n.getNetwork()[networkSize-1].size(); i++) {
        float randInput0 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        float randInput1 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        float randInput2 = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        trainValues0.push_back(randInput0);
        trainValues1.push_back(randInput1);
        trainValues2.push_back(randInput2);
    }
    vector<vector<Neuron> > oldNetwork;
    for (int i = 0; i < n.getNetwork().size(); i++) {
        oldNetwork.push_back(n.getNetwork()[i]);
    }
    
    for (int i = 0; i < 10; i++) {
        n.forwardPropagate(inputValues0);
        n.backPropagate(trainValues0);
        n.forwardPropagate(inputValues1);
        n.backPropagate(trainValues1);
        n.forwardPropagate(inputValues2);
        n.backPropagate(trainValues2);
    }
    vector<vector<Neuron> > newNetwork = n.getNetwork();
    //iterate over layers
    bool allLayersGotUpdated = true;
    for (int i = 0; i < oldNetwork.size(); i++) {
        bool layerGotUpdated = false;
        // iterate over neurons
        for (int n = 0; n < oldNetwork[i].size(); n++) {
            // iterate over weights
            for (int m = 0; m < oldNetwork[i][n].getWeightCount(); m++) {
                if (oldNetwork[i][n].getWeights()[m] != newNetwork[i][n].getWeights()[m]) {
                    layerGotUpdated = true;
                }
            }
        }
        if (!layerGotUpdated) {
            // cout << "Layer_" << i << " (of " << n.getNetwork().size() << ") didn't get updated" << endl;
            allLayersGotUpdated = false;
        }
    }
    if (!allLayersGotUpdated) {
        return false;
    } else {
        return true;
    }
}

void backpropagationUpdatesEachLayerTest() {
    // if layers don't get updated 10 times in a row, we might have a problem. Otherwise it's probably RNG
    bool updated = false;
    for (int i = 0; i < 10; i++) {
        if (backpropagationUpdatesEachLayerTestProc()) {
            updated = true;
        }
    }
    if (updated) {
        cout << "backpropagationUpdatesAllWeightsTest() success" << endl;
    } else {
        cout << "backpropagationUpdatesAllWeightsTest() failed ";
    }
}



void backpropagationPlayfield() {
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
    neuronCalculatesCorrectOutputTest();
    forwardPropagationCalculatesCorrectValueTest();
    backpropagationCalculatesCorrectValueTest();
    backpropagationImprovesNetworkOutputTest();
    backpropagationUpdatesEachLayerTest();
    //todo: fowardpropagation mit festen Gewichten wie in backpropagation testen (mit Tutorial-Ausgaben vergleichen)
}