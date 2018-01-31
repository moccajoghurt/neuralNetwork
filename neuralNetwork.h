#include <vector>
#include <cstdlib> // srand, rand
#include <ctime>

using namespace std;

class Neuron {
public:
    Neuron();
    void initWeights(int inputCount);
    int getWeightCount() {
        return this->weights.size();
    }
    float getLastSum() {
        return lastCalculatedWeightedInputSum;
    }
    float getLastCalculatedOutput() {
        return lastCalculatedOutput;
    }
    float getBias() {
        return bias;
    }
    float getOutput(vector<float> input);
    vector<float> getLastInput() {
        return lastInput;
    }
    float getLearningRate() {
        return learningRate;
    }
    void updateWeight(int index, float newVal);
    /******************* FUNCTIONS FOR UNIT TESTS */
    vector<float> getWeights() {
        return weights;
    }
    float getBiasWeight() {
        return biasWeight;
    }
    void setBiasWeight(float weight) {
        biasWeight = weight;
    }
    void setWeights(vector<float> weights);
    /*******************/
private:
    vector<float> weights;
    float bias;
    float biasWeight;
    float simplifiedSigmoidFunc(float val);
    float logisticFunc(float val);
    float lastCalculatedWeightedInputSum;
    float lastCalculatedOutput;
    vector<float> lastInput;
    float learningRate;
};


class Network {
public:
    Network() {
        srand (static_cast <unsigned> (time(0)));
    }
    void createLayer(int neuronCount, int neuronInputsCount);
    void forwardPropagate(vector<float> input);
    void backPropagate(vector<float> trainingValues);
    float derivateSimplifiedSigmoidFunc(float value);
    float derivateLogisticFunc(float value);
    /******************* FUNCTIONS FOR UNIT TESTS */
    vector<vector<Neuron> >& getNetwork() {
        return neuralNetwork;
    }
    void setWeights(int layerIndex, vector<vector<float> > weightsParam);
    /*******************/
private:
    vector<vector<Neuron> > neuralNetwork;
};