#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <array>
#include <omp.h>

using namespace std;
using namespace std::chrono;

struct MnistData {
    vector<float> pixels;
    int label;
};

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

vector<float> softmax(const vector<float>& input) {
    vector<float> expValues(input.size());
    float sumExpValues = 0.0f;

    for (size_t i = 0; i < input.size(); i++) {
        expValues[i] = exp(input[i]);
        sumExpValues += expValues[i];
    }

    for (size_t i = 0; i < input.size(); i++) {
        expValues[i] /= sumExpValues;
    }

    return expValues;
}

class SimpleANN {
public:
    SimpleANN() {}
    float performance;
    SimpleANN(int inputSize, int hiddenSize, int outputSize,float mean, float std_dev)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) ,mean(mean),std_dev(std_dev){
        // Initialize weights and biases (randomly or with custom values)
        weights1.resize(inputSize * hiddenSize);
        biases1.resize(hiddenSize);
        weights2.resize(hiddenSize * outputSize);
        biases2.resize(outputSize);

        // Example: Set weights and biases to 0.5f
        random_device rd;
        default_random_engine generator(rd());
        normal_distribution<float> distribution(mean, std_dev);

        #pragma omp parallel for
        for (int i = 0; i < inputSize * hiddenSize; i++)
            weights1[i] = distribution(generator);

        #pragma omp parallel for
        for (int i = 0; i < hiddenSize; i++)
            biases1[i] = distribution(generator);

        #pragma omp parallel for
        for (int i = 0; i < hiddenSize * outputSize; i++)
            weights2[i] = distribution(generator);

        #pragma omp parallel for
        for (int i = 0; i < outputSize; i++)
            biases2[i] = distribution(generator);
    }

    void add_gaussian()
    {
        random_device rd;
        default_random_engine generator(rd());
        normal_distribution<float> distribution(mean, std_dev);

        #pragma omp parallel for
        for (int i = 0; i < inputSize * hiddenSize; i++)
            weights1[i] += distribution(generator);

        #pragma omp parallel for
        for (int i = 0; i < hiddenSize; i++)
            biases1[i] += distribution(generator);

        #pragma omp parallel for
        for (int i = 0; i < hiddenSize * outputSize; i++)
            weights2[i] += distribution(generator);

        #pragma omp parallel for
        for (int i = 0; i < outputSize; i++)
            biases2[i] += distribution(generator);
    }
    void sub_gaussian()
    {
        random_device rd;
        default_random_engine generator(rd());
        normal_distribution<float> distribution(mean, std_dev);

        #pragma omp parallel for
        for (int i = 0; i < inputSize * hiddenSize; i++) {
            weights1[i] -= distribution(generator);
            weights1[i] *= distribution(generator);
        }

        #pragma omp parallel for
        for (int i = 0; i < hiddenSize; i++)
            biases1[i] -= distribution(generator);

        #pragma omp parallel for
        for (int i = 0; i < hiddenSize * outputSize; i++) {
            weights2[i] -= distribution(generator);
            weights2[i] *= distribution(generator);
        }

        #pragma omp parallel for
        for (int i = 0; i < outputSize; i++)
            biases2[i] -= distribution(generator);
    }

    void forwardPropagation(const MnistData& data, vector<float>& output) {
        // Layer 1
        vector<float> layer1(hiddenSize);
        for (int i = 0; i < hiddenSize; i++) {
            float value = biases1[i];
            for (int j = 0; j < inputSize; j++) {
                value += weights1[i * inputSize + j] * data.pixels[j];
            }
            layer1[i] = sigmoid(value);
        }

        // Layer 2
        output.resize(outputSize);
        for (int i = 0; i < outputSize; i++) {
            float value = biases2[i];
            for (int j = 0; j < hiddenSize; j++) {
                value += weights2[i * hiddenSize + j] * layer1[j];
            }
            output[i] = value;
        }

        // Softmax
        output = softmax(output);
    }

    float calculateAccuracy(const vector<MnistData>& dataSet) {
        int correctCount = 0;
        for (const auto& data : dataSet) {
            vector<float> output(outputSize);
            forwardPropagation(data, output);
            int predictedLabel = distance(output.begin(), max_element(output.begin(), output.end()));
            if (predictedLabel == data.label) {
                correctCount++;
            }
        }
        performance = static_cast<float>(correctCount) / dataSet.size();
        return static_cast<float>(correctCount) / dataSet.size();
    }

public:
    int inputSize, hiddenSize, outputSize,mean,std_dev;
    vector<float> weights1, biases1, weights2, biases2;
};

bool compareByPerformance(const SimpleANN& obj1, const SimpleANN& obj2) {
    return obj1.performance > obj2.performance;
}

int main() {

    const int organisms = 40;
    int generations = 40;

    cout<<"####Details####\nTotal Organisms : "<<organisms<<"\nGenerations : "<<generations<<endl;


    string csvFile = "mnist.csv"; // Replace with the path to your CSV file
    vector<MnistData> mnistDataSet;
    array<SimpleANN, organisms> objects;
    array<SimpleANN, organisms> next_generation;
    vector<int> choices = {0,1, 2, 3,4,5,6,7,8,9};
    vector<double> weights = {0.15353399, 0.13818059, 0.12436253, 0.11192628, 0.10073365, 0.09066029,0.08159426, 0.07343483, 0.06609135 ,0.05948221};

    ifstream file(csvFile);

    if (!file.is_open()) {
        cerr << "Error: Unable to open the CSV file." << endl;
        return 1;
    }

    string line;
    MnistData data;

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;

        // Read the label first
        getline(ss, cell, ',');
        data.label = stoi(cell);

        // Read the pixel values
        data.pixels.clear();
        while (getline(ss, cell, ',')) {
            float pixelValue = stof(cell);
            data.pixels.push_back(pixelValue);
        }

        mnistDataSet.push_back(data);
    }

    file.close();

    cout << "MNIST dataset loaded successfully!" << endl;
    cout << "Number of samples: " << mnistDataSet.size() << endl;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(0.0f, 9.0f);
    uniform_int_distribution<int> random_choice(0, 4);
    uniform_int_distribution<int> select(0,7);

    for(int i=0; i<organisms; i++){
        objects[i] = SimpleANN(784, 128, 10, 0.0f, dist(gen));
    }

    auto start = high_resolution_clock::now();
    for(int i=0; i<organisms; i++){
        float accuracy = objects[i].calculateAccuracy(mnistDataSet);
        cout << "Organism "<<i<<" Accuracy: " << accuracy * 100 << endl;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken for eval without parallelization: " << duration.count()/1000000 << " seconds" << endl;

    int choice;
    int good1,good2;
    for (int j=0; j<generations; j++)
    {
        auto start = high_resolution_clock::now();
        cout << "______Generation " << j << "_____" << endl;
        #pragma omp parallel for
        for(int i=0; i<organisms; i++)
        {
            try
            {
                float accuracy = objects[i].calculateAccuracy(mnistDataSet);
                cout << "Organism "<<i<<" Accuracy: " << accuracy * 100 << endl;
            }
            catch(exception& e)
            {
                cout << "Some error.." << endl;
            }
        }

        sort(objects.begin(), objects.end(), compareByPerformance);
        cout << "Top Accuracy : "<<objects[0].performance*100<<endl;
        next_generation[0] = objects[0];
        next_generation[1] = objects[1];

        next_generation[2] = objects[0];
        next_generation[2].weights2 = objects[1].weights2;
        next_generation[2].calculateAccuracy(mnistDataSet);

        next_generation[3] = objects[1];
        next_generation[3].weights2 = objects[0].weights2;

        #pragma omp parallel for
        for(int i=4; i<organisms; i++)
        {
            choice = random_choice(gen);
            if (choice == 0)
            {
                next_generation[i] = SimpleANN(784, 128, 10, 0.0f, dist(gen));
                good1 = select(gen);
                good2 = select(gen);
                while(good2 == good1)
                {
                    good2 = select(gen);
                }
                next_generation[i].weights1 = objects[good1].weights1;
                next_generation[i].weights2 = objects[good2].weights2;
            }
            else if (choice == 1)
            {
                good1 = select(gen);
                next_generation[i] = objects[good1];
                next_generation[i].add_gaussian();
            }
            else if (choice == 2)
            {
                good1 = select(gen);
                next_generation[i] = objects[good1];
                next_generation[i].sub_gaussian();
            }
            else if (choice == 3)
            {
                next_generation[i] = SimpleANN(784, 128, 10, 0.0f, dist(gen));
                good1 = select(gen);
                good2 = select(gen);
                while(good2 == good1)
                {
                    good2 = select(gen);
                }
                next_generation[i].weights1 = objects[good1].weights1;
                next_generation[i].weights2 = objects[good2].weights2;
                next_generation[i].add_gaussian();
            }
            else if (choice == 4)
            {
                next_generation[i] = SimpleANN(784, 128, 10, 0.0f, dist(gen));
                good1 = select(gen);
                good2 = select(gen);
                while(good2 == good1)
                {
                    good2 = select(gen);
                }
                next_generation[i].weights1 = objects[good1].weights1;
                next_generation[i].weights2 = objects[good2].weights2;
                next_generation[i].sub_gaussian();
            }
        }

        cout << "Passing gen....." << endl;
        for(int i=0; i<objects.size(); i++)
        {
            objects[i] = next_generation[i];
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        cout << "Time taken by generation : " << duration.count()/1000000 << " seconds" << endl;
    }

    return 0;
}
