# Genetic Algorithm Neural Network for MNIST

[![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)](https://isocpp.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-Parallel-green.svg)](https://www.openmp.org/)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com)

> A high-performance implementation of Genetic Algorithm optimization for Neural Networks, specifically designed for MNIST digit classification with OpenMP parallelization.

## Table of Contents

- [ Features](#-features)
- [Architecture](#️-architecture)
- [ Prerequisites](#-prerequisites)
- [ Installation](#-installation)
- [ Usage](#-usage)
- [ Performance](#-performance)
- [ Genetic Operations](#-genetic-operations)
- [ Results](#-results)
- [ Contributing](#-contributing)
- [ License](#-license)

##  Features

 **Key Highlights:**
-  **Neural Network**: 2-layer feedforward network (784→128→10)
-  **Genetic Algorithm**: Evolution-based weight optimization
-  **OpenMP Parallelization**: Multi-threaded performance boost
-  **MNIST Dataset**: Handwritten digit recognition
-  **Multiple Mutation Strategies**: Gaussian addition/subtraction with crossover
- **Real-time Performance Tracking**: Generation-by-generation accuracy monitoring

## Architecture

```
Input Layer (784) → Hidden Layer (128) → Output Layer (10)
                  ↓
               Sigmoid        Softmax
```

## 🔧 Prerequisites

Before you begin, ensure you have:

-  **C++ Compiler**: GCC 7.0+ or Clang 5.0+ with C++17 support
-  **OpenMP**: For parallel processing capabilities
-  **MNIST Dataset**: CSV format (`mnist.csv`)

###  System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU Cores | 2 | 4+ |
| Storage | 100MB | 500MB |

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/genetic-neural-network.git
cd genetic-neural-network
```

### Prepare MNIST Dataset
Download the MNIST dataset in CSV format and place it as `mnist.csv` in the project directory.

### Compile the Code

#### With OpenMP (Recommended)
```bash
g++ -std=c++17 -fopenmp -O3 -o genetic_nn_parallel genetic_nn_parallel.cpp
```

#### 📱 Sequential Version
```bash
g++ -std=c++17 -O3 -o genetic_nn_sequential genetic_nn_sequential.cpp
```

## Usage

###  Run with Parallelization
```bash
./genetic_nn_parallel
```

### Run Sequential Version
```bash
./genetic_nn_sequential
```

### Configuration

Modify these parameters in the source code:

```cpp
const int organisms = 40;      // Population size
int generations = 40;          // Number of generations
int hiddenSize = 128;          // Hidden layer neurons
```

##  Performance

### Speed Comparison

| Version | Population | Generations | Avg. Time/Generation |
|---------|------------|-------------|---------------------|
| Sequential | 10 | 10 | ~15-20 seconds |
| Parallel (4 cores) | 40 | 40 | ~8-12 seconds |

### Typical Results
-  **Initial Accuracy**: ~10-15%
-  **Final Accuracy**: ~60-80% (varies by run)
-  **Convergence**: Usually within 20-30 generations

##  Genetic Operations

###  Selection Strategy
- **Elite Selection**: Top 2 performers always survive
- **Crossover**: Weight mixing between top 8 organisms
- **Tournament Selection**: Random selection from top performers

### Mutation Types

| Operation | Description | Probability |
|-----------|-------------|-------------|
| **Crossover** | Mix weights between two parents | 20% |
| **Add Gaussian** | Add random noise to weights | 20% |
| **Sub Gaussian** | Subtract noise + multiply | 20% |
| **Crossover + Add** | Crossover followed by addition | 20% |
| **Crossover + Sub** | Crossover followed by subtraction | 20% |

## Results

### Sample Output
```
####Details####
Total Organisms : 40
Generations : 40

MNIST dataset loaded successfully!
Number of samples: 60000

______Generation 0_____
Organism 0 Accuracy: 12.34
Organism 1 Accuracy: 15.67
...
Top Accuracy : 78.45

Time taken by generation : 8 seconds
```

### Performance Visualization
The algorithm typically shows:
- **Rapid Initial Growth**: 10% → 40% in first 10 generations
-  **Steady Improvement**: 40% → 70% in next 20 generations  
-  **Plateau Phase**: Final 10 generations for fine-tuning


##  Code Structure

```
├── genetic_nn_parallel.cpp    # OpenMP parallelized version
├── genetic_nn_sequential.cpp  # Sequential version
├── mnist.csv                  # MNIST dataset (not included)
└── README.md                  # This file
```

## Key Classes and Functions

### `SimpleANN` Class
- **Constructor**: Initialize network with random weights
- **`forwardPropagation()`**: Calculate network output
- **`calculateAccuracy()`**: Evaluate performance on dataset
- **`add_gaussian()`**: Add mutation noise
- **`sub_gaussian()`**: Subtract and multiply mutation

###  Genetic Functions
- **`compareByPerformance()`**: Sort organisms by fitness
- **Selection Logic**: Choose parents for next generation
- **Crossover Operations**: Mix genetic material

##  Troubleshooting

### Common Issues:

**Compilation Error**: `fatal error: omp.h: No such file`
```bash
# Install OpenMP
sudo apt-get install libomp-dev  # Ubuntu/Debian
brew install libomp              # macOS
```

**Dataset Not Found**: `Error: Unable to open the CSV file`
- Ensure `mnist.csv` is in the project directory
- Check file permissions

**Poor Performance**: Accuracy stuck below 20%
- Try different random seeds
- Adjust mutation rates
- Increase population size or generations

## Educational Value

This project demonstrates:
-  **Genetic Algorithms**: Population-based optimization
- **Neural Networks**: Feedforward architecture
-  **Parallel Computing**: OpenMP implementation
- **Machine Learning**: Classification problem solving
-  **Evolutionary Computing**: Bio-inspired algorithms


Made with ❤️ by NiceGuy

</div>
