# GANN: Genetic Algorithm Neural Network for Crypto Trading

GANN (Genetic Algorithm Neural Network) is a program designed to train trading bots using neural networks for cryptocurrency trading on exchanges. The program leverages genetic algorithms to optimize neural networks that make trading decisions based on various indicators.

## Project Description

GANN trains trading bots by dividing their functions into two pairs of neural networks: one pair for buying and one pair for selling:

### Buying Neural Network Pair:

- **Indicator Optimization Network**: The first neural network is trained to select the optimal indicator parameters for buying.
- **Signal Generation Network**: The second neural network is trained to generate a buy signal based on these indicators.

### Selling Neural Network Pair:

- **Indicator Optimization Network**: The first neural network is trained to select the optimal indicator parameters for selling.
- **Signal Generation Network**: The second neural network is trained to generate a sell signal based on these indicators.

## Features

- **Indicator Configuration**: Allows configuring the parameters of indicators used by the buying and selling neural networks.
- **Neural Network Training**: Utilizes genetic algorithms to optimize the neural networks.
- **Saving and Loading Settings**: Supports saving and loading bot configurations and training parameters from files.
- **Bot Operations**:
  - Supports setting up time frames for bot operations.
  - Saves the genes of trained bots for future use.

## Project Structure

- **`parameters/`**: Directory containing configuration files.
  - **`config.yaml`**: Main configuration file for global program settings.
  - **`*.bot`**: Bot configuration files that include neural network parameters and time frames.
  - **`*.gen`**: Files storing the genes of trained bots.

## How to Use

### Configuration Setup

1. Open the `config.yaml` file in the `parameters` directory and adjust the global settings of the program.
2. In the `parameters` directory, create or edit `.bot` files to configure specific bots:
   - Set the neural network parameters for buying and selling.
   - Define the time frames the bots will operate within.

**Requires the `engine` C++ file.**
