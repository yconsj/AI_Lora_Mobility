# AI_LoRa_Mobility

This project investigates the use of AI (reinforcement learning) for LoRa mobility using OMNeT++.  
Python scripts are used to train a model, evaluate it in a low-fidelity Stable-Baselines3 (SB3) environment, export it to a high-fidelity OMNeT++ simulation, and evaluate its performance.
This project uses [OMNeT++ Official Website](https://omnetpp.org/intro/)), [INET](https://inet.omnetpp.org/), [FLoRa](https://flora.aalto.fi/), [TensorFlow Lite Micro (Arduino)](https://github.com/tensorflow/tflite-micro-arduino-examples)

## Installation and Setup

### Prerequisites
- **Python 3.9** (Ensure you have it installed)
- **OMNeT++** (Download and install from [OMNeT++ Official Website](https://omnetpp.org/intro/))
- **INET and FLoRa** (Ensure they are set up correctly)

### Steps to Set Up
1. Install required Python packages:
   ```sh
   pip3 install -r requirements.txt
   ```
2. Download and install **OMNeT++** following the instructions [here](https://omnetpp.org/intro/).
3. Ensure **INET** and **FLoRa** are correctly set up.
4. Create a file named `config.json` inside `baselines3/` and populate it with the following structure (adjust paths as needed):
   ```json
   {
       "project_dir": "repository/path",
       "omnetpp_root": "omnetpp/path",
       "logfile_path": "repository/path/flora/simulations/scenarios/advancedcase/logFile",
       "control_logfile_path": "repository/path/flora/simulations/scenarios/basecontrolcase/stationaryPacketLog.json",
       "training_info_path": "repository/path/inet4.4/src/inet/RL/modelfiles/training_info.json",
       "model_path": "repository/path/inet4.4/src/inet/RL/modelfiles/gen_model.tflite",
       "scenario_path": "repository/path/flora/simulations/scenarios/advancedcase",
       "ini_file_name": "omnetpp.ini"
   }
   ```

## Usage

### Training a New Model
Run the following script to train a new model:
```sh
python python_scripts/baselines3/twodenvrunner.py
```
- The **latest iteration** is saved as:
  ```sh
  python_scripts/baselines3/stable-model.zip
  ```
- The **best iteration** is saved as:
  ```sh
  python_scripts/baselines3/stable-model-2d-best/stable-model.zip
  ```

### Testing a Trained Model
To test a trained model, run:
```sh
python python_scripts/baselines3/test2dmodel.py
```

### Exporting a Trained Model
Before using the trained model in OMNeT++, it must be converted to TensorFlow Lite:
```sh
python python_scripts/baselines3/sb3_to_tflite.py
```
This script also generates a header file containing necessary information for OMNeT++ evaluation.

### Running the Simulation in OMNeT++
#### Building the Project
- Before running the OMNeT++ simulation, ensure the model has been exported using the export script.
- Build the OMNeT++ project to include the newly generated header file.
- If the model architecture (number of layers/nodes) changes, rerun the export script and rebuild the project to ensure compatibility.

### Evaluating a Trained Model in OMNeT++
To evaluate a trained model (`stable-model-2d-best/stable-model.zip`) in **OMNeT++**, run:
```sh
python python_scripts/sim_omnet_evaluator.py
```
- To specify a scenario, modify the `ini_config` variable inside `env.run_simulation(...)`.
- A list of available scenarios can be found in:
  ```sh
  flora/simulations/scenarios/advancedcase/omnetpp.ini
  ```
  
---
This project serves as a framework for integrating reinforcement learning with LoRa mobility in OMNeT++. Further refinements can expand its capabilities for real-world deployment.
