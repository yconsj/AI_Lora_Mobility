# New class that inherits from OmnetEnv
import json
import os
import subprocess


from sim_runner import OmnetEnv


class BaseControlCaseEnv(OmnetEnv):
    def __init__(self, config_file='config.json'):
        super().__init__(config_file)
        # Update the simulation path to BaseControlCase
        self.sim_path = os.path.join(self.project_dir, 'flora', 'simulations/scenarios/basecontrolcase').replace('\\',
                                                                                                                 '/')

    # Override run_simulation to remove the episode_seed

    def run_simulation(self, run_number=0):
        # Construct the command to run OMNeT++ simulation with runnumber
        command = (
                f'"{self.mingwenv_cmd_path}" -mingw64 -no-start -defterm -c '
                f'"cd {self.sim_path} && '
                f'source {self.setenv_script} && ' +
                f'opp_run -r {run_number} --seed-set={run_number} -m -u Cmdenv -n ../../../src:../..:../../../../inet4.4/examples:../../../../inet4.4/showcases:../../../../inet4.4/src:../../../../inet4.4/tests/validation:../../../../inet4.4/tests/networks:../../../../inet4.4/tutorials:../../../../tflite-micro-arduino-examples -x inet.common.selfdoc:inet.linklayer.configurator.gatescheduling.z3:inet.emulation:inet.showcases.visualizer.osg:inet.examples.emulation:inet.showcases.emulation:inet.transportlayer.tcp_lwip:inet.applications.voipstream:inet.visualizer.osg:inet.examples.voipstream --image-path=../../../../inet4.4/images -l ../../../../inet4.4/src/libINET.dll -l ../../../src/libflora.dll omnetpp.ini "'
        )
        print(f'Running BaseControlCase simulation command with runnumber {run_number}!')

        try:
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print("BaseControlCase simulation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running BaseControlCase simulation: {e}")
            print("Standard Output:", e.stdout)
            print("Standard Error:", e.stderr)


def load_stationary_data():
    from RL_tensor import load_config
    config = load_config("config.json")
    stationary_log_path = config['control_logfile_path']
    # Check if the file exists
    if not os.path.exists(stationary_log_path):
        # Create a new empty JSON file
        with open(stationary_log_path, 'w') as f:
            json.dump({}, f)  # Dump an empty dictionary to create an empty JSON file
        print(f"Created new empty JSON file at: {stationary_log_path}")

    # Load the data from the JSON file
    with open(stationary_log_path, 'r') as f:
        return json.load(f)


base_control_case_env = None


def run_base_control_case(run_number):
    global base_control_case_env

    # Check if the base control case environment is initialized
    if base_control_case_env is None:
        base_control_case_env = BaseControlCaseEnv()  # Initialize the environment if not already done

    print(f"Running base control case for run number: {run_number}")
    # Run the simulation for the base control case with the given run number
    base_control_case_env.run_simulation(run_number=run_number)


if __name__ == "__main__":
    base_control_sim = BaseControlCaseEnv()  # Create an instance of the BaseControlCaseEnv class
    base_control_sim.run_simulation(run_number=20)  # Run the simulation with a specified runnumber
