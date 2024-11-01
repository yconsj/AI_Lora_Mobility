import subprocess
import os
import json
class OmnetEnv:
    def __init__(self, config_file='config.json'):
        self.config = self.load_config(config_file)
        self.project_dir = self.config['project_dir']
        self.omnetpp_root = self.config['omnetpp_root']

        # Path variables based on project dir and omnet dir
        self.mingw_tools = os.path.join(self.omnetpp_root, 'tools', 'win32.x86_64', 'opt', 'mingw64', 'bin').replace('\\', '/')
        self.setenv_script = os.path.join(self.omnetpp_root, "setenv").replace('\\', '/')
        self.sim_path = os.path.join(self.project_dir, 'flora', 'simulations/scenarios/basecase').replace('\\', '/')
        self.ini_file = os.path.join(self.sim_path, 'omnetpp.ini').replace('\\', '/')
        self.mingwenv_cmd_path = os.path.join(self.omnetpp_root, "tools/win32.x86_64/msys2_shell.cmd").replace('\\', '/')

        # Unpack tools and dependencies if not already unpacked
        self.unpack_tools()

        # Set environment variables
        self.set_environment()

    def load_config(self, config_file):
        # Get the directory of the currently running script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the full path to the config file
        config_path = os.path.join(script_dir, config_file)
        with open(config_path, 'r') as f:
            return json.load(f)

    def unpack_tools(self):
        toolchain_file = os.path.join(self.omnetpp_root, 'tools', 'opp-tools-win32-x86_64-mingw64-toolchain.7z').replace('\\', '/')
        dependencies_file = os.path.join(self.omnetpp_root, 'tools', 'opp-tools-win32-x86_64-mingw64-dependencies.7z').replace('\\', '/')
        
        if os.path.exists(toolchain_file):
            subprocess.run(['7za', 'x', '-aos', '-y', '-owin32.x86_64', toolchain_file], check=True)
            os.remove(toolchain_file)
        
        if os.path.exists(dependencies_file):
            subprocess.run(['7za', 'x', '-aos', '-y', '-owin32.x86_64/opt/mingw64', dependencies_file], check=True)
            os.remove(dependencies_file)
            qtbinpatcher = os.path.join(self.omnetpp_root, 'tools', 'win32.x86_64', 'opt', 'mingw64', 'bin', 'qtbinpatcher.exe').replace('\\', '/')
            subprocess.run([qtbinpatcher, '--qt-dir=win32.x86_64/opt/mingw64'], check=True)

    def set_environment(self):
        os.environ["HOME"] = os.path.dirname(self.omnetpp_root)
        os.environ["PATH"] = self.mingw_tools + ";" + os.environ.get("PATH", "")
        os.environ["PYTHONPATH"] = os.path.join(self.omnetpp_root, 'python').replace('\\', '/') + ";" + os.environ.get("PYTHONPATH", "")

    def run_simulation(self, episode_seed = 0):
        # Construct the command to run OMNeT++ simulation
        # Note flora dll after inet dll, in opp_run cmd
        command = (
            f'"{self.mingwenv_cmd_path}" -mingw64 -no-start -defterm -c '
            f'"cd {self.sim_path} && '
            f'source {self.setenv_script} && '
            f'opp_run --seed-set={episode_seed} -m -u Cmdenv -n ../../../src:../..:../../../../inet4.4/examples:../../../../inet4.4/showcases:../../../../inet4.4/src:../../../../inet4.4/tests/validation:../../../../inet4.4/tests/networks:../../../../inet4.4/tutorials:../../../../tflite-micro-arduino-examples -x inet.common.selfdoc:inet.linklayer.configurator.gatescheduling.z3:inet.emulation:inet.showcases.visualizer.osg:inet.examples.emulation:inet.showcases.emulation:inet.transportlayer.tcp_lwip:inet.applications.voipstream:inet.visualizer.osg:inet.examples.voipstream --image-path=../../../../inet4.4/images -l ../../../../inet4.4/src/libINET.dll -l ../../../src/libflora.dll omnetpp.ini "'
        )

        print(f'Running Omnet simulation command!')
        
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Simulation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running simulation: {e}")
            print("Standard Output:", e.stdout)
            print("Standard Error:", e.stderr)


# Main function to execute the simulation
if __name__ == "__main__":
    simulation = OmnetEnv()  # Create an instance of the simulation
    simulation.run_simulation()  # Run the simulation