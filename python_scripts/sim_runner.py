import subprocess
import os
import json


# Load configuration from JSON file
def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        return json.load(f)

# Get configuration values
config = load_config()
project_dir = config['project_dir']
omnetpp_root = config['omnetpp_root']

# Path variables based on project dir and omnet dir
mingw_tools = os.path.join(omnetpp_root, 'tools', 'win32.x86_64', 'opt', 'mingw64', 'bin').replace('\\', '/')
setenv_script =  os.path.join(omnetpp_root, "setenv").replace('\\', '/')
sim_path = os.path.join(project_dir, 'flora', 'simulations').replace('\\', '/')
omnetpp_command = "opp_run"
ini_file = os.path.join(sim_path, 'omnetpp.ini').replace('\\', '/')
mingwenv_cmd_path = os.path.join(omnetpp_root, "tools/win32.x86_64/msys2_shell.cmd").replace('\\', '/')

# Unpack toolchains and dependencies if not already unpacked
def unpack_tools():
    toolchain_file = os.path.join(omnetpp_root, 'tools', 'opp-tools-win32-x86_64-mingw64-toolchain.7z').replace('\\', '/')
    dependencies_file = os.path.join(omnetpp_root, 'tools', 'opp-tools-win32-x86_64-mingw64-dependencies.7z').replace('\\', '/')
    
    if os.path.exists(toolchain_file):
        subprocess.run(['7za', 'x', '-aos', '-y', '-owin32.x86_64', toolchain_file], check=True)
        os.remove(toolchain_file)
    
    if os.path.exists(dependencies_file):
        subprocess.run(['7za', 'x', '-aos', '-y', '-owin32.x86_64/opt/mingw64', dependencies_file], check=True)
        os.remove(dependencies_file)
        qtbinpatcher = os.path.join(omnetpp_root, 'tools', 'win32.x86_64', 'opt', 'mingw64', 'bin', 'qtbinpatcher.exe').replace('\\', '/')
        subprocess.run([qtbinpatcher, '--qt-dir=win32.x86_64/opt/mingw64'], check=True)

# Set environment variables
def set_environment():
    os.environ["HOME"] = os.path.dirname(omnetpp_root)
    os.environ["PATH"] = mingw_tools + ";" + os.environ.get("PATH", "")
    os.environ["PYTHONPATH"] = os.path.join(omnetpp_root, 'python').replace('\\', '/') + ";" + os.environ.get("PYTHONPATH", "")

def run_simulation():
    # Ensure tools are unpacked
    unpack_tools()
    
    # Set environment variables
    set_environment()

    # Construct the command to run OMNeT++ simulation
    command = (
        f' "{mingwenv_cmd_path}" -mingw64 -c '
        f'"cd {sim_path} && '
        f'source {setenv_script} && '
        f'opp_run -m -u Cmdenv -n ../src:.:../../inet4.4/examples:../../inet4.4/showcases:../../inet4.4/src:../../inet4.4/tests/validation:../../inet4.4/tests/networks:../../inet4.4/tutorials -x inet.common.selfdoc:inet.linklayer.configurator.gatescheduling.z3:inet.emulation:inet.showcases.visualizer.osg:inet.examples.emulation:inet.showcases.emulation:inet.transportlayer.tcp_lwip:inet.applications.voipstream:inet.visualizer.osg:inet.examples.voipstream --image-path=../../inet4.4/images -l ../../inet4.4/src/libINET.dll -l ../src/libflora.dll omnetpp.ini"'
    )

    print(f'Running command: {command}')
    
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Simulation completed successfully.")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)

# Run the simulation function
run_simulation()
