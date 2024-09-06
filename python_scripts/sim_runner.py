import subprocess

# Path to your OMNeT++ executable or script
omnet_command = "/path/to/opp_run"  # Modify this with the correct path to opp_run or your simulation script

# Arguments for the OMNeT++ simulation
args = ["-u", "Cmdenv", "-c", "General", "omnetpp.ini"]

# Combine the command and arguments
command = [omnet_command] + args

# Run the OMNeT++ simulation
try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Simulation Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error during simulation:\n", e.stderr)