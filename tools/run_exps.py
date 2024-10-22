import subprocess
import time
import os

games = [
    "ALE/Seaquest-v5",
    "ALE/WizardOfWor-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Asterix-v5",
    "ALE/DoubleDunk-v5",
    "ALE/Kangaroo-v5",
    "ALE/Atlantis-v5"
]
base_command = ''
home_directory = os.path.expanduser('~')
if os.path.exists(f"{home_directory}/miniconda3/"):
    base_command = (f"{home_directory}/miniconda3/bin/conda run -n Reinforcement-Learning --no-capture-output python "
                    f"{home_directory}/rl/algorithms/dqn.py --env_name")
elif os.path.exists(f"{home_directory}/anaconda3/"):
    base_command = (f"{home_directory}/anaconda3/bin/conda run -n Reinforcement-Learning --no-capture-output python "
                    f"{home_directory}/rl/algorithms/dqn.py --env_name")

python_path = f"export PYTHONPATH=$PYTHONPATH:/{home_directory}/rl"
# Activate your virtual environment command

# Loop through each game, creating a screen session for each one
for game in games:
    # Creating a unique screen name for each game session
    screen_name = f"{game.split('/')[1]}"

    # Combine commands to activate env and start training in one command string
    combined_command = f"{python_path}&&{base_command} '{game}' --n_times 5"

    # Full command to run in screen
    screen_command = f"screen -dmS {screen_name} bash -c '{combined_command}; exec bash'"

    # Execute the command to start the screen session with the training running
    subprocess.run(screen_command, shell=True)
    time.sleep(1)
print("Training sessions started in separate screen sessions.")
