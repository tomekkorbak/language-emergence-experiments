import os

from mrunner.experiment import Experiment

cmds = []
for p in range(1, 12):
    cmds.append(f'python teamwork/main.py --population_size {p}')

experiments_list = [Experiment(project='', name='', parameters=None, script=cmd, python_path='.', paths_to_dump='',
                               env={"NEPTUNE_API_TOKEN": os.environ["NEPTUNE_API_TOKEN"]},) for cmd in cmds]
