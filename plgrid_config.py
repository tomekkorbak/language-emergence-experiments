import os

from mrunner.experiment import Experiment

cmds = []
for f in [2, 5, 10, 12, 15, 20, 25, 50, 100, 250]:
    for v in [5, 10, 15, 20, 25, 50]:
            cmds.append(f'python teamwork/main.py --batch_size 64 --n_epochs 20000 --population_size 1 '
                        f'--max_len 2 --vocab_size {v} --validation_freq 10 --sender_hidden 100  --n_features {f}')

experiments_list = [Experiment(project='', name='', parameters=None, script=cmd, python_path='.', paths_to_dump='',
                               env={"NEPTUNE_API_TOKEN": os.environ["NEPTUNE_API_TOKEN"]},) for cmd in cmds]
