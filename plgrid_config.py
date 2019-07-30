import os

from mrunner.experiment import Experiment

cmds = []
cmds.append(f'python compositionality/main.py --batch_size 64 --n_epochs 20000 --max_len 2 --vocab_size 15 --validation_freq 10  --n_features 10')

experiments_list = [Experiment(project='', name='', parameters=None, script=cmd, python_path='.', paths_to_dump='',
                               env={"NEPTUNE_API_TOKEN": os.environ["NEPTUNE_API_TOKEN"]},) for cmd in cmds]
