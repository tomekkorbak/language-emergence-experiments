import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neptunelib.session import Session

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(11.7,8.27)})

class NeptuneHelper(object):
    def __init__(self, organization, project_name):
        self.api_token = os.environ.get('NEPTUNE_API_KEY')
        self.session = Session(api_token=self.api_token)
        self.organization = organization
        self.project_name = project_name

        self.projects = self.session.get_projects(self.organization)
        self.project = self.projects[self.project_name]

    def get_experiments_by_tag(self, tag):
        return self.project.get_experiments(tag=tag)

    def get_parameters_for_experiments(self, tag, prune=False):
        dfs = []
        experiments = self.project.get_experiments(tag=tag)
        for experiment in experiments:
            df = experiment.parameters
            df['experiment_id'] = experiment.id
            dfs.append(df)
        df = pd.concat(dfs)
        if prune:
            c = [c for c in df.columns if len(set(df[c])) != 1]
            df = df[c]
        df.set_index('experiment_id', inplace=True)

        return df

    def get_channel_for_experiments(self, tag, channel_name,
                                    final_values_only=False,
                                    drop_x_channel=False,
                                    parameters_to_add=[]):
        assert type(channel_name) is str, "This helper does not support multiple columnns, yet(?)"
        dfs = []
        experiments = self.project.get_experiments(tag=tag)
        for experiment in experiments:
            values = experiment.get_numeric_channels_values(channel_name)
            values['experiment_id'] = experiment.id
            for parameter in parameters_to_add:
                p = experiment.parameters[parameter][0]
                try:
                    p = float(p)
                except:
                    pass

                values[parameter] = p

            if drop_x_channel:
                values.drop(rf"x", axis=1, inplace=True)

            if final_values_only:
                values = values[-1:]
            dfs.append(values)

        return pd.concat(dfs)

if __name__ == "__main__":
    helper_obv = NeptuneHelper('tomekkorbak', 'tomekkorbak/obverter2')
    columns =  ['1_RSA', '2_RSA', '1_context independence', '2_context independence', 'train_accuracy', 'test_accuracy',
                'test_first_accuracy', 'test_second_accuracy']
    data = {}
    for column in columns:
        df = helper_obv.get_channel_for_experiments('kokosy', column, final_values_only=True, drop_x_channel=True)
        data[column + '_mean'] = df[column].mean()
        data[column + '_std'] = df[column].std()

    data['RSA_mean'] = (data['1_RSA_mean'] + data['2_RSA_mean']) / 2
    data['RSA_std'] = (data['1_RSA_std'] + data['2_RSA_std']) / 2
    del data['1_RSA_mean'], data['2_RSA_mean'], data['1_RSA_std'], data['2_RSA_std']

    data['context_independence_mean'] = (data['1_context independence_mean'] + data['2_context independence_mean']) / 2
    data['context_independence_std'] = (data['1_context independence_std'] + data['2_context independence_std']) / 2
    del data['1_context independence_mean'], data['2_context independence_mean'], data['1_context independence_std'], data['2_context independence_std']

    data['both_test_accuracy_mean'] = (data['test_first_accuracy_mean'] + data['test_second_accuracy_mean']) / 2
    data['both_test_accuracy_std'] = (data['test_first_accuracy_std'] + data['test_second_accuracy_std']) / 2
    del data['test_first_accuracy_mean'], data['test_second_accuracy_mean'], data['test_first_accuracy_std'], data['test_second_accuracy_std']
    df = pd.DataFrame.from_dict({k: [v] for k, v in data.items()})
    print(df)
    print(df.to_latex())


    # helper_obv = NeptuneHelper('tomekkorbak', 'tomekkorbak/template-transfer')
    # columns = ['compRSA', 'comp_train_accuracy', 'comp_test_accuracy', 'comp_test_first_accuracy', 'comp_test_second_accuracy', 'compcontext independence']
    # data = {}
    # for column in columns:
    #     df = helper_obv.get_channel_for_experiments('rrr', column, final_values_only=True, drop_x_channel=True)
    #     data[column + '_mean'] = df[column].mean()
    #     data[column + '_std'] = df[column].std()
    #
    # data['comp_test_accuracy_both_mean'] = (data['comp_test_first_accuracy_mean'] + data['comp_test_second_accuracy_mean']) / 2
    # data['comp_test_accuracy_both_std'] = (data['comp_test_first_accuracy_std'] + data['comp_test_second_accuracy_std']) / 2
    # del data['comp_test_first_accuracy_mean'], data['comp_test_second_accuracy_mean'], data['comp_test_first_accuracy_std'], data['comp_test_second_accuracy_std']
    # df = pd.DataFrame.from_dict({k: [v] for k, v in data.items()})
    # print(df)
    # print(df.to_latex())
