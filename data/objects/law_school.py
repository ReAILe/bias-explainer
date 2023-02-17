import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from justicia import utils
import os


class Law_School:

    def __init__(self, verbose=True, config=0):
        self.filename = os.path.dirname(
            os.path.realpath(__file__)) + "/../raw/law_school.csv"
        self.name = "law_school"
        # print("Adult dataset")
        if(config == 0):
            self.known_sensitive_attributes = ['race', 'sex']
        elif(config == 1):
            self.known_sensitive_attributes = ['race']
        elif(config == 2):
            self.known_sensitive_attributes = ['sex']
        else:
            raise ValueError(
                str(config) + " is not a valid configuration for sensitive groups")
        self.config = config

        self.keep_columns = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa',
                             'zgpa', 'fulltime', 'family_income', 'sex', 'race', 'tier', 'pass_bar']
        self.categorical_attributes = [
            'fulltime', 'sex', 'race', 'pass_bar']
        self.continuous_attributes = [
            'decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'family_income']
        self.verbose = verbose

        self.mediator_attributes = []

        if(verbose):
            print("Sensitive attributes:", self.known_sensitive_attributes)

    def get_df(self, repaired=False):

        df = pd.read_csv(self.filename)

        # scale
        scaler = MinMaxScaler()
        df[self.continuous_attributes] = scaler.fit_transform(
            df[self.continuous_attributes])

        df = df[self.keep_columns]

        for known_sensitive_attribute in self.known_sensitive_attributes:
            if(known_sensitive_attribute in self.continuous_attributes):
                df = utils.get_discretized_df(df, columns_to_discretize=[
                                              known_sensitive_attribute])
                df = utils.get_one_hot_encoded_df(
                    df, [known_sensitive_attribute])
                self.continuous_attributes.remove(known_sensitive_attribute)

        df.rename(columns={'pass_bar': 'target'}, inplace=True)
        self.keep_columns.remove('pass_bar')
        self.keep_columns.append("target")

        df.to_csv(os.path.dirname(os.path.realpath(__file__)) +
                  "/../raw/reduced_law_school.csv", index=False)

        if(repaired):
            df = pd.read_csv(os.path.dirname(os.path.realpath(
                __file__)) + "/../raw/repaired_law_school.csv")

        if(self.verbose):
            print("-number of samples: (before dropping nan rows)", len(df))
        # drop rows with null values
        df = df.dropna()
        if(self.verbose):
            print("-number of samples: (after dropping nan rows)", len(df))

        return df
