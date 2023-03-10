"""
    Code source: https://github.com/SALib/SALib
"""


import pandas as pd 
# magic string indicating DF columns holding conf bound values
CONF_COLUMN = '_conf'
def varplot(Si_df, ax=None):
    """Create bar chart of results.

    Parameters
    ----------
    * Si_df: pd.DataFrame, of sensitivity results

    Returns
    ----------
    * ax : matplotlib axes object

    Examples
    ----------
    >>> from SALib.plotting.bar import plot as barplot
    >>> from SALib.test_functions import Ishigami
    >>>
    >>> # See README for example problem specification
    >>>
    >>> X = saltelli.sample(problem, 512)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=False)
    >>> total, first, second = Si.to_df()
    >>> barplot(total)
    """
    conf_cols = Si_df.columns.str.contains(CONF_COLUMN)

    confs = Si_df.loc[:, conf_cols]
    confs.columns = [c.replace(CONF_COLUMN, '') for c in confs.columns]

    Sis = Si_df.loc[:, ~conf_cols]

    ax = Sis.plot(kind='bar', yerr=confs, ax=ax)
    return ax

class ResultDict(dict):
    '''Dictionary holding analysis results.

    Conversion methods (e.g. to Pandas DataFrames) to be attached as necessary
    by each implementing method
    '''
    def __init__(self, *args, **kwargs):
        super(ResultDict, self).__init__(*args, **kwargs)

    def to_df(self):
        '''Convert dict structure into Pandas DataFrame.'''
        return pd.DataFrame({k: v for k, v in self.items() if k != 'names'},
                            index=self['names'])

    def plot(self, ax=None):
        '''Create bar chart of results'''
        Si_df = self.to_df()

        if isinstance(Si_df, (list, tuple)):
            import matplotlib.pyplot as plt  # type: ignore

            if ax is None:
                fig, ax = plt.subplots(1, len(Si_df))

            for idx, f in enumerate(Si_df):
                barplot(f, ax=ax[idx])

            axes = ax
        else:
            axes = barplot(Si_df, ax=ax)

        return axes
