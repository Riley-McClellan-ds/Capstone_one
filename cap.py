import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

plt.rcParams.update({'font.size': 16, 'font.family': 'sans'})

#FIX YOUR FUCKING GRAPH FONT SIZES ETC SET DTI OR WHATEVER TO SOMETHING
#FIX YOUR FUCKING GRAPH FONT SIZES ETC SET DTI OR WHATEVER TO SOMETHING
#FIX YOUR FUCKING GRAPH FONT SIZES ETC SET DTI OR WHATEVER TO SOMETHING
#FIX YOUR FUCKING GRAPH FONT SIZES ETC SET DTI OR WHATEVER TO SOMETHING
#FIX YOUR FUCKING GRAPH FONT SIZES ETC SET DTI OR WHATEVER TO SOMETHING

random.seed(30)

pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 50)

tc1 = pd.read_csv('/Users/riley/Desktop/DSI/den-19/capstones/Capstone_one/tree-csvs/new_york_tree_census_2015.csv')
#sample for testing will change tc1 back to tc upon deletion

tc = tc1.sample(50000, random_state=2)
tsp = pd.read_csv('/Users/riley/Desktop/DSI/den-19/capstones/Capstone_one/tree-csvs/new_york_tree_species.csv') 

# tc.drop(index=297358, inplace=True) #can only use with orig tc

# You don't have guards and stewards for dead trees or stumps all the issues except 51 were directly related to that issue the 48 of the 51 
# Nan values were in the problems section

#assign health category
tc.loc[tc.status == 'Dead', 'health'] = 'Dead'
tc.loc[tc.status == 'Stump', 'health'] = 'Stump'
#assign sidewalk to binomials
tc.loc[tc.sidewalk == 'NoDamage', 'sidewalk'] = 'No'
tc.loc[tc.sidewalk == 'Damage', 'sidewalk'] = 'Yes'

tc['spc_latin'] = tc['spc_latin'].replace([np.nan],'Unkown')
tc['spc_common'] = tc['spc_common'].replace([np.nan],'Unknown')
tc['steward'] = tc['steward'].replace([np.nan],'None')
tc['guards'] = tc['guards'].replace([np.nan],'None')
tc['sidewalk'] = tc['sidewalk'].replace([np.nan],'None')
tc['problems'] = tc['problems'].replace([np.nan],'Unknown') #perhaps isolate and differentiate the 48 unique NaN problems

indicator_cols = ['curb_loc', 'spc_latin', 'steward', 'guards', 'sidewalk', 'user_type', \
    'problems', 'root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other', \
        'brch_light', 'brch_shoe', 'brch_other', 'zipcode', 'borocode' ]

binom_cols = ['root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other', \
        'brch_light', 'brch_shoe', 'brch_other', 'sidewalk']
for i in binom_cols:
    tc.loc[tc.sidewalk == 'Yes'] = 1
    tc.loc[tc.sidewalk == 'No'] = 0

non_bin_ind_col = ['curb_loc', 'steward', 'sidewalk', 'user_type', 'borocode']

#need to account for problem column and zipcode
# 'curb_loc' is categorical needs each value processed as a "Yes" separately Use intermediate value to have "third option"
# only while switcing to yes for the sake of using function to plot
#zipcode

better_col_names = {
    'curb_loc': 'Curb Location',
    'spc_latin': 'Latin Name',
    'steward': 'Signs of Care',
    'guards': 'Tree Guards (structure)',
    'sidewalk': 'Sidewalk Damage',
    'user_type': 'Data recorder',
    'problems': 'Other Problems',
    'root_stone': 'Roots Restricted',
    'root_grate': 'Roots Restricted by Grate',
    'root_other': 'Root issues Other',
    'trunk_wire': 'Trunk Restricted by Wire',
    'trnk_light': 'Trunk Restricted by Lights',
    'trnk_other': "Trunk Restriction (Other)",
    'brch_light': "Branch Restriced by Lights",
    'brch_shoe': "Branches with Shoes on them",
    'brch_other': "Branch Issue (Other)",
    'zipcode': 'Zipcode',
    'borocode': "Borough Code"
}

def aggregate_tree_indicators(df, new_col):
    df[new_col] = tc

def incomplete_function_to_f_remove_nan_health(df):
    lst = 0
    for i in tc.index:
        if df.loc[i, 'health'] not in ['Good', 'Fair', 'Poor', 'Dead', 'Stump']:
            lst += 1
    pass

def find_nan_values():
    """[Identifies columns with NaN values and returns number of NaN values per col]

    Returns:
       [List]: 2 lists in parralell with column name and number of NaN values respectively 
    """    
    nan_columns = []
    nan_nums = [] # in parrallel with nan_columns
    for i in tc.columns:
        if tc[i].isnull().values.any():
            nan_columns.append(i)
    for i in tc.columns:
        if tc[i].isnull().sum():
            nan_nums.append(tc[i].isnull().sum())
    return nan_columns, nan_nums

# for i in find_nan_values()
# print(i)

# print(indicator_cols)
# lst = 0
# for i in tc.index:
#     if tc.loc[i, 'status'] == 'Dead':
#         tc.loc[i, 'health'] = 'Dead'

# for i in tc.index:
#     if tc.loc[i, 'health'] not in ['Good', 'Fair', 'Poor', 'Stump']:
#         lst += 1
# print(lst)
# # lsty = list()
# # print('start')
# df.loc[df.my_channel > 20000, 'my_channel'] = 0
# tc.loc[tc.status == 'Dead', 'health'] = 'Dead'

# tc.loc[i, 'spc_latin'] = 'Unkown'
   
# for i in tc.index:
#     if type(tc.loc[i, 'spc_latin']) == type(1.2):
#         lsty.append(i)




# tc.loc[lsty[0], 'spc_latin'] = 'Unkown'
# print(tc.loc[lsty[0], 'spc_latin'])


def check_nan_amount(df, columns):
    '''
    input:
    df: dataframe
    columns: list of string column names
    
    output:
    a tuple containing a set of indexes where there in a NaN value in one of the 
    given columns as well as the length of that set.
    '''

    idx = set()
    for i in columns:
        for b in df[df[i].isnull()].index.values:
            idx.add(b)
    return len(idx), idx

# print(check_nan_amount(tc, nan_columns))

# idx = set()

# for i in bad_columns:
#     tc[tc[i].isnull()].index.values
#     for b in tc[tc[i].isnull()].index.values:
#         idx.add(b)
# print(len(idx))


#Dict of unique values in each column
cv = dict()

for i in tc.columns:
    cv[i] = []

for i in tc.columns:
    for value in tc[i].unique():
        cv[i].append(value)
    # cv[i].append(tc[i].unique())

for i in non_bin_ind_col:
    print(cv[i])

# print(cv['brch_other'])


# sum_root = 0
# for i in tc.index:
#     if tc.loc[i, 'root_stone'] == 'Yes':
#         sum_root += 1
# print(sum_root)


def tot_tree_health(df):
    health_dict = dict()
    for item in cv['health']:
        health_dict[item] = 0
        for idx in df.index:
            if df.loc[idx, 'health'] == item:
                health_dict[item] += 1
    return health_dict

def categorical_val(df, column):
    """ Takes a column with categorical infomration and plots unique values vs healthy tree count

    Args:
        df ([dataframe]): [Tree cencus]
        column ([Str]): [Categorical response column]
    """    
    
    indicators = dict()
    cat_idx = dict()
    for item in cv[column]:
        if item not in ["Dead", "None"]:
            indicators[item] = 0
            cat_idx[item] = set()
            for idx in df.index:
                if df.loc[idx, column] == item:
                    cat_idx[item].add(idx)
                    indicators[item]+= 1
    return cat_idx

def indicator_value(df, columns, value="Yes"):
    """[Takes a list of columns and returns a dictionary with values that are sets of indexes
    where the value was "Yes"]

    Args:
        df ([pandas dataframe]): 
        columns ([list]): List of strings indicating column names

    Output:
    two dictionaries one with column and number of hits, the other with column and a list of 'yes' indexes. 
    """    
    indicators = dict()
    idc_idx = dict()
    for item in columns:
        indicators[item] = 0
        idc_idx[item] = set()
        for idx in df.index:
            if df.loc[idx, item] == value:
                idc_idx[item].add(idx)
                indicators[item]+= 1
    return idc_idx

def indicator_vs_health(df, indicator_idx_set):
    """[summary]

    Args:
        df ([pandas datafram]): 
        indicator ([set of indexes where indicator == 'Yes']):
    Description:
    Function outputs data sorted by tree health in a format that can be graphed

    Output: dictionary where keys are Health values and values are sets of idx's
    """    
    idc_health = {
        "Good": 0,
        "Fair": 0,
        "Poor": 0,
        "Dead": 0,
    }
    health_idx =  {
        "Good": set(),
        "Fair": set(),
        "Poor": set(),
        "Dead": set(),
    }
    for idx in indicator_idx_set:
        for categ in idc_health:
            if df.loc[idx, 'health'] == categ:
                idc_health[categ] +=1
                health_idx[categ].add(idx)
        # if df.loc[i, 'health'] == "Good":
        #     idc_health["Good"] +=1
            
        # if df.loc[i, 'health'] == "Fair":
        #     idc_health["Fair"] +=1
        # if df.loc[i, 'health'] == "Poor":
        #     idc_health["Poor"] +=1
        # if df.loc[i, 'health'] == "Dead":
        #     idc_health["Dead"] +=1
    return idc_health

# _ , ind_idx = indicator_value(tc, binom_cols)
# for i in binom_cols:
#     print(i)
#     print(indicator_vs_health(tc, ind_idx[i]))
#     print()
#where the species in the list of names is NaN
# print(sp_lst[10])

def graph_tot_health(ax):

    D1 = tot_tree_health(tc)
    D = {k: v for k, v in sorted(D1.items(), reverse=True, key=lambda item: item[1])}
    x = np.arange(len(D))

    labels = D.keys()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
             
    balues = ax.bar(x, D.values(), align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of trees by Health category')
    ax.set_title('Tree Health')

    autolabel(balues)
    fig.tight_layout()


def plottheshit(dick, ax, label_start="Potential Indicators"):
    """[summary]

    Args:
        dick ([dict of dict]): Column: A dictionary with keys being individual indicators
        Values are the number of trees found presenting a given indicator in each health category.
    """    
    labels = sorted(list(dick.keys()))
    Good = []
    Fair = []
    Poor = []
    Dead = []
    status = {
        "Good": Good,
        "Fair": Fair,
        "Poor": Poor,
        "Dead": Dead
    }
    better_labels = []
    if labels[0] in better_col_names.keys():
        for i in labels:
            better_labels.append(better_col_names[i])
    else:
        better_labels = labels
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')


    print(labels)
    for col in labels:
        for k, v in status.items():
            v.append(dick[col][k])

    x = np.arange(len(labels))
    width = .25

    rects1 = ax.bar(x - width*1.5, Good, width, label='Good', Color="green")
    rects2 = ax.bar(x - width/2, Fair, width, label='Fair', color="orange")
    rects3 = ax.bar(x + width/2, Poor, width, label='Poor', color="magenta")
    if not sum(Dead) == 0:
        rects4 = ax.bar(x + width*1.5, Dead, width, label='Dead', color="black")
        autolabel(rects4)
    rects_lst = [rects1, rects2, rects3]

    ax.set_ylabel('Number of Trees Per Category')
    ax.set_title(f"{label_start} compared to Tree Health")
    ax.set_xticks(x)
    ax.set_xticklabels(better_labels, rotation=45, ha="right")
    ax.legend()

    for i in rects_lst:
        autolabel(i)
    fig.tight_layout()



def plot_cat_col(column, ax):
    """[Plots an individual column defined as Categorical]

    Args:
        column ([Str]): [Column to be graphed]
    
    """    
    cat_idx = categorical_val(tc, column)
    better_column = better_col_names[column]
    cat_vs_health = dict()
    for i in cat_idx.keys():
        cat_vs_health[i] = indicator_vs_health(tc, cat_idx[i])
    plottheshit(cat_vs_health, ax,label_start=better_column)

def plot_binom_cols(binom_cols, ax):
    """[Plots columns defined as Binomial ( values correlating to "Yes" (1) or "No" (0)]

    Args:
        binom_cols ([list]): [list of Binomial Columns]
        ax ([Ax]): [Axis for graph to plot to]
    """    
    idc_idx = indicator_value(tc, binom_cols)
    idx_by_health_dict = dict()
    for i in binom_cols:
        idx_by_health_dict[i] = indicator_vs_health(tc, idc_idx[i])
    plottheshit(idx_by_health_dict, ax)
    
# fig, ax = plt.subplots(figsize=(16,10))
# plot_binom_cols(binom_cols, ax)
# plt.show()

# fig, ax = plt.subplots(figsize=(16,7))
# plot_cat_col('borocode', ax)

# plt.show()

# fig, ax = plt.subplots()
# graph_tot_health(ax)
# plt.show()









