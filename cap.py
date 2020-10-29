import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import scipy.stats as stats
import time

t0 = time.time()
print(t0)
plt.rcParams.update({'font.size': 16, 'font.family': 'sans'})
plt.style.use('ggplot')




pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 50)

tc = pd.read_csv('/Users/riley/Desktop/DSI/den-19/capstones/Capstone_one/tree-csvs/new_york_tree_census_2015.csv')
#sample for testing will change tc1 back to tc upon deletion

# tc = tc1.sample(50000, random_state=2)
tsp = pd.read_csv('/Users/riley/Desktop/DSI/den-19/capstones/Capstone_one/tree-csvs/new_york_tree_species.csv') 

# tc.drop(index=297358, inplace=True) #can only use with orig tc

# You don't have guards and stewards for dead trees or stumps all the issues except 51 were directly related to that issue the 48 of the 51 
# Nan values were in the problems section


indicator_cols = ['curb_loc', 'spc_latin', 'steward', 'guards', 'sidewalk', 'user_type', \
    'problems', 'root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other', \
        'brch_light', 'brch_shoe', 'brch_other', 'zipcode', 'borocode' ]

binom_cols = ['root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other', \
        'brch_light', 'brch_shoe', 'brch_other', 'sidewalk']

non_bin_ind_col = ['curb_loc', 'steward', 'sidewalk', 'user_type', 'borocode']

def clean_data():
    #assign health category
    tc.loc[tc.status == 'Dead', 'health'] = 'Dead'
    tc.loc[tc.status == 'Stump', 'health'] = 'Stump'
    #assign sidewalk to binomials
    tc.loc[tc.sidewalk == 'NoDamage', 'sidewalk'] = 0
    tc.loc[tc.sidewalk == 'Damage', 'sidewalk'] = 1

    tc['spc_latin'] = tc['spc_latin'].replace([np.nan],'Unknown')
    tc['spc_common'] = tc['spc_common'].replace([np.nan],'Unknown')
    tc['steward'] = tc['steward'].replace([np.nan],'None')
    tc['guards'] = tc['guards'].replace([np.nan],'None')
    tc['sidewalk'] = tc['sidewalk'].replace([np.nan],'None')
    tc['problems'] = tc['problems'].replace([np.nan],'Unknown') 

    for i in binom_cols:
        tc[i] = tc[i].replace("Yes", 1)
        tc[i] = tc[i].replace("No", 0)

clean_data()
t1 = time.time()
print(t1-t0)
print('clean')
print()
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
    'borocode': "Borough Code",
    'tot_ind': "Aggregate Indicator Value"
}

binary_sum_df = binom_cols.copy()
binary_sum_df.append("tot_ind")
binary_sum_df.append('health')
# print(tc[binom_cols].sum(axis=1))

def aggregate_tree_indicators(df, new_col):
    df[new_col] = df[binom_cols].sum(axis=1)
    agg_df = df[binary_sum_df]
    return agg_df

agg_df = aggregate_tree_indicators(tc, "tot_ind")
t2 = time.time()
print(t2-t1)
print('agg')
print()
def incomplete_function_to_f_remove_nan_health(df):
    lst = 0
    for i in df.index:
        if df.loc[i, 'health'] not in ['Good', 'Fair', 'Poor', 'Dead', 'Stump']:
            lst += 1
    pass

def find_nan_values(df):
    """[Identifies columns with NaN values and returns number of NaN values per col]

    Returns:
       [List]: 2 lists in parralell with column name and number of NaN values respectively 
    """    
    nan_columns = []
    nan_nums = [] # in parrallel with nan_columns
    for i in df.columns:
        if df[i].isnull().values.any():
            nan_columns.append(i)
    for i in df.columns:
        if df[i].isnull().sum():
            nan_nums.append(df[i].isnull().sum())
    return nan_columns, nan_nums



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

# a, b = check_nan_amount(tc, tc.columns)
# print(a)


#Dict of unique values in each column
cv = dict()

for i in tc.columns:
    cv[i] = []

for i in tc.columns:
    for value in tc[i].unique():
        cv[i].append(value)
t3 = time.time()
print(t3-t2)
print('cv')
print()
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

def indicator_value(df, columns, value=1):
    """[Takes a list of columns and returns a dictionary with values that are sets of indexes
    where the value was 1]

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



def plot_cat_col(df, column, ax):
    """[Plots an individual column defined as Categorical]

    Args:
        column ([Str]): [Column to be graphed]
    
    """    
    cat_idx = categorical_val(df, column)
    better_column = better_col_names[column]
    cat_vs_health = dict()
    for i in cat_idx.keys():
        cat_vs_health[i] = indicator_vs_health(df, cat_idx[i])
    plottheshit(cat_vs_health, ax,label_start=better_column)

def plot_binom_cols(df, binom_cols, ax):
    """[Plots columns defined as Binomial ( values correlating to 1 or 0]

    Args:
        binom_cols ([list]): [list of Binomial Columns]
        ax ([Ax]): [Axis for graph to plot to]
    """    
    idc_idx = indicator_value(df, binom_cols)
    idx_by_health_dict = dict()
    for i in binom_cols:
        idx_by_health_dict[i] = indicator_vs_health(df, idc_idx[i])
    plottheshit(idx_by_health_dict, ax)
    
# fig, ax = plt.subplots(figsize=(16,10))
# plot_binom_cols(binom_cols, ax)
# plt.show()

# fig, ax = plt.subplots(figsize=(16,7))
# plot_cat_col(agg_df, 'tot_ind', ax)

# plt.show()

# fig, ax = plt.subplots()
# graph_tot_health(ax)
# plt.show()


# stat, p_val = stats.ttest_ind(ctr_signed_in.CTR, ctr_not_signed_in.CTR, equal_var=False)

# print('The statistic is: {} \nP-value: {}'.format(stat ,p_val))





def get_dick(df, column):
    """[outputs dictionary necessary to graph/understand breakdown of indicators compa]

    Args:
        df ([type]): [description]
        column ([type]): [description]

    Returns:
        [type]: [description]
    """    
    cat_idx = categorical_val(df, column)
    better_column = better_col_names[column]
    cat_vs_health = dict()
    for i in cat_idx.keys():
        cat_vs_health[i] = indicator_vs_health(df, cat_idx[i])
    # plottheshit(cat_vs_health, ax,label_start=better_column)
    return cat_vs_health


def ratio_of_G_to_nG(cat_vs_health):
    """[Takes a dict of dict's broken down by tree health]

    Args:
        cat_vs_health ([Dict]): [Dictionary of Dictionaries containing G F P breakdown for trees]

    Returns:
        [variables necessary for p_value calculation and graphing]: [tuple of values]
    """    
    zero_good = 0
    zero_bad = 0

    tot_ind_good = 0
    tot_uh_trees = 0
    for i in cat_vs_health.keys():
        if i == 0:
            zero_good += cat_vs_health[i]['Good']
            zero_bad += cat_vs_health[i]['Fair']
            zero_bad += cat_vs_health[i]['Poor']
        else:
            tot_ind_good += cat_vs_health[i]['Good']
            tot_uh_trees += cat_vs_health[i]['Fair']
            tot_uh_trees += cat_vs_health[i]['Poor']
    if (zero_bad + zero_good) == 0:
        return [1, 1, 1, 1, "no", 1,]
    else:
        p_val = zero_bad/ (zero_bad + zero_good)

    zero_ind_tot = (tot_uh_trees + tot_ind_good)

    if (tot_uh_trees + tot_ind_good) == 0:
        return [1, 1, 1, 1, "no", 1,]
    else:
        ind_ratio = tot_uh_trees/ (tot_uh_trees + tot_ind_good)

    n_val = (tot_uh_trees + tot_ind_good)
    return zero_bad, zero_ind_tot, p_val, tot_uh_trees, n_val, ind_ratio



def get_p_value(n, p, bad, limit=4000, graph_dist=False, graph_p=False):
    if n == "no":
        return "no_p_value_for_you"
    binomial_mean = p * n
    binomial_var = n * p * (1-p)
    
    normal_approx = stats.norm(binomial_mean, np.sqrt(binomial_var))
    def hide_dist_graph():
        binomial = stats.binom(n=n, p=p)
        std = np.sqrt(binomial_var)
        x = np.linspace(0, n, num=n)

        fig, axs = plt.subplots( figsize=(16, 6))
        bar_sizes = [binomial.pmf(i) for i in range(n+1)]
        bars = axs[1].bar(range(n+1), bar_sizes, color="grey", align="center")
        axs[1].plot(x, normal_approx.pdf(x), linewidth=3)
        axs[1].set_xlim(binomial_mean-4*std, binomial_mean+4*std)
        axs[0].set_title("# of Unhealthy trees under null")
        plt.show()
    def hide_dist_p_graph():
        
        fig, ax = plt.subplots(1, figsize=(16, 3))

        ax.plot(x, normal_approx.pdf(x), linewidth=3)
        ax.set_xlim(binomial_mean-4*std, binomial_mean+4*std)
        ax.fill_between(x, normal_approx.pdf(x), 
                        where=(x >= n-1), color="red", alpha=0.5)
        ax.set_title("p-value Region")
        plt.show()
    if graph_dist == True:
        hide_dist_graph()
    if graph_p ==True:
        hide_dist_p_graph()
    p_value = 1 - normal_approx.cdf(bad-.1)
    print(f"p-value for indicator values is: {p_value:.4f}")
    return p_value
    
t4 = time.time()
print(t4-t3)
print('plotvals')
print()
 
# print(get_dick(agg_df, 'tot_ind'))

def get_indv_p_values(agg_df):
    p_value_dict = dict()
    temp_dict = dict()
    temp_dict_helper = get_dick(agg_df, 'tot_ind')
    temp_dict[0] = temp_dict_helper[0]
    for i in temp_dict_helper.keys():
        if i != 0:
            temp_dict[i] = temp_dict_helper[i]
            print(temp_dict)
            _, w, p, t, n, rt = ratio_of_G_to_nG(temp_dict)
            p_value_dict[i] = get_p_value(n, p, t, limit=w+t)
            temp_dict.pop(i)
    return p_value_dict

def reveal_results():
    p_values = get_indv_p_values(agg_df)
    alpha_bon = .05 / len(p_values)
    for i in p_values.keys():
        if p_values[i] < alpha_bon:
            print(f'We reject the null hypothesis with a value of {p_values[i]} at an indicator score of {str(i)}')
        else:
            print(f'We fail to reject the null hypothesis with a value of {p_values[i]} at an indicator score of {str(i)}')

reveal_results()

# print(get_indv_p_values(agg_df))
print()

t5 = time.time()
print(t5-t4)
print('get_dick')
print()
print(t5)
print()
#! fig, ax = plt.subplots(figsize=(16,7))
#! plot_cat_col(agg_df, 'tot_ind', ax)



zero_bad, zero_ind_tot, p_val, tot_uh_trees, n_val, ind_ratio = ratio_of_G_to_nG(get_dick(agg_df, 'tot_ind'))
for i in [zero_bad, zero_ind_tot, p_val, tot_uh_trees, n_val, ind_ratio]:
    print(i)


get_p_value(n_val, p_val, tot_uh_trees, limit=zero_ind_tot+tot_uh_trees)

# plt.show()


# cva = dict()

# for i in agg_df.columns:
#     cva[i] = []

# for i in agg_df.columns:
#     for value in agg_df[i].unique():
#         cva[i].append(value)
# print(cva)

