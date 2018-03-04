import datetime
import numpy as np
import os
import pandas as pd
import re

# signal functions
def string_in_line(line, string):
    evaluation = string in line
    return(evaluation)

def end_on_same_line(line):
    evaluation = True
    return(evaluation)

def end_if_new_line(line):
    evaluation = ('\n' == line)
    return(evaluation)

# stats table functions
def stats_table_line_list_transform(line_list, **kwargs):
    new_line_list = line_list[1:]
    new_line_list = [line.strip() for line in new_line_list]
    new_line_list = [
        [float(i) for i in line.split() if is_float_try(i)]
        for line in new_line_list
    ]
    line_list = [line_list[1].split()] + new_line_list
    line_list = [x for x in line_list if x]
    df = pd.DataFrame(line_list)
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))
    return(df)

# convergence stat functions
def converged_line_list_transform(line_list, **kwargs):
    regex = '\((.*)[<|>]'
    str_list = re.findall(regex, line_list[0])[0].strip().split()
    float_list = [np.float(i) for i in str_list]
    return(float_list)

# likelihood functions
def likelihood_line_list_transform(line_list, **kwargs):
    return(line_list[0])

def parse_likelihoods(
    search_str_list,
    find_str_list, 
    extraction_func
):
    """post-process output from likelihood str search
    for each entry in search_str_list, find the first str in search_str_list in the entry (if any)
    """
    out_dict = {find_str:[] for find_str in find_str_list}
    for line in search_str_list:
        idx_list = [
            idx for idx, find_str in enumerate(find_str_list)
            if find_str.lower() in line.lower()
        ]
        if idx_list: 
            ll = extraction_func(line)
            key = find_str_list[idx_list[0]]
            out_dict[key].append(ll)
    return(out_dict)

# helper functions
def is_float_try(str):
    """try cast str as a float
    """
    try:
        float(str)
        return True
    except ValueError:
        return False

def extract_likelihood_float(string):
    regex = '([-+]?\d*\.\d+|\d+)\n'
    fl = np.float(
        re.findall(regex, string)[0]
    )
    return(fl)

def extract_average_time_float(string):
    regex = '\:(\d*\.\d+|\d+)ms'
    fl = np.float(
        re.findall(regex, string)[0]
    )
    return(fl)

def extract_time_from_str(string):
    regex = '([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+)'
    time = datetime.datetime.strptime((re.findall(regex, string)[0]), '%H:%M:%S.%f').time()
    return(time)

