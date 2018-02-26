import re
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_error_logs(
    fpath,
    start_signal_func,
    end_signal_func,
    **kwargs
):
    """
    in
        fpath: str, full path of file
        start_signal_func: function
        end_signal_func: function
    out
        output_list: list of recordings
        time_list: list of times of recordings

    description:
    open the file at fpath,
    iterate over each line, 
    if start_signal_func evaluates to true on a line, start recording from that line (inclusive),
    if end_signal_func evaluates to true on a line, end recording on that line (not inclusive),
    recording means to add the lines to a list from where start_signal_func_evaluates to true (inclusive)
    to where end_signal_func evaluates to true (not inclusive)
    optionally format each line and/or optionally format each recording
    """
    recording = False
    output_list = []
    time_list = []
    
    with open(fpath, 'r') as f:
        for idx, line in enumerate(f):
            line = f.readline()
            if start_signal_func(
                line,
                **kwargs
            ):
                recording = True
                time_list.append(line)
                line_list = []
            elif end_signal_func(
                line, recording, 
                **kwargs
            ):
                recording = False
                if kwargs.get('line_list_transform_func'):
                    line_list = kwargs.get('line_list_transform_func')(line_list, **kwargs)
                output_list.append(line_list)
            if recording:
                if kwargs.get('line_transform_func'):
                    line = kwargs.get('line_transform_func')(line, **kwargs)
                if line:
                    line_list.append(line)
    return(output_list, time_list)

# signal functions
def start_signal_str(line, **kwargs):
    """return true if start_signal_str is in the line
    """
    evaluation = kwargs.get('start_signal_str') in line
    return(evaluation)

def end_signal_str(line, **kwargs):
    """return true if recording is on and end_signal_str is in the line
    """
    evaluation = recording and kwargs.get('end_signal_str') in line
    return(evaluation)

def end_signal_new_line(line, recording, **kwargs):
    """return true if recording is on
    """
    return(recording)

def end_signal_stats_table(line, recording, **kwargs):
    """return true if recording is on and the line is equal to a new line only
    """
    evaluation = recording and ('\n' == line)
    return(evaluation)

# stats table functions
def stats_table_line_transform(line, **kwargs):
    line = line.strip()
    return(line)

def stats_table_line_list_transform(line_list, **kwargs):
    new_line_list = line_list[1:]
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
def converged_line_transform(line, **kwargs):
    return(line)

def converged_line_list_transform(line_list, **kwargs):
    regex = '\((.*)[<|>]'
    str_list = re.findall(regex, line_list[0])[0].strip().split()
    float_list = [np.float(i) for i in str_list]
    return(float_list)

# likelihood functions
def likelihood_line_transform(line, **kwargs):
    return(line)

def likelihood_line_list_transform(line_list, **kwargs):
    return(line_list[0])

def parse_likelihoods(
    find_str_list, 
    search_str_list,
):
    """post-process output from likelihood str search
    for each entry in search_str_list, find the first str in search_str_list in the entry (if any)
    """
    out_list = [[] for find_str in find_str_list]
    for line in search_str_list:
        idx_list = [
            idx for idx, find_str in enumerate(find_str_list)
            if find_str in line.lower()
        ]
        if idx_list: 
            ll = extract_float_from_str(line)
            out_list[idx_list[0]].append(ll)
    return(out_list)

# helper functions
def is_float_try(str):
    """try cast str as a float
    """
    try:
        float(str)
        return True
    except ValueError:
        return False

def extract_float_from_str(string):
    regex = '([-+]?\d*\.\d+|\d+)\n'
    fl = np.float(
        re.findall(regex, string)[0]
    )
    return(fl)

def extract_time_from_str(string):
    regex = '([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+)'
    time = datetime.datetime.strptime((re.findall(regex, string)[0]), '%H:%M:%S.%f').time()
    return(time)
