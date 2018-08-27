import numpy as np
import re

def strings_in_line(line, strings):
    evaluation = np.any([string in line for string in strings])
    return(evaluation)

def end_on_same_line(line):
    evaluation = True
    return(evaluation)

def converged_line_list_transform(line_list, **kwargs):
    regex = '\((.*)[<|>]'
    str_list = re.findall(regex, line_list[0])[0].strip().split()
    float_list = [np.float(i) for i in str_list]
    return(float_list)
