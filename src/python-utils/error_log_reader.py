import numpy as np
import os

def get_info(
    search_dir, **kwargs
):
    file_search_str_list = kwargs.get('file_search_str_list')
    file_list = kwargs.get('file_list', [])
    parse_error_log_args = kwargs.get('parse_error_log_args')
    processing_func = kwargs.get('processing_func')
    print(file_search_str_list, file_list)

    path_list = file_search(
        search_dir, file_search_str_list
    )
    path_list = path_list + file_list
    out_list = []
    for fpath in path_list:
        reader_out = parse_error_log(
            fpath,
            **parse_error_log_args
	)
        processed_list = processing_func(
            reader_out
	)
        out_list.append(processed_list)
    return(out_list)

def parse_error_log(
    fpath,
    **kwargs
):
    """
    Parses obsidian error logs line by line.
    :param fpath: full path of file
    :type fpath: str
    :param start_signal_func_list: evaluates true when start recording
    :type start_signal_func_list: list of functions
    :param end_signal_func_list: evaluates true when stop recording
    :type end_signal_func_list: list of functions
    :return output_list: list of recordings
    :type output_list: list
    :return time_list: list of times of recordings
    :type time_list: list 

    description:
    open the file at fpath,
    iterate over each line, 
    if start_signal_func evaluates to true on a line, start recording from that line (inclusive),
    if end_signal_func evaluates to true on a line, end recording on that line (not inclusive),
    recording means to add the lines to a list from where start_signal_func_evaluates to true (inclusive)
    to where end_signal_func evaluates to true (not inclusive)
    optionally format each line and/or optionally format each recording
    """
    start_signal_func_list = kwargs.get('start_signal_func_list')
    end_signal_func_list = kwargs.get('end_signal_func_list')

    recording = False
    output_list = []
    end_signal_func_standard = lambda line, **kwargs: False
    end_signal_func = end_signal_func_standard
    
    with open(fpath, 'r') as f:
        for idx, line in enumerate(f):
            line = f.readline()
            if (not recording):
                evaluation_list = [
                    start_signal_func(line) 
                    for start_signal_func in start_signal_func_list
                    ]
                evaluation = np.any(evaluation_list)
                if evaluation: 
                    idx = np.where(evaluation_list)[0][0]
                    end_signal_func = end_signal_func_list[idx]
                    recording = True
                    line_list = []
            elif end_signal_func(line):
                recording = False
                end_signal_func = end_signal_func_standard
                if kwargs.get('line_list_transform_func'):
                    line_list = kwargs.get('line_list_transform_func')(line_list, **kwargs)
                output_list.append(line_list)
            if recording:
                if kwargs.get('line_transform_func'):
                    line = kwargs.get('line_transform_func')(line, **kwargs)
                if line:
                    line_list.append(line)
    return(output_list)

def file_search(
    search_dir,
    search_str_list,
    return_first_item = False
):
    matching_list = []
    for item in os.listdir(search_dir):
        if os.path.isfile(os.path.join(search_dir, item)):
            if np.any([search_str in item for search_str in search_str_list]):
                matching_list.append(item)
    path_list = [os.path.join(search_dir, item) for item in matching_list]
    if return_first_item: path_list = path_list[0]
    print(path_list)
    return(path_list)

