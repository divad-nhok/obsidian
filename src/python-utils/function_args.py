import misc_functions
import numpy as np

kwargs_likelihood = dict(
    file_search_str_list = ['shard.pbs.e', 'shard_small.pbs.e'],
    parse_error_log_args = dict(
        start_signal_func_list = [
            lambda line: misc_functions.string_in_line(line, 'likelihood')
        ],
        end_signal_func_list = [
            misc_functions.end_on_same_line
        ],
        line_list_transform_func = misc_functions.likelihood_line_list_transform,
    ),
    processing_func = lambda out: misc_functions.parse_likelihoods(
        out,
        find_str_list = [
            'magnetic likelihood sigma',
            'magnetics',
            'gravity likelihood sigma',
            'gravity'
        ],
        extraction_func = misc_functions.extract_likelihood_float
    )
)

kwargs_evaluationtime = dict(
    file_search_str_list = ['shard.pbs.e', 'shard_small.pbs.e'],
    parse_error_log_args = dict(
        start_signal_func_list = [
            lambda line: misc_functions.string_in_line(line, 'asyncworker.hpp:67')
        ],
        end_signal_func_list = [
            misc_functions.end_on_same_line
        ],
        line_list_transform_func = misc_functions.likelihood_line_list_transform,
    ),
    processing_func = lambda out: misc_functions.parse_likelihoods(
        out,
        find_str_list = [
            'Magnetic',
            'Gravity'
        ],
        extraction_func = misc_functions.extract_average_time_float
    )
)

kwargs_convergence = dict(
    file_search_str_list = ['obsidian.pbs.e', 'obsidian_error.txt'],
    parse_error_log_args = dict(
        start_signal_func_list = [
            lambda line: misc_functions.strings_in_line(
                line, 
                [
                    'mcmc.hpp:270',
                ]
            )
        ],
        end_signal_func_list = [
            misc_functions.end_on_same_line
        ],
        line_list_transform_func = misc_functions.converged_line_list_transform,
    ),
    processing_func = lambda out: np.stack(out, axis = 0)
)

kwargs_statstable = dict(
    file_search_str_list = ['obsidian.pbs.e', 'obsidian_error.txt'],
    parse_error_log_args = dict(
        start_signal_func_list = [
            lambda line: misc_functions.strings_in_line(
                line, 
                [
                    'mcmc.hpp:260',
                ]
            )
        ],
        end_signal_func_list = [
            lambda line: misc_functions.string_in_line(line, 'I0')
        ],
        line_list_transform_func = misc_functions.stats_table_line_list_transform,
    ),
    processing_func = lambda out: out
)

kwargs_statstable2 = dict(
    file_search_str_list = ['obsidian.pbs.e', 'obsidian_error.txt'],
    parse_error_log_args = dict(
        start_signal_func_list = [
            lambda line: misc_functions.string_in_line(line, 'mcmc.hpp:290')
        ],
        end_signal_func_list = [
            lambda line: misc_functions.string_in_line(line, 'I0')
        ],
        line_list_transform_func = misc_functions.stats_table_line_list_transform,
    ),
    processing_func = lambda out: out
)
