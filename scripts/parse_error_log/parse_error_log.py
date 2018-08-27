import numpy as np
import os
import pandas as pd
import subprocess
import sys

import error_log_reader
import misc_functions

#print(str(sys.argv))
data_dir = sys.argv[1]
fname = sys.argv[2]
#print(data_dir)
#fname = [fname for fname in os.listdir(data_dir) if 'obsidian.pbs.e' in fname][0]
fname = os.path.join(data_dir,fname)

# last stats table
fname_stats = os.path.join(data_dir, 'last_stats_table.csv')
if not os.path.isfile(fname_stats):
	result = subprocess.run(
		'/project/RDS-FSC-obsidian-RW/obsidian-dk/scripts/parse_error_log/parse_error_log.sh {}'.format(fname),
		shell=True,
		stdout=subprocess.PIPE
	)
	out = result.stdout.decode('utf-8').strip('\n')

	str_list = out.replace('-', '').split(' ')
	str_list = list(filter(None, str_list))
	len_chunk = 10
	str_list = [str_list[x:x+len_chunk] for x in range(0, len(str_list), len_chunk)]
	df_last_stats = pd.DataFrame(str_list)
	df_last_stats.columns = df_last_stats.iloc[0]
	df_last_stats = df_last_stats.reindex(df_last_stats.index.drop(0))

	df_last_stats.to_csv(fname_stats, index = False, header = True)

# last rhat table
"""
fname_converge = os.path.join(data_dir, 'last_convergence.csv')
if not os.path.isfile(fname_converge):
	line_no_converge = 257
	kwargs_convergence = lambda line_no : dict(
	    file_search_str_list = ['obsidian.pbs.e', 'obsidian_error.txt'],
	    parse_error_log_args = dict(
		start_signal_func_list = [
		    lambda line: misc_functions.strings_in_line(
			line, 
			[
			    'mcmc.hpp:{}'.format(line_no),
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

	out_convergence = error_log_reader.get_info(
	    data_dir,
	    **kwargs_convergence(line_no_converge)
	)[0]

	arr = out_convergence[-1]
	df = pd.DataFrame(
	    [np.array(list(range(len(arr)))), arr]
	).T
	df.columns = ['ChainID', 'Rhat']
	df['ChainID'] = df['ChainID'].astype(int)

	df.to_csv(fname_converge, index = False)
"""
