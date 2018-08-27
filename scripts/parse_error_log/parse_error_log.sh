#mcmc_line_no=247
mcmc_line_no=265
no_lines=`< $1 wc -l`
backwards_line_no=`tac $1 | grep -m 1 -n "mcmc.hpp:$mcmc_line_no" | cut -d : -f 1 | tail -1`
line_start_no=$((no_lines - backwards_line_no + 1))
next_line_no=$((line_start_no + 1))
line_text=`sed "${line_start_no}q;d" $1`
length=`sed -n "$next_line_no,$ p" $1 | grep -m 1 -n I0 | cut -d : -f 1 | tail -1`
line_end_no=$((line_start_no + length - 1))
text=`sed -n "${next_line_no},${line_end_no}p" $1`
echo $text
