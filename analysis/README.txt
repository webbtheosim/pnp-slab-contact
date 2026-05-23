# Summary of directory structure

There is a summary of all derived data and cases that wre run in sweep_summary_added.csv

The main sweeps that were run also have generated plots and associated csvs. These are asymmetry, diff, gap, peclet, rate, and salt. Peclet and rate are the same sweep, just using a different representation of the x-axis. 

Within the summary csv you can see that there is a label of "cases" which provides a code name for the simulation (tied to those parameters and the sweep). 

Each case also has a sub-directory that has generated plots as well as accompanying .csv file for the data used to make those plots. For reference these include

1) "03" which is the end of the equilibration period (the relevant initial state
2) "06" which is the end of the approach period
3) "08" or "09" which is a time during the contact (middle or end, right before pull away)
4) "12" which is the time at the end of the separation period
