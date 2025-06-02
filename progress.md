# Progress
We keep track of the scheduling features that can be simulated, and CP modeled using our environment

## Scheduling Machine Setups ($\alpha$)

Setup | Entry | Implemented | CP model
---- | :------:| :------: | :--------:
Single Machine | $1$ | $\checkmark$ | $\checkmark$
Identical Parallel | $Pm$ | $\checkmark$ | $\checkmark$
Uniform Parallel | $Qm$ | $\checkmark$ | $\checkmark$
Unrelated Parallel | $Rm$ |  |
Flow shop | $Fm$ | |
Flexible Flow shop | $FFm$ | |
Job shop | $Jm$ | $\checkmark$ | $\checkmark$
Flexible Job shop | $FJm$ | |
Open shop | $Om$ | $\checkmark$ | $\checkmark$


## Task constraints ($\beta$)

Constraint | Entry | Implemented | CP model
---- | :------:| :------: | :--------:
Preemptive | $pmpt$ | $\checkmark$ | $\checkmark$
Precedence | $prec$ | $\checkmark$ | $\checkmark$
No wait | $nwt$ | $\checkmark$ | $\checkmark$
Release times | $r_j$ | $\checkmark$ | $\checkmark$
Due times | $d_j$ | $\checkmark$ | $\checkmark$
Machine | $M_j$ | $\checkmark$ | $\checkmark$
Setup times | $s_{jk}$ | $\checkmark$ |
Batch processing | $batch(b)$ | |
Breakdowns | $brkdwn$ | |



## Objectives ($\gamma$)
Objective | Entry | Implemented | CP model
---- | :------:| :------: | :--------:
Makespan | $C_{\max}$ | $\checkmark$ | $\checkmark$
Total Completion time | $\Sigma C_j$ | $\checkmark$ | $\checkmark$
Weighted Completion time | $\Sigma w_j C_j$ | $\checkmark$ | $\checkmark$
Maximum Lateness | $L_{\max}$ | $\checkmark$ | $\checkmark$
Total Tardiness | $\Sigma T_j$ | $\checkmark$ |
Weighted Tardiness | $\Sigma w_j T_j$ | $\checkmark$ | $\checkmark$
Total Earliness | $\Sigma T_j$ | $\checkmark$ |
Weighted Earliness | $\Sigma w_j T_j$ | $\checkmark$ |
Total Tardy Jobs | $\Sigma U_j$ | |
Weighted Tardy Jobs | $\Sigma w_j U_j$ | |