from glob import glob
import os
import sys
import argparse
import itertools as it
import subprocess


parser = argparse.ArgumentParser(description="Submit jobs.",
									add_help=True)
parser.add_argument('-ml', action='store', dest='mls', type=str,
					default=(
						'fomo_nsga2_lr_fnr'
						',fomo_nsga2_lr_fnr_linear'
						',fomo_nsga2_lr_fnr_mlp'
						',fomo_nsga2_xgb_fnr'
						',fomo_nsga2_xgb_fnr_linear'
						',fomo_nsga2_xgb_fnr_mlp'
						)
)
parser.add_argument('-seeds', action='store', type=str, dest='SEEDS',
					default='14724,24284,31658,6933,1318,16695,27690,8233,24481,6832,'
					'13352,4866,12669,12092,15860,19863,6654,10197,29756,14289,'
					'4719,12498,29198,10132,28699,32400,18313,26311,9540,20300')
parser.add_argument('-datasets', action='store', type=str, 
					default='student,adult,lawschool,communities,lsac_bar')
parser.add_argument('-datadir', action='store', type=str, 
					default='dataset')
parser.add_argument('-rdir', action='store', 
		default='results/', type=str, help='Results directory')
parser.add_argument('-n_trials', action='store', dest='N_TRIALS', default=20,
					type=int, help='Number of trials to run')
parser.add_argument('-n_jobs', action='store', default=1,
					type=int, help='Number of parallel jobs')
parser.add_argument('-mem', action='store', dest='mem', default=1000, type=int,
					help='memory request and limit (MB)')
parser.add_argument('--slurm', action='store_true',
					default=False, help='Run on an slurm HPC')
parser.add_argument('-time', action='store', dest='time', 
					default='01:10:00', type=str, help='time in HR:MN:SS')
args = parser.parse_args()

n_trials = len(args.SEEDS) if args.N_TRIALS < 1 else args.N_TRIALS
seeds = args.SEEDS.split(',')[:n_trials]
mls = args.mls.split(',')
datasets = args.datasets.split(',')
print('running these datasets:', datasets)
print('and these methods:', mls)
print('using these seeds:', seeds)

q = 'ECODE'

# write run commands
all_commands = []
job_info = []
# submit per dataset,trial,learner
for seed, dataset, ml in it.product(seeds, datasets, mls):
	rdir = '/'.join([args.rdir, dataset])+'/'
	os.makedirs(rdir, exist_ok=True)

	datafile = '/'.join([args.datadir,dataset])+'.csv'

	all_commands.append(
		f'python evaluate.py -data {datafile} -ml {ml} -rdir {rdir} -seed {seed}'
	)

	job_info.append({
		'dataset': dataset,
		'ml': ml,
		'seed': seed,
		'rdir': rdir
	})

print(len(job_info), 'total jobs created')
if args.slurm:
	# write a jobarray file to read commans from
	jobarrayfile = 'jobfiles/joblist.txt'
	os.makedirs('jobfiles', exist_ok=True)
	for i, run_cmd in enumerate(all_commands):
		# mode = 'w' if i == 0 else 'a'
		# with open(jobarrayfile, mode) as f:
		# 	f.write(f'{run_cmd}\n')

		job_name = '_'.join([f'{job_info[i][x]}' for x in
							['ml','dataset','seed']])
		job_file = f'jobfiles/{job_name}.sbatch'
		out_file = job_info[i]['rdir'] + job_name + '_%J.out'
		# error_file = out_file[:-4] + '.err'

		batch_script = (
			f"""#!/usr/bin/bash 
#SBATCH --output={out_file} 
#SBATCH --job-name={job_name} 
#SBATCH --partition={q} 
#SBATCH --ntasks={1} 
#SBATCH --cpus-per-task={1} 
#SBATCH --time={args.time}
#SBATCH --mem={args.mem} 
#SBATCH --nodelist=rdt01694

echo $CONDA_EXE run -n exp-fomo {run_cmd}
$CONDA_EXE run -n exp-fomo {run_cmd}
"""
		)

		with open(job_file, 'w') as f:
			f.write(batch_script)

		print(run_cmd)
		# print(job_file, ':')
		# print(batch_script)
		sbatch_response = subprocess.check_output(
			[f'sbatch {job_file}'], shell=True).decode()     # submit jobs
		print(sbatch_response)
