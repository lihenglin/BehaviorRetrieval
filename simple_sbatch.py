import tempfile
import subprocess
import os
import argparse
import copy

SLURM_LOG_DEFAULT = os.path.join(YOUR_REPO, "slurm_logs")
ENV_SETUP_SCRIPT = os.path.join(YOUR_REPO, "setup_shell.sh")

SLURM_ARGS = {
    "account" : {"type": str, "required": False, "default": "iliad"},
    "partition" : {"type": str, "required": False, "default": "iliad"},
    "time" : {"type": str, "default": "24:00:00"},
    "nodes": {"type": int, "default": 1},
    "ntasks-per-node": {"type": int, "default": 1},
    "cpus": {"type": int, "required": True},
    "gpus": {"type": str, "required": False, "default": None},
    "mem": {"type": str, "required": True},
    "output": {"type" : str, "default": SLURM_LOG_DEFAULT},
    "error": {"type" : str, "default": SLURM_LOG_DEFAULT},
    "job-name" : {"type": str, "required": False, "default": "sbatch"},
    "exclude" : {"type": str, "required": False, "default": None},
    "nodelist": {"type": str, "required": False, "default": None}
}

SLURM_NAME_OVERRIDES = {
    "gpus" : "gres",
    "cpus": "cpus-per-task"
}

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    """
    if "=" in s:
        items = s.split('=')
        key = items[0].strip() # we remove blanks around keys, as is logical
        if len(items) > 1:
            # rejoin the rest:
            value = '='.join(items[1:])
        return (key, value)
    else:
        key = s.strip()
        return (key, None)

def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

def get_jobs(args):
    all_jobs = []
    if len(args.entry_point) < len(args.arguments):
        # If we only were given one entry point but many script arguments, replicate the entry point
        assert len(args.entry_point) == 1
        args.entry_point = args.entry_point * len(args.arguments)

    for entry_point, arguments in zip(args.entry_point, args.arguments):
        job = parse_vars(arguments)
        if args.seeds_per_job > 1:
            # copy all of the configratuions and add seeds
            seed = int(job.get('seed', 0))
            for i in range(args.seeds_per_job):
                seeded_job = job.copy() # Should be a shallow dictionary, so copy OK
                seeded_job['seed'] = seed + i
                all_jobs.append((entry_point, seeded_job))
        else:
            all_jobs.append((entry_point, job))
    return all_jobs

def write_slurm_header(f, args):
    # Make a copy of the args to prevent corruption
    args = copy.deepcopy(args)
    # Modify everything in the name space to later write it all at once
    for key in SLURM_ARGS.keys():
        assert key.replace('-', '_') in args, "Key " + key + " not found."
    
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not os.path.isdir(args.error):
        os.makedirs(args.error)

    args.output = os.path.join(args.output, args.job_name + "_%A.out")
    args.error = os.path.join(args.error, args.job_name + "_%A.err")
    args.gpus = "gpu:" + str(args.gpus) if args.gpus is not None else args.gpus

    NL = '\n'
    f.write("#!/bin/bash" + NL)
    f.write(NL)
    for arg_name in SLURM_ARGS.keys():
        arg_value = vars(args)[arg_name.replace('-', '_')]
        if arg_name in SLURM_NAME_OVERRIDES:
            arg_name = SLURM_NAME_OVERRIDES[arg_name]
        if not arg_value is None:
            f.write("#SBATCH --" + arg_name + "=" + str(arg_value) + NL)
    
    f.write(NL)
    f.write('echo "SLURM_JOBID = "$SLURM_JOBID' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_NNODES = "$SLURM_NNODES' + NL)
    f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR' + NL)
    f.write('echo "working directory = "$SLURM_SUBMIT_DIR' + NL)
    f.write(NL)
    f.write(". " + ENV_SETUP_SCRIPT)
    f.write(NL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-point", type=str, action='append', default=None)
    parser.add_argument("--seeds-per-job", type=int, default=1)
    parser.add_argument("--arguments", metavar="KEY=VALUE", nargs='+', action='append', help="Set kv pairs used as args for the entry point script.")
    # Add Slurm Arguments
    for k, v in SLURM_ARGS.items():
        parser.add_argument("--" + k, **v)
    args = parser.parse_args()
    assert args.entry_point is not None, "Must provide at least one entry point."

    jobs = get_jobs(args)

    # Call python subprocess to launch the slurm jobs.
    procs = []
    for current_job in jobs:
        _, slurm_file = tempfile.mkstemp(text=True, prefix='job', suffix='.sh')
        print("Launching job with slurm configuration:", slurm_file)

        with open(slurm_file, 'w+') as f:
            write_slurm_header(f, args)
            # Now that we have written the header we can launch the jobs.
            entry_point, script_args = current_job
            command_str = ['python', entry_point]
            for arg_name, arg_value in script_args.items():
                command_str.append("--" + arg_name)
                if arg_value is not None:
                    command_str.append(str(arg_value))
            command_str = ' '.join(command_str) + '\n'
            f.write(command_str)
            
            # Now launch the job
            proc = subprocess.Popen(['sbatch', slurm_file])
            procs.append(proc)

    exit_codes = [p.wait() for p in procs]
