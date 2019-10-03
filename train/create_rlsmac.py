import itertools
import argparse


def create_configs(defaults={}, *args):
    for elem in itertools.product(*args):
        elem_config = defaults.copy()
        for el_config in elem:
            elem_config.update(el_config)
        yield elem_config


def create_arguments(config):
    s = []
    for k, v in config.items():
        s.append(f"        --{k} {v}")
    return " \\\n".join(s)


def create_surrogate(task_id):
    s1 = [
        "if [ {} -eq $SLURM_ARRAY_TASK_ID ]; then".format(task_id),
        "    python ~/git/thesis/SMAC3/train/train_smac.py \\",
    ]
    s2 = [
        "    exit $?",
        "fi",
    ]
    return "\n".join(s1), "\n".join(s2)


def run(cluster):
    agents = [
        {"agent": "deep_q", "lr": 0.01, "eps_decay_starts": 500, "num_checkpoints": 10},
        {"agent": "deep_q", "lr": 0.001, "eps_decay_starts": 500, "num_checkpoints": 10},
        {"agent": "random"},
        {"agent": "default_smac", "verbose": "INFO"},
    ]

    bench = [
        {"bench": "camelback"},
        {"bench": "goldsteinprice"},
        {"bench": "hartmann3"},
        {"bench": "hartmann6"},
    ]

    defaults = {
        "num_episodes": 10000,
    }

    header = [
        "#!/bin/bash",
        "#SBATCH -p {}".format(cluster),
        "#SBATCH --mem 32000",
        "#SBATCH -t 7-00:00",
        "#SBATCH -c 2",
        "#SBATCH -a 1-16 # array size",
        "#SBATCH -D /home/bozkurth/git/thesis/SMAC3/train",
        "#SBATCH -o log/%x.%N.%A.%a.out",
        "#SBATCH -e log/%x.%N.%A.%aerr",
        "#SBATCH --mail-type=END,FAIL",
        "",
        "source activate the",
    ]

    script = "\n".join(header) + "\n\n"
    for task_id, config in enumerate(create_configs(defaults, bench, agents)):
        s1, s2 = create_surrogate(task_id + 1)
        s = create_arguments(config)
        script += s1 + "\n" + s + "\n" + s2 + "\n"

    with open("rlsmac.txt", "w") as f:
        f.write(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", default="cpu_ivy")

    args = parser.parse_args()

    run(args.cluster)