import itertools
import argparse


states = ["rsaps", "classic"]
actions = ["acquisition_func", "exploration_weight"]
rewards = ["rsaps", "log_mean_regret"]
agents = ["deep_q-0.01", "deep_q-0.001"]
benchs = ["camelback", "goldsteinprice", "hartmann3", "hartmann6"]

script = []

task_id = 0
for state in states:
    for action in actions:
        for reward in rewards:
            for agent in agents:
                for bench in benchs:
                    task_id += 1
                    script += [
                        "if [ {} -eq $SLURM_ARRAY_TASK_ID ]; then".format(task_id),
                        "    python ~/git/thesis/SMAC3/train/train_smac.py \\",
                        "        --agent {} \\".format(agent.split("-")[0]),
                        "        --num_episodes 1000 \\",
                        "        --eval_num_episodes 100 \\",
                        "        --eps_decay_starts 1000 \\",
                        "        --lr {} \\".format(agent.split("-")[-1]),
                        "        --c_freq 50 \\",
                        "        --c_dir /home/bozkurth/git/thesis/SMAC3/train/final/{} \\".format("-".join([agent, state, action, reward, bench])),
                        "        --obs {} \\".format(state),
                        "        --act {} \\".format(action),
                        "        --rew {} \\".format(reward),
                        "        --bench {} \\".format(bench),
                        "        --horizon 75 \\",
                        "        --act_repeat 5",
                        "    exit $?",
                        "fi",
                        "",
                    ]

script = [
    "#!/bin/bash",
    "#SBATCH -p cpu_ivy",
    "#SBATCH --mem 32000",
    "#SBATCH -t 7-00:00",
    "#SBATCH -c 1",
    "#SBATCH -a 1-{}".format(task_id),
    "#SBATCH -D /home/bozkurth/git/thesis/SMAC3/train/final",
    "#SBATCH -o /home/bozkurth/git/thesis/SMAC3/train/final/log/%x.%N.%A.%a.out",
    "#SBATCH -e /home/bozkurth/git/thesis/SMAC3/train/final/log/%x.%N.%A.%a.err",
    "#SBATCH --mail-type=END,FAIL",
    "",
    "source activate the",
    "",
] + script

script = "\n".join(script)

with open("rlsmac.txt", "w") as f:
    f.write(script)
















# def create_configs(defaults={}, *args):
#     for elem in itertools.product(*args):
#         elem_config = defaults.copy()
#         for el_config in elem:
#             elem_config.update(el_config)
#         yield elem_config


# def create_arguments(config):
#     s = []
#     for k, v in config.items():
#         s.append(f"        --{k} {v}")
#     return " \\\n".join(s)


# def create_surrogate(task_id):
#     s1 = [
#         "if [ {} -eq $SLURM_ARRAY_TASK_ID ]; then".format(task_id),
#         "    python ~/git/thesis/SMAC3/train/train_smac.py \\",
#     ]
#     s2 = [
#         "    exit $?",
#         "fi",
#     ]
#     return "\n".join(s1), "\n".join(s2)


# def run(cluster):
#     agents = [
#         {"agent": "deep_q", "lr": 0.01, "eps_decay_starts": 1000, "num_checkpoints": 10},
#         {"agent": "deep_q", "lr": 0.001, "eps_decay_starts": 1000, "num_checkpoints": 10},
#         {"agent": "deep_q", "lr": 0.0001, "eps_decay_starts": 1000, "num_checkpoints": 10},
#         {"agent": "random"},
#     ]

#     bench = [
#         {"bench": "camelback"},
#         {"bench": "goldsteinprice"},
#         {"bench": "hartmann3"},
#         {"bench": "hartmann6"},
#     ]

#     defaults = {
#         "num_episodes": 1000,
#         "horizon": 75,
#         "obs": "rsaps",
#         "act": "exploration_weight",
#         "rew": "rsaps",
#     }

#     header = [
#         "#!/bin/bash",
#         "#SBATCH -p {}".format(cluster),
#         "#SBATCH --mem 32000",
#         "#SBATCH -t 7-00:00",
#         "#SBATCH -c 2",
#         "#SBATCH -a 1-{}".format(len(bench) * len(agents)),
#         "#SBATCH -D /home/bozkurth/git/thesis/SMAC3/train/final",
#         "#SBATCH -o /home/bozkurth/git/thesis/SMAC3/train/final/log/%x.%N.%A.%a.out",
#         "#SBATCH -e /home/bozkurth/git/thesis/SMAC3/train/final/log/%x.%N.%A.%a.err",
#         "#SBATCH --mail-type=END,FAIL",
#         "",
#         "source activate the",
#     ]

#     script = "\n".join(header) + "\n\n"
#     for task_id, config in enumerate(create_configs(defaults, bench, agents)):
#         s1, s2 = create_surrogate(task_id + 1)
#         s = create_arguments(config)
#         script += s1 + "\n" + s + "\n" + s2 + "\n"

#     with open("rlsmac.txt", "w") as f:
#         f.write(script)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cluster", default="cpu_ivy")

#     args = parser.parse_args()

#     run(args.cluster)
