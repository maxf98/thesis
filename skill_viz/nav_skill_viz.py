import json
import matplotlib.pyplot as plt
import os

# need to visualise the maze used in the experiments...
import maze_env.mazes as mazes
maze_type = 'square_a'
ENV_LIMS = dict(
    square_a=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_bottleneck=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 9.55), x=(-0.5, 9.5), y=(-0.5, 9.5)),
    square_corridor=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_corridor2=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_tree=dict(xlim=(-6.55, 6.55), ylim=(-6.55, 0.55), x=(-6.5, 6.5), y=(-6.5, 0.5))
)

# expects a folder which contains log files
LOG_DIR = "logs"

NUM_SKILLS = 10
NUM_TRAJECTORIES = 20


# determines colors assigned to skills
def get_cmap():
    if NUM_SKILLS <= 10:
        cmap = plt.get_cmap('tab10')
    elif 10 < NUM_SKILLS <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('viridis', NUM_SKILLS)
    return cmap


# formatting
def config_subplot(ax, title=None, extra_lim=0., fontsize=14):
    env_config = ENV_LIMS[maze_type]
    ax.set_xlim(env_config["xlim"][0] - extra_lim, env_config["xlim"][1] + extra_lim)
    ax.set_ylim(env_config["ylim"][0] - extra_lim, env_config["ylim"][1] + extra_lim)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_visible(False)


def plot_maze(ax):
    maze = mazes.mazes_dict[maze_type]['maze']
    maze.plot(ax)


def plot_all_skills(cmap, trajectories, figsize=(5, 5), alpha=0.2, linewidth=2):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    plot_maze(ax)
    for i in range(NUM_TRAJECTORIES):
        for skill in range(NUM_SKILLS):
            traj = trajectories[skill][i]
            xs = [step[0] for step in traj]
            ys = [step[1] for step in traj]
            ax.plot(xs, ys, label="Skill #{}".format(skill), color=cmap(skill), alpha=alpha,
                    linewidth=linewidth, zorder=10)

    # mark starting point
    ax.plot(trajectories[0][0][0][0], trajectories[0][0][0][1], marker='o', markersize=8, color='black', zorder=11)

    config_subplot(ax, title="Skills")

    return ax


# currently produces [[[[]]]]...
# skill -> trajectory -> step -> x, y
def load_data(path):
    skill_trajectories = [[] for _ in range(NUM_SKILLS)]
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.endswith(".json"):
                continue
            with open(entry) as file:
                data = json.load(file)
                for i in range(NUM_TRAJECTORIES):
                    trajectory = data[str(i)]
                    skill = trajectory[0]['skill']
                    states = [step['state'] for step in trajectory]
                    skill_trajectories[skill].append(states)

    return skill_trajectories


cmap = get_cmap()
skill_trajectories = load_data('logs')
plot_all_skills(cmap, skill_trajectories)
plt.show()