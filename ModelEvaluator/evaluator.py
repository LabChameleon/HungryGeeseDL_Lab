from multiagent import Agent as multiAgent
from multiagent_prev_enemy_map_h_b import Agent as multiAgent_h_b
from kaggle_environments import make
import os

agent = multiAgent_h_b("training_adam-opposite_actions-target_net-clamp_env-reward_prev-enemy", 30000)
agent1 = multiAgent("training_adam-opposite_actions-target_net-clamp_env-reward", 50000)
agent2 = multiAgent("training_adam-opposite_actions-target_net-clamp_env-reward", 80000)
agent3 = multiAgent("training_adam-opposite_actions-target_net-clamp_env-reward", 80000)

num_runs = 10

env = make("hungry_geese", debug=True)
for i in range(num_runs):
    env.reset()
    env.run([agent.agent, "greedy","greedy","greedy"])
    html = env.render(mode="html", width=700, height=600)

    if not os.path.exists("./runs/" + agent.agent_class + "-" + agent.experiment_name):
        os.makedirs("./runs/" + agent.agent_class + "-" + agent.experiment_name)

    f = open("./runs/"+ agent.agent_class + "-" + agent.experiment_name +"/goose_run_{}.html".format(i), "w")
    print(i)
    f.write(html)
    f.close()

