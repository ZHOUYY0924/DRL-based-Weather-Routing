import os
import argparse
from utils import create_directory, plot_learning_curve
from D3QN_action import D3QN
from environment_ocean import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
matplotlib.rc("font", family='Microsoft YaHei')

envpath = '/home/xgq/conda/envs/pytorch1.6/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=15000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/D3QN/')
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')

args = parser.parse_args()

save_path = r'D3QN_picture'
def main():
    num_of_select_dir = 3
    num_of_select_speed = 5
    action_dim = num_of_select_dir * num_of_select_speed
    waterway_csv_path = r"ship_net_combined.csv"
    waterway_data = pd.read_csv(waterway_csv_path)
    waterway_data_LON = waterway_data['lon']
    waterway_data_LAT = waterway_data['lat']
    waterway_data_label = waterway_data['label']

    x_start = [waterway_data_LON.tolist()[0], waterway_data_LAT.tolist()[0]]
    x_goal = [waterway_data_LON.tolist()[-1], waterway_data_LAT.tolist()[-1]]

    env = Environment(x_start, x_goal, max_step=len(set(waterway_data_label)),waterway_csv_path=waterway_csv_path)
    state_dim = len(env.initial_state)

    agent = D3QN(alpha=0.05, state_dim=state_dim, action_dim=action_dim,
                 fc1_dim=64, fc2_dim=256*2, fc3_dim=256, fc4_dim=32,
                 ckpt_dir=args.ckpt_dir, gamma=0.95, tau=0.05, epsilon=1.0,
                 eps_end=0.01, eps_dec=0.0001, max_size=1000 * 1*4, batch_size=64 * 4 * 4)

    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    # agent.load_models(15000)
    total_rewards, avg_rewards, epsilon_history = [], [], []
    Number_of_successful_rounds = []


    time_start = time.time()
    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        finish = False
        observation = env.reset()
        LON_list = [observation[0]]
        LAT_list = [observation[1]]
        t_lon_list = []
        t_lat_list = []
        t_action_list = []
        t_label_list = []
        t_reward_list = []
        t_swh_pos_list = []
        t_swd_pos_list = []
        t_swp_pos_list = []
        t_ws_pos_list = []
        t_wd_pos_list = []
        t_cs_pos_list = []
        t_cd_pos_list = []
        t_oil_total_list = []
        t_duration_list = []
        t_sclected_speed_list = []
        t_sclected_dir_list = []
        t_oil_cus_list = []
        t_V_real_list = []
        t_duration_list2 = []
        t_heading_list = []

        n = 0
        while not done:
            n += 1
            action = agent.choose_action(observation, isTrain=True)
            observation_, reward, done, finish, duration, oil_total, sclected_speed, sclected_dir, swh_pos_list, swd_pos_list, swp_pos_list, ws_pos_list, wd_pos_list, cs_pos_list, cd_pos_list, oil_cus_list, duration_list, V_real_list, heading_list = env.step(
                action, num_of_select_dir)

            t_lon_list.append(observation_[0])
            t_lat_list.append(observation_[1])
            t_oil_total_list.append(oil_total)
            t_duration_list.append(duration)
            t_action_list.append(action)
            t_sclected_speed_list.append(sclected_speed)
            t_sclected_dir_list.append(sclected_dir)
            t_label_list.append(n)
            t_reward_list.append(reward)
            t_swh_pos_list.extend(swh_pos_list)
            t_swd_pos_list.extend(swd_pos_list)
            t_swp_pos_list.extend(swp_pos_list)
            t_ws_pos_list.extend(ws_pos_list)
            t_wd_pos_list.extend(wd_pos_list)
            t_cs_pos_list.extend(cs_pos_list)
            t_cd_pos_list.extend(cd_pos_list)
            t_oil_cus_list.extend(oil_cus_list)
            t_duration_list2.extend(duration_list)
            t_V_real_list.extend(V_real_list)
            t_heading_list.extend(heading_list)

            agent.remember(observation, action, reward, observation_, done)
            if (episode + 1) % 2 == 0:
                agent.learn()
            total_reward += reward
            observation = observation_
            LON_list.append(observation_[0])
            LAT_list.append(observation_[1])
        agent.decrement_epsilon()
        Number_of_successful_rounds.append(finish)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        epsilon_history.append(agent.epsilon)
        print('EP:{} Reward:{:.2f} Avg_reward:{:.2f} Duration:{:.2f} oil_total:{:.2f} Epsilon:{} Finish:{}'.
              format(episode + 1, total_reward, avg_reward, duration, oil_total, agent.epsilon, finish))

        if (episode + 1) % 100 == 0:
            agent.save_models(episode + 1)
        if (episode + 1) % 100 == 0 or episode == 0:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, edgecolor='black')
            ax.scatter([x_start[0]+180], [x_start[1]], marker='o')
            ax.scatter([x_goal[0]+180], [x_goal[1]], marker='o')
            ax.scatter([each+180 for each in waterway_data_LON],waterway_data_LAT, marker='x',c='cyan',s=1)
            ax.scatter([each+180 for each in LON_list], LAT_list,c='r',s=1)
            plt.title('reward:'+str(round(total_reward,2))+', duration:'+str(round(duration,2))+', oil:'+str(round(oil_total,2)))
            plt.savefig(save_path + '//' + str(episode + 1) + '.png')
        plt.ioff()
        plt.close()
    time_end = time.time()
    duration_time = time_end - time_start
    print('耗时：', round(duration_time,2), 's')
    episodes = [i+1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_rewards, title='Reward', ylabel='reward',
                        figure_file=args.reward_path)
    plot_learning_curve(episodes, epsilon_history, title='Epsilon', ylabel='epsilon',
                        figure_file=args.epsilon_path)

if __name__ == '__main__':
    main()
