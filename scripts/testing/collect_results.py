import os
import sys

configs_and_policies = [('lstm','lstm_maze_config'),('feudal','feudal_maze_config')]
mazes = ['VisionMaze3-v0','VisionMaze5-v0','VisionMaze9-v0','VisionMaze25-v0']
room_mazes = ['RoomMaze3-v0','RoomMaze5-v0','RoomMaze7-v0','RoomMaze9-v0']

all_mazes = mazes + room_mazes

def collect_results(num):
    outfile = open('results_{}.txt'.format(num),'w')
    os.chdir('../training/')
    for m in all_mazes:
        for i in range(len(configs_and_policies)):
            policy,config = configs_and_policies[i]
            log_dir = 'results/maze_tests/{0}/{1}/'.format(m,policy)
            fname = log_dir + 'train_0_rewards.txt'
            print fname
            f = open(fname,'r')
            for line in f:
                pass
            value = line.split('\t')[1]
            outfile.write('{}\t{}\t{}'.format(m,policy,value))
            # exit(0)


if __name__ == '__main__':
    num = int(sys.argv[1])
    collect_results(num)

