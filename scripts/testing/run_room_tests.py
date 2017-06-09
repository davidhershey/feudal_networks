import os


configs_and_policies = [('lstm','lstm_maze_config'),('feudal','feudal_maze_config')]
#mazes = ['VisionMaze3-v0','VisionMaze5-v0','VisionMaze9-v0','VisionMaze25-v0']
room_mazes = ['RoomMaze3-v0','RoomMaze5-v0','RoomMaze7-v0','RoomMaze9-v0']

all_mazes = room_mazes

def run_tests():
    port = 12000
    num_steps = 100000
    os.chdir('../training/')
    for m in all_mazes:
        for i in range(len(configs_and_policies)):
            policy,config = configs_and_policies[i]
            log_dir = 'results/maze_tests/{0}/{1}'.format(m,policy)
            command = 'nohup python train.py -w 1 -m child -p {0} -c {1} -e {2} -l {3}  --ports {4} --steps {5}&'.format(policy,config,m,log_dir,port,num_steps)
            os.system(command)
            print 'Running ', m, ' with policy ', policy, ' on port ', port
            port+=100
            # exit(0)


if __name__ == '__main__':
    run_tests()
