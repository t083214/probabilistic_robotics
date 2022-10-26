import sys
sys.path.append('../scripts')

from mcl import *

class ResetMcl(Mcl):
        def __init__(self, envmap, init_pose, max_num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
            distance_dev_rate=0.14, direction_dev=0.05):
            super().__init__(envmap, init_pose, 1, motion_noise_stds, distance_dev_rate, direction_dev)
            self.alphas = {}

        def observation_update(self, observation):
            for p in self.particles:
                p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
            
            alpha = sum([p.weight for p in self.particles])
            obsnum = len(observation)
            if not obsnum in self.alphas: self.alphas[obsnum] = []
            self.alphas[obsnum].append(alpha)

            self.set_ml()
            self.resampling()

def trial():
    time_interval = 0.1
    world = World(30, time_interval, debug = False)

    m = Map()
    for ln in [(2, -3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([-4, -4, 0]).T
    pf = ResetMcl(m, initial_pose, 1000)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    return  pf

pf = trial()

for num in pf.alphas: ###mclalpharesult
    print("landmarks:", num, "particles:", len(pf.particles), "min:", min(pf.alphas[num]), "max:", max(pf.alphas[num]))