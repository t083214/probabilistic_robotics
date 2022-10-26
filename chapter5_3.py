import sys
sys.path.append('../scripts')
from robot import *
from scipy.stats import multivariate_normal

class EstimationAgent(Agent):
	def __init__(self, time_interval, nu, omega, estimator):
		super().__init__(nu, omega)
		self.estimator = estimator
		self.time_interval = time_interval
		self.prev_nu = 0.0
		self.prev_omega = 0.0

	def draw(self, ax, elems):
		#print('estimation agent draw call')
		#elems.append(ax.text(0, 0, "hoge", fontsize=10))
		self.estimator.draw(ax, elems)
		# This function is called at world.draw function 

	def decision(self, observation=None):
		self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
		self.prev_nu, self.prev_omega = self.nu, self.omega
		return self.nu, self.omega

class Particle:
	def __init__(self, init_pose):
		self.pose = init_pose

	def motion_update(self, nu, omega, time, noise_rate_pdf):
		ns = noise_rate_pdf.rvs()
		noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
		noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
		self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)
		#self.pose = IdealRobot.state_transition(nu, omega, time, self.pose)

class Mcl:
	def __init__(self, init_pose, num, motion_noise_stds):
		self.particles = [Particle(init_pose) for i in range(num)]

		v = motion_noise_stds
		c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
		self.motion_noise_rate_pdf = multivariate_normal(cov=c)

	def draw(self, ax, elems):
		xs = [p.pose[0] for p in self.particles]
		ys = [p.pose[1] for p in self.particles]
		vxs = [math.cos(p.pose[2]) for p in self.particles]
		vys = [math.sin(p.pose[2]) for p in self.particles]
		elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha = 0.5))

	def motion_update(self, nu, omega, time):
		for p in self.particles: p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
		#print(self.motion_noise_rate_pdf.cov)


def trial(motion_noise_stds):
	time_interval = 0.1
	world = World(30 ,time_interval)
	initial_pose = np.array([0, 0, 0]).T
	estimator = Mcl(initial_pose, 100, motion_noise_stds)
	circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
	r = Robot(initial_pose, sensor=None, agent=circling, color="red")
	world.append(r)
	
	world.draw()


#
#m = Map()

#for ln in [(-4, 2), (2, -3), (3, 3)]: m.append_landmark(Landmark(*ln))
#world.append(m)


#estimator.motion_update(0.2, 10.0/180*math.pi, 0.1)
#for p in estimator.particles:
#	print(p.pose)
#circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator)


trial({"nn":0.1, "no":0.2, "on":0.3, "oo":0.4})
#trial({"nn":0.01, "no":0.02, "on":0.03, "oo":0.04})