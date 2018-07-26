import numpy as np

# Definition of constants
G = 6.67e-20  # Gravitational Constant
AU = 149597870  # km
SolarMass = 1.989e30  # kg
EarthMass = 5.972e24  # kg

class CBody: # Definition of a celestrial body
    state = None  # 3*3 matrix for state of motion (km)
    m = 1  # mass in kg
    name = None
    def __init__(self, state, mass=0, radius=0, speed=0, theta=0, phi=np.pi / 2, name = ''):
        self.name=name
        if (state == None):
            self.state = np.zeros((3, 3))
            self.state[0] = np.array([
                np.sin(phi) * np.cos(theta) * radius,
                np.sin(phi) * np.sin(theta) * radius,
                np.cos(phi)
            ])
            self.state[1] = np.array([
                -speed * np.sin(phi) * np.sin(theta),
                speed * np.sin(phi) * np.cos(theta),
                0
            ])
        else:
            self.state = state.astype(float)
        self.m = mass
        return

    def step0(self, dt):  # recursive method for propagation with 2nd level taylor expansion
        original = np.copy(self.state)
        self.state[0] += dt * self.state[1] + self.state[2] * dt * dt / 2
        # self.state[1] += dt * self.state[2] / 2
        self.state[2] = 0  # Clear acceleration vector
        return original

    # dt = time gap, original = motion states before state0
    def step1(self, dt, original):
        self.state[1] += (original[2] + self.state[2]) / 2 * dt
        self.state[2]=0
        return self.state

    def pullEachOther(self, cbody):  # compute mutual attraction
        p01 = (self.state[0] - cbody.state[0])  # Vector pointing from another cbody to this one
        F = p01 * (G * self.m * cbody.m / (np.linalg.norm(p01) ** 3))  # Newton's Law of gravitation
        cbody.state[2] += F / cbody.m
        self.state[2] -= F / self.m


"""* Test Cbody stepping method
* Earth mass = 5.972e24kg DtoSun=149,597,870 km Speed = 29.783 km/s
* Jupiter mass = 1.898e27kg DtoSun=778,547,200 km Speed = 13.06 km/s
* Venus mass = 4.867e24kg DtoSun=108,200,000 km Speed = 35.03 km/s
* Sun mass = 1.989e30kg DtoSun=0 km Speed = 0 km/s"""


earth = CBody(None, mass=5.972e24, radius=149597870, theta=3, speed=29.783,
              name='earth')
jupiter = CBody(None, mass=1.898e27, radius=778547200, theta=40, speed=13.06,
                name='jupiter')
venus = CBody(None, mass=4.867e24, radius=108200000, theta=25, speed=35.03,
              name='venus')
sun = CBody(None, mass=1.989e30, radius=0, speed=0, theta=180,
            name='sun')
mercury = CBody(None, 3.3022e23, 0.387098*AU, 47.89, 355,
                name='mercury')
bodies = [earth, jupiter, sun, venus, mercury]

def computeRelations(bodies):
    i = 1
    for cbody in bodies:
        for j in range(i, len(bodies)):
            cbody.pullEachOther(bodies[j])
        i += 1

#K-2 method for integration, third order taylor expansion
def stepAll(bodies, dt):
    computeRelations(bodies)
    stateCapture = np.empty([len(bodies), 3, 3])
    i = 0
    #first step
    for body in bodies:
        stateCapture[i] = body.step0(dt)
        i += 1
    computeRelations(bodies)
    i = 0
    #second step
    for body in bodies:
        stateCapture[i] = body.step1(dt, stateCapture[i])
        i += 1
    return stateCapture

totalSteps = 25000
gapPerStep = 20000
def simulate(totalSteps, stepInterval):
    stateRecord = np.empty([1000, len(bodies), 3, 3])
    g = totalSteps / 1000
    for i in range(totalSteps):
        if (i % g == 0):
            stateRecord[int(i / g)] = stepAll(bodies, stepInterval)
        else:
            stepAll(bodies, stepInterval)
    return stateRecord

stateRecord = simulate(totalSteps,gapPerStep)
visualization = np.transpose(stateRecord, (1, 2, 3, 0))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure(figsize=(6, 6))
ax = fig.gca(projection='3d')
#ax = plt.gca()

ax.grid()

for i in range(len(bodies)):
    ax.plot(visualization[i][0][0], visualization[i][0][1], #label=bodies[i].name)  #
    visualization[i][0][2], label=bodies[i].name)
ax.legend()


dotplots = []

def init():
    ax.set_xlim(-1e9, 1e9)
    ax.set_ylim(-1e9, 1e9)
    for i in range(len(bodies)):
       plot, = ax.plot([],[], 'bo',lw=2)
       plot.set_data(visualization[i][0][0][0], visualization[i][0][1][0],visualization[i][0][2][0])
       dotplots.append(plot)

def update(i):
    for j in range(len(bodies)):
       dotplots[j].set_data(visualization[j][0][0][i], visualization[j][0][1][i],visualization[j][0][2][i])

ani = animation.FuncAnimation(fig, update, range(len(visualization[0][0][0])), init_func=init, interval=1, blit=False)

plt.show()
plt.interactive(True)
