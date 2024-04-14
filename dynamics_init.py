import numpy as np

def integ(N,bounds, func):
    """
    uplim = bounds[1]
    lowlim = bounds[0]
    dn = (uplim-lowlim)/N #what we're iterating
    final = 0
    for i in range(0,N):
        final += (dn/2)*(func[0]+2*func[])
    return final
    """


class dynamicObject:
    def __init__(self,vel,dist,time,mass):
        """
        Args:
            self (): self
            vel (float): average direction array for the density's initial velocity
            dist (float): distance of the charge density from the center of mass in the system
            time (float): the current time step from simulation start where t=0 in s (or ms?)
            mass (float): object mass in kg
        """
        self.vel = vel
        self.dist = dist
        self.time = time
        self.mass = mass

    def getVelovity(self):
        return self.vel
    
    def updateVel(self, newVel):
        self.vel = newVel

    def getAcceleration(self, prevVels, steps):
        """
        Args:
            prevVels (ndarray): Velocities at the previous timesteps we want to find the acceleration over
            steps (int): The precision of the integration technique
        """
        timestep = 3e-5 #around 30 microseconds
        endtime = self.time
        starttime = endtime - (len(prevVels)*timestep)
        velocities = prevVels.append(self.getVelovity())
        accel = integ(steps, (starttime, endtime) , velocities)
        return accel


    #def force(self, externalForce):

    def getPos(self, dt):
        """
        Args:
            self (): self
        """
        self.vel

    #def updatePos(self):
    


class chargeDensity(dynamicObject):

    def __init__(self, charge, vel, dist, time, mass):
        """
        Args:
            self (): self
            charge (float): The total charge of the density
        """
        self.charge = charge
        self.mass = mass
        self.__calculateInitVel()
        dynamicObject.__init__(self, vel, dist, time, mass) 

    def __calculateInitVel(self):
        """
        This private method calculates the initial velovity of our charge/current density as a normal distribution around the provided average 
        (for us this average will likely be 0.5c)
        """
        avgVel = self.vel #system velocity which will probably be set to 0.5 times the speed of light
        standDev = 0.01
        normVelMag = np.random.normal(avgVel, standDev, 1)[0] #magnitude of the velocity pulled from a gaussian distribution with a standard deviation of 0.01
        velVec = np.array([normVelMag*(-np.sin(normVelMag*self.time/self.dist)), normVelMag*(np.cos(normVelMag*self.time/self.dist)),0]) #velocity vector in cartestian coordinates
        self.vel = velVec

    def getCharge(self):
        return self.charge
    
    def getMass(self):
        numb =  #number density of our electrons

    #def getCurrent(self):
    #    return self.charge*self.vel
    
