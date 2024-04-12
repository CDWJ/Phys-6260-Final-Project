import numpy as np


class dynamicObject:
    def __init__(self,vel,dist,time):
        """
        Args:
            self (): self
            vel (float): average direction array for the density's initial velocity
            dist (float): distance of the charge density from the center of mass in the system
            time (float): the current time step from simulation start where t=0 in s (or ms?)
        """
        self.vel = vel
        self.dist = dist
        self.time = time

    def getVelovity(self):
        return self.vel
    


class chargeDensity(dynamicObject):

    def __init__(self, charge):
        """
        Args:
            self (): self
            charge (float): The total charge of the density 
        """
        self.charge = charge
        dynamicObject.__init__(self,vel, dist, time)
        self.__recalculateVel()
        

    def __recalculateVel(self):
        avgVel = self.vel #system velocity which will probably be set to 0.5 times the speed of light
        standDev = 0.01
        normVelMag = np.random.normal(avgVel, standDev, 1) #magnitude of the velocity pulled from a gaussian distribution with a standard deviation of 0.01
        velVec = np.array([normVelMag*(-np.sin(normVelMag*self.time/self.dist)), normVelMag*(np.cos(normVelMag*self.time/self.dist)),0]) #velocity vector in cartestian coordinates
        self.vel = velVec
    
    def getCharge(self):
        return self.charge
    
    def getCurrent(self):
        return self.charge*self.vel
    

    #def force(self, externalForce):


    #def updateVel(self):