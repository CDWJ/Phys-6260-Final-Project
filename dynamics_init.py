import numpy as np


class dynamicObject:
    def __init__(self,vel,pos,dist,time,accel,mass, timestep):
        """
        Args:
            self (): self
            vel (ndarray): average direction array for the objects's initial velocity
            dist (float): distance of the charge density from the center of mass in the system
            time (float): the current time step from simulation start where t=0 in s (or ms?)
            mass (float): object mass in kg
            timestep(float): constant timestep in the system
        """
        self.vel = vel
        self.pos = pos
        self.dist = dist
        self.time = time
        self.accel = accel
        self.mass = mass
        self.timestep = timestep

    def getVelovity(self):
        return self.vel
    
    def updateAccel(self, force):
        """
        updates the acceleration of the object.
        Args:
            force(ndarray): total force on the system
        """
        newAccel = force/self.mass
        self.updateVel(newAccel)
        self.accel = newAccel

    def updateVel(self, newAccel):
        """
        updates the velocity based on the new acceleration
        Args:
            newAccel(ndArray): acceleration vector
        """
        newVel = (self.accel+newAccel)*self.timestep/2
        self.updatePos(newVel)
        self.vel = newVel

    
    def updatePos(self, newVel):
        newPos = (self.vel+newVel)*self.timestep/2
        self.pos = newPos
    

    def getPos(self, dt):
        """
        Args:
            self (): self
        """
        self.vel


class chargeDensity(dynamicObject):

    def __init__(self, charge,Lforce, vel, pos, dist,time, accel, mass, timestep):
        """
        Args:
            self (): self
            charge (float): The total charge of the density
            Lforce (ndarray): External Lorentz force on the charge density (could be initially 0)
        """
        self.charge = charge
        self.mass = mass
        self.Lforce = Lforce
        self.__calculateInitVel()
        dynamicObject.__init__(self, vel, pos, dist,time, accel, mass, timestep) 

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

    def updateLforce(self, extforce):
        """
        args:
            extforce (ndarry): our new lorentz force vector
        """
        updated = self.Lforce - extforce
        self.updateForce(updated)

    def getcentrifugalForce(self):
        return self.mass*(self.vel**2/self.dist)

    def updateForce(self, newLforce):
        updated = newLforce + self.getcentrifugalForce()
        self.force = updated
        self.updateAccel(updated)

    #def getCurrent(self):
    #    return self.charge*self.vel

    #def getMagField(self):
    #   return magField
    
