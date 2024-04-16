import numpy as np


class dynamicObject:
    def __init__(self,vel,pos,time,accel,mass, timestep):
        """
        All arrays are in form (x, y, z) or their corresponding unit vectors
        Args:
            self (): self
            vel (ndarray): average direction array for the objects's initial velocity
            pos (ndarray): position vector for the object
            time (float): the current time step from simulation start where t=0 in s (or ms?)
            accel (ndarray): the acceleration vector for the system
            mass (float): object mass in kg
            timestep(float): constant timestep in the system
        """
        self.vel = vel
        self.pos = pos
        self.time = time
        self.accel = accel
        self.mass = mass
        self.timestep = timestep

    def getDist(self):
        """returns the distance (float): distance of the charge density from the center of mass in the system. (Assuming the center of mass
        is at position (0,0,0))
        """
        cent = np.array([0,0,0])
        dist = np.sqrt(self.pos[0]**2 + self.pos[1]**2 + self.pos[-1]**2)
        return dist
    
    def updateAccel(self, force):
        """
        updates the acceleration of the object based on the force on the object
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
        """
        updates the position based on the new velocities
            Args:
                newAccel(ndArray): acceleration vector
        """
        newPos = (self.vel + newVel)*self.timestep/2
        self.pos = newPos
    
    def updateTime(self):
        newTime = self.time + self.timestep
        self.time = newTime

    def getPos(self, dt):
        return self.pos
    
    def getVelovity(self):
        return self.vel


class chargeDensity(dynamicObject):

    def __init__(self, charge, Lforce, vel, pos, time, accel, mass, timestep):
        """
        Args:
            self (): self
            charge (float): The total charge of the density
            Lforce (ndarray): External Lorentz force on the charge density (could be initially 0)
        """
        self.charge = charge
        self.mass = mass
        self.Lforce = Lforce

        #the attributes that need to be initialized for __calculateInitVel() to work
        self.vel = vel
        self.pos = pos
        self.time = time

        self.__calculateInitVel()
        dynamicObject.__init__(self, vel, pos, time, accel, mass, timestep) 

    def __calculateInitVel(self):
        """
        This private method calculates the initial velovity of our charge/current density as a normal distribution around the provided average 
        (for us this average will likely be 0.5c)
        """
        avgVel = self.vel #system velocity which will probably be set to 0.5 times the speed of light
        standDev = 0.01
        normVelMag = np.random.normal(avgVel, standDev, 1)[0] #magnitude of the velocity pulled from a gaussian distribution with a standard deviation of 0.01
        dist = self.getDist()
        velVec = np.array([normVelMag*(-np.sin(normVelMag*self.time/dist)), normVelMag*(np.cos(normVelMag*self.time/dist)),0]) #velocity vector in cartestian coordinates
        self.vel = velVec

    def getCharge(self):
        return self.charge
    
    #def getMass(self):
    #    numb = #number density of our electrons

    def updateLforce(self, extforce):
        """
        args:
            extforce (ndarry): our new lorentz force vector
        """
        updated = self.Lforce - extforce
        self.updateForce(updated)

    def getcentrifugalForce(self):
        return self.mass*(self.vel**2/self.getDist())

    def updateForce(self, newLforce):
        updated = newLforce + self.getcentrifugalForce()
        self.force = updated
        self.updateAccel(updated)

    #def getCurrent(self):
    #    return self.charge*self.vel

    #def getMagField(self):
    #   return magField
    
