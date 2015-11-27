# ---------------------------------------------------------------------------------------------------- # 
# ------------------- TWO DIMENSIONAL FINITE DIFFERENCE METHOD WITH FIXED SOURCES -------------------- #

# Abhishek Sharma, BioMIP, Ruhr University, Bochum
# Last updated : August 13th, 2015

__author__ = "abhishek"

"""
    We solve two dimensional reaction-diffusion equation with fixed point sources which could be dynamically
    turned "ON" or "OFF". In this version, we used only single chemical field for simplicity. The simulation
    domain is shown in the figure, 
    
                +----------------------------------+
                |                                  |
                |  +   $   +   $   +   $   +   $   |   
                |			   	   |
                |  $   +   $   +   $   +   $   +   |
                |				   |
                |  +   $   +   $   +   +   +   $   |
                |				   |
                |  $   +   $   +   $   +   $   +   |
                |				   |
                |  +   $   +   $   +   $   +   $   |       $ : Sensor
                |				   |       + : Actuator
                |  $   +   $   +   $   +   $   +   |
                |				   |
                |  +   $   +   $   +   $   +   $   |
                |				   |
                +----------------------------------+
     
                        --- Docking Chip ---


The governing equation is given by,

                      dC/dt = Diff*d2C/dx2 + K delta(x-xP),
                      
 together with Dirchilet Boundary Conditions at the ends.                       
    
"""
import time
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import linalg
from scipy.linalg import solve
import gobject
import gtk
import matplotlib
matplotlib.use('GTkAgg')
from pylab import *

nX, nY = 100, 100			# Number of grid points along X, Y directions
L = 100					# Edge length of simulation domain
dx, dy = L/(nX-1), L/(nY-1)		# Spatial step width
tT = 100.				# Total time of simulation
dt = 0.1				# Simulation time step
Dc = 2.5				# Diffusion constant for chemical field
Kc = 0.005				# Kinetic rate constant for dissipation
u = np.zeros((nX, nY))			# Initial chemical field distribution
u[1,1] = 2.
un = np.zeros((nX, nY))			# New chemical field distribution
noA = 3 				# Number of actors in one row
noS = 3					# Number of sensors in one row
noR = 3					# Number of rows in simulation domain
dC = 0.5				# Maximum noise in sensor readout
swT = 30				# NUmber of steps after which electrodes switches
UW, UE, US, UN = 0., 0., 0., 0. 	# Defining Boundary Conditions 

# Defining boundary conditions

boundC = np.zeros((nX-2, nY-2))		
boundC[0,:] = UW/dx**2 
boundC[nX-3,:] = UE/dy**2
boundC[:,0] = US/dx**2
boundC[:,nY-3] = UN/dy**2

boundC[0,0] = UW/dx**2 + US/dy**2
boundC[nX-3,0] = UE/dx**2 + US/dy**2
boundC[0,nY-3] = UW/dx**2 + UN/dy**2
boundC[nX-3, nY-3] =  UE/dx**2 + UN/dy**2

boundC = boundC*dt*Dc

# Creating Sparse matrix for two dimensions using Kronecker Product between two matrices

mat1 = np.zeros(nX-2)
mat2 = np.zeros(nX-3)
mat3 = np.zeros(nX-3)
mat4 = np.zeros(nX-2)
mat5 = np.zeros((nX-2)*(nY-2))

mat1[:], mat2[:], mat3[:], mat4[:], mat5[:] = -2., 1., 1., 1., 1.


Ax = scipy.sparse.diags(diagonals=[mat1, mat2, mat3],
                        offsets=[0, -1, 1], shape=(nX-2, nX-2),
                        format='csr') 
                                       
Ay = scipy.sparse.diags(diagonals=[mat1, mat2, mat3],
                        offsets=[0, -1, 1], shape=(nY-2, nY-2),
                        format='csr')

Adx = scipy.sparse.diags(diagonals=[mat4],  
                         offsets=[0], shape=(nX-2, nX-2),
                         format='csr')   
                        
Ady = scipy.sparse.diags(diagonals=[mat4], 
                         offsets=[0], shape=(nY-2, nY-2),      
                         format='csr')  
                                              
A = scipy.sparse.kron((Ay/dy**2), Adx) + scipy.sparse.kron(Ady, (Ax/dx**2))

dMat = scipy.sparse.diags(diagonals=[mat5],
                          offsets=[0], shape=((nX-2)*(nY-2), (nX-2)*(nY-2)),
                          format='csr') - Dc*dt*A - Kc*dt*A
                          
                                                                            
# print dMat.todense()

# Matrix inversion, will take some time

# idMat = np.linalg.inv(dMat.todense())

# Adding electrodes as fixed point sources

posX= np.delete(np.linspace(0, L, (noS+noA+2), endpoint=False),0)
posY = np.delete(np.linspace(0, L, (noR+2), endpoint=False),0)
pX,pY = np.meshgrid(posX, posY)
pX, pY = pX.flatten(), pY.flatten()
noS, noA = int(len(pX)/2), int(len(pX)/2)

sensors, actors = np.zeros([noS, 2]), np.zeros([noA, 2])
sensorsPos, actorsPos = np.zeros([noS, 1]), np.zeros([noA, 1])

cntS, cntA = 0., 0.

for i in range(len(pX)):    
    if i%2 == 0:
          sensors[cntS, 0], sensors[cntS, 1] = int(pX[i]), int(pY[i])          
          cntS+=1

    else:
          actors[cntA, 0], actors[cntA, 1] = int(pX[i]), int(pY[i])                               
          cntA+=1
                                       
# Locating the position of electrodes sparse array

arrPos = np.zeros((nX-2)*(nY-2))

for i in range(noS):
     sensorsPos[i] = (nX-2)*(sensors[i,1] - 1) + sensors[i,0] - 1
     arrPos[int(sensorsPos[i])] = 1.

# for j in range(noA):
#     actorsPos[j] = (nX-2)*(actors[j,1] - 1) + actors[j,0] - 1
#     arrPos[int(actorsPos[j])] = 0.1

elS = np.random.randint(noA, size=1)
aPos = np.random.randint(noA, size=elS)
actorPos = np.zeros([noA, 1])
                           
for j in range(len(aPos)):     
     actorsPos[j] = (nX-2)*(actors[int(aPos[j]),1]) + actors[int(aPos[j]),0]
     arrPos[int(actorsPos[j])] = 1.
                         
# Running simulation

U = (un[1:-1, 1:-1] + boundC).flatten()
cnt, m = 0, 1
fig = plt.figure(1)
img = subplot(111)
im = img.imshow( u, cmap=cm.gnuplot2, interpolation='bilinear', origin='lower')
manager = get_current_fig_manager()
fig.colorbar(im) # Show the colorbar

def updatefig(*args):
     global u, un, m, cnt, aPos
     cnt+=1
     print "Step. no : ", cnt
     
     un = u
     arrPos = np.zeros((nX-2)*(nY-2))  

     for i in range(noS):
          sensorsPos[i] = (nX-2)*(sensors[i,1]) + sensors[i,0] 
          arrPos[int(sensorsPos[i])] = 0.

     for j in range(len(aPos)):     
          actorsPos[j] = (nX-2)*(actors[int(aPos[j]),1]) + actors[int(aPos[j]),0]
          arrPos[int(actorsPos[j])] = 1.
     
     if(cnt%swT==0):          
         elS = np.random.randint(noA, size=1)
         aPos = np.random.randint(noA, size=elS)
         actorPos = np.zeros([noA, 1])

         #for j in range(len(aPos)):     
         #    actorsPos[j] = (nX-2)*(actors[int(aPos[j]),1]) + actors[int(aPos[j]),0] 
         #    arrPos[int(actorsPos[j])] = 1.
     
     U = (un[1:-1, 1:-1] + boundC).flatten()
     # U = np.reshape(np.dot(idMat, U) + arrPos, (nX-2, nY-2))
     U = np.reshape(scipy.sparse.linalg.spsolve(dMat, U) + arrPos, (nX-2, nY-2))
     u[1:-1, 1:-1] = U
     u[0, :] = UW
     u[nX-1, :] = UE
     u[:, 1] = US
     u[:, nY-1] = UN
     
     im.set_array(u)
     manager.canvas.draw()
     m+=1
     # print "Computing and rendering u for m =", m
     if m >= 1000:
          return False
     return True

gobject.idle_add(updatefig)

show()


# --------------------------------------------------- end-of-file ------------------------------------- # 

