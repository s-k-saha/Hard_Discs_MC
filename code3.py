import time
import numpy as np
import sys
from numba import njit
import math
import matplotlib.pyplot as plt

L=float(sys.argv[1])
N=int(sys.argv[2])
r=float(sys.argv[3])
T=float(sys.argv[4])

Nt=760
Nens=10

beta=1/T
pi=math.pi
m=1

dl=L/(2**0.5)/1000

rho=N/L/L
eta=rho*pi*r*r


#cell list specific parameters
Ncell= int(L/r/2) #number of cells along a linear dimension
Ns=Ncell
sigma= L/Ncell #cell edge length
Ncells=Ncell*Ncell


@njit
def init_config(): #cubical close packing (max possible eta ~ 0.785398163397448)
	Nmax=int(L/r/2) #max discs per dimension
	x=r
	y=r

	xarr=[]
	yarr=[]
	pxarr=np.random.random(size=N)/np.sqrt(2)
	pyarr=np.random.random(size=N)/np.sqrt(2)
	count=0
	for i in range(Nmax):
		x=r
		for j in range(Nmax):
			if count==N:
				return np.array(xarr),np.array(yarr),pxarr,pyarr
			xarr.append(x)
			yarr.append(y)
			x+=2*r
			count+=1
		y+=2*r


@njit
def nbr2D():
    nbrarr = np.zeros((Ncells, 9), dtype=np.int32)
    for i in range(Ncells):
        ix = i % Ncell
        iy = i // Ncell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                jx = (ix + dx) % Ncell
                jy = (iy + dy) % Ncell
                j = jy * Ncell + jx
                idx = (dy + 1) * 3 + (dx + 1)  # maps (dx,dy) to [0..8]
                nbrarr[i, idx] = j
    return nbrarr


nbrarr=nbr2D()
	



@njit
def build_cell_list(xarr,yarr):
	cell_list=np.ones((Ncells,4),dtype=np.int32)*-1
	#end=np.zeros(Ncells,dtype=np.int32) #index of the first empty position in the i-th cell  
	which_cell=np.zeros(N,dtype=np.int32) #index of the cell the i-th particle belongs to
	
	for i in range(N):
		x,y=int(xarr[i]/sigma),int(yarr[i]/sigma)
		pos=(y*Ncell+x)
		which_cell[i]=pos
		for _ in range(4):
		    if cell_list[pos][_]==-1:
		        cell_list[pos][_]=i
		        break
		#end[pos]+=1 
	return cell_list,which_cell
		
	


@njit
def isvalid(xarr,yarr,pos,xnew,ynew,cell_list,which_cell):
	cellno=int(ynew/sigma)*Ncell+int(xnew/sigma)
	
	for i in range(9):
		for j in range(4):
			k=cell_list[nbrarr[cellno][i]][j]
			dx=abs(xnew-xarr[k])
			dy=abs(ynew-yarr[k])
			
			dx=min(dx,L-dx)
			dy=min(dy,L-dy)
			
			if k!=pos and dx*dx+dy*dy < 4*r*r:
				return False
	return True


@njit
def remove_particle(pos,cell_list,which_cell):
    for j in range(4):
        if cell_list[which_cell[pos]][j]==pos:
            cell_list[which_cell[pos]][j]=-1
            return

@njit
def add_particle(pos,cell_list,which_cell):
    for j in range(4):
        if cell_list[which_cell[pos]][j]==-1:
            cell_list[which_cell[pos]][j]=pos
            return


@njit
def update(xarr,yarr,pxarr,pyarr,cell_list,which_cell):
	for _ in range(N):
		pos=np.random.randint(N)

		theta=np.random.random()*2*pi
		dx=math.cos(theta)*r/2
		dy=math.sin(theta)*r/2

		xnew=(xarr[pos]+dx)%L
		ynew=(yarr[pos]+dy)%L

		if isvalid(xarr,yarr,pos,xnew,ynew,cell_list,which_cell):
			thetap=np.random.random()*2*pi

			dpx=math.cos(thetap)
			dpy=math.sin(thetap)


			dp2=2*pxarr[pos]*dpx+2*pyarr[pos]*dpy+dpx*dpx+dpy*dpy

			if dp2<0 or np.random.random()<np.exp(-beta*dp2/2/m):
				xarr[pos]=xnew
				yarr[pos]=ynew
				pxarr[pos]+=dpx
				pyarr[pos]+=dpy
				
				
				remove_particle(pos,cell_list,which_cell)
				x,y=int(xnew/sigma),int(ynew/sigma)
				newpos=(y*Ncell+x)
				
				which_cell[pos]=newpos
				add_particle(pos,cell_list,which_cell)

@njit
def Eng(pxarr,pyarr):
	return np.sum(pxarr*pxarr+pyarr*pyarr)/2/m

@njit
def x2(xarr,yarr):
	return np.sum(xarr*xarr+yarr*yarr)


@njit
def dist(x1,x2,y1,y2):
	dx=abs(x1-x2)
	dy=abs(y1-y2)
	
	dx=min(dx,L-dx)
	dy=min(dy,L-dy)
	return dx*dx+dy*dy
	
	
@njit
def gr(xarr,yarr,garr):
	for i in range(N):
		for j in range(N):
			if i!=j:
				ds=dist(xarr[i],xarr[j],yarr[i],yarr[j])**.5
				garr[int(ds/dl)]+=1


@njit
def get_data(x2arr,t):
	for _ in range(Nens):
		xarr,yarr,pxarr,pyarr=init_config()
		cell_list,which_cell=build_cell_list(xarr,yarr)
		j=1
		for i in range(Nt):
			while j<=t[i]:
				update(xarr,yarr,pxarr,pyarr,cell_list,which_cell)
				j+=1
			x2arr[i]+=x2(xarr,yarr)

@njit
def getss_data():
	garr=np.zeros(1000)
	xarr,yarr,pxarr,pyarr=init_config()
	cell_list,which_cell=build_cell_list(xarr,yarr)
	
	for i in range(1000000):
		update(xarr,yarr,pxarr,pyarr,cell_list,which_cell)
	for i in range(4000000):
		update(xarr,yarr,pxarr,pyarr,cell_list,which_cell)
		gr(xarr,yarr,garr)
	return garr/4000000

@njit
def time_state():
	xarr,yarr,pxarr,pyarr=init_config()
	for _ in range(Nt):
		update(xarr,yarr,pxarr,pyarr)
	return xarr,yarr



def tarr(n):
    t=[1,]
    for i in range(n):
        t.append(int(t[-1]*1.01+1))
    return t
    

def view_config(xarr,yarr):
	fig,ax=plt.subplots()
	for i in range(N):
		x,y=xarr[i],yarr[i]
		circ=plt.Circle((x,y),r,edgecolor='black',facecolor='orange')
		ax.add_patch(circ)
	ax.set_aspect('equal','box')
	ax.set_xlim(0,L)
	ax.set_ylim(0,L)
	plt.scatter(xarr,yarr,color='black',s=2)
	plt.xticks(np.arange(0, L, 2*r),['']*len(np.arange(0, L, 2*r)))
	plt.yticks(np.arange(0, L, 2*r),['']*len(np.arange(0, L, 2*r)))
	plt.grid()
	ax.set_title(f"eta={eta}")
	plt.show()

if __name__=="__main__":
	x2arr=np.zeros(Nt)
	t=tarr(Nt)
	k=1
	while True:
	    get_data(x2arr,t)
	    np.savetxt(f"x2L{L}N{N}r{r}T{T}.dat",x2arr/Nens/k)
	    with open(f"x2L{L}N{N}r{r}T{T}.dat","a") as fl:
	        fl.write(f"eta={eta};{k}x{Nens} ens done")
	    k+=1
	
	
	#Engarr,x2arr=get_data()
	#data=np.concatenate((Engarr.reshape((-1,1)),x2arr.reshape((-1,1))),axis=1)
	#np.savetxt(f"EngL{L}N{N}r{r}T{T}.dat",data)

