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

Nt=1000
Nens=40000

beta=1/T
pi=math.pi
m=1

dl=L/(2**0.5)/1000

rho=N/L/L
eta=rho*pi*r*r



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
def isvalid(xarr,yarr,pos,xnew,ynew):
	for i in range(N):
		dx=abs(xnew-xarr[i])
		dy=abs(ynew-yarr[i])
		
		dx=min(dx,L-dx)
		dy=min(dy,L-dy)
		
		if i!=pos and dx*dx+dy*dy < 4*r*r:
			return False
	return True

@njit
def update(xarr,yarr,pxarr,pyarr):
	for _ in range(N):
		pos=np.random.randint(N)

		theta=np.random.random()*2*pi
		dx=math.cos(theta)*r
		dy=math.sin(theta)*r

		xnew=(xarr[pos]+dx)%L
		ynew=(yarr[pos]+dy)%L

		if isvalid(xarr,yarr,pos,xnew,ynew):
			thetap=np.random.random()*2*pi

			dpx=math.cos(thetap)
			dpy=math.sin(thetap)


			dp2=2*pxarr[pos]*dpx+2*pyarr[pos]*dpy+dpx*dpx+dpy*dpy

			if dp2<0 or np.random.random()<np.exp(-beta*dp2/2/m):
				xarr[pos]=xnew
				yarr[pos]=ynew
				pxarr[pos]+=dpx
				pyarr[pos]+=dpy

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
def get_data():
	Engarr=np.zeros(Nt)
	x2arr=np.zeros(Nt)


	for _ in range(Nens):
		xarr,yarr,pxarr,pyarr=init_config()
		for i in range(Nt):
			update(xarr,yarr,pxarr,pyarr)


			Engarr[i]+=Eng(pxarr,pyarr)
			x2arr[i]+=x2(xarr,yarr)

	return Engarr/Nens,x2arr/Nens

@njit
def getss_data():
	garr=np.zeros(1000)
	xarr,yarr,pxarr,pyarr=init_config()
		
	for i in range(100000):
		update(xarr,yarr,pxarr,pyarr)
	for i in range(1000000):
		update(xarr,yarr,pxarr,pyarr)
		gr(xarr,yarr,garr)
	return garr/1000000

@njit
def time_state():
	xarr,yarr,pxarr,pyarr=init_config()
	for _ in range(Nt):
		update(xarr,yarr,pxarr,pyarr)
	return xarr,yarr
@njit
def run_t():
	xarr,yarr,pxarr,pyarr=init_config()
	for i in range(100000):
		update(xarr,yarr,pxarr,pyarr)
	return xarr,yarr

@njit
def check_config(xarr,yarr):
	count_False=0
	for i in range(N):
		for j in range(N):
			if i!=j and dist(xarr[i],xarr[j],yarr[i],yarr[j])< 4*r*r:
				count_False+=1
	return count_False		
	
	

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
	print(f"eta={eta}")
	
	ti=time.time()
	x,y=run_t()
	tf=time.time()
	print(f"{tf-ti} s")
	view_config(x,y)
	print(check_config(x,y))


	#Engarr,x2arr=get_data()
	#data=np.concatenate((Engarr.reshape((-1,1)),x2arr.reshape((-1,1))),axis=1)
	#np.savetxt(f"EngL{L}N{N}r{r}T{T}.dat",data)

