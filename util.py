import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import njit
import math
import time

L=float(sys.argv[1])
N=int(sys.argv[2])
r=float(sys.argv[3])
T=float(sys.argv[4])

Nt=1000
Nens=40000

ds=4*r
Ns=int(L/ds)
ds=L/Ns

beta=1/T
pi=math.pi
m=1

dl=L/(2**0.5)/1000

rho=N/L/L
eta=rho*pi*r*r

#cell_list=[]



##@njit
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
def view_config(xarr,yarr):
	fig,ax=plt.subplots()
	for i in range(N):
		x,y=xarr[i],yarr[i]
		circ=plt.Circle((x,y),r,edgecolor='black',facecolor='orange')
		ax.add_patch(circ)
	ax.set_aspect('equal','box')
	ax.set_xlim(0,L)
	ax.set_ylim(0,L)
	ax.set_title(f"eta={eta}")
	plt.show()




##@njit
def make_cell(xarr,yarr):
	cl=[[] for i in range(Ns*Ns)]
	for i in range(N):
		j1= int(xarr[i]/ds)
		i1=int(yarr[i]/ds)
		cl[i1*Ns+j1].append(i)
	return cl 

def nbr2D():
	nbrarr=np.zeros((Ns*Ns,9),dtype=int)
	for i in range(Ns*Ns):
		ix,iy=(i%Ns),(i//Ns)
		if 1<=ix<=Ns-2 and 1<=iy<=Ns-2 :
			nbrarr[i][0]= i+1#East
			nbrarr[i][4]= i-1#West
			nbrarr[i][2]= i+Ns#North
			nbrarr[i][6]= i-Ns#South
			nbrarr[i][1]= i+Ns+1#NE
			nbrarr[i][3]= i+Ns-1#NW
			nbrarr[i][5]= i-Ns-1#SW
			nbrarr[i][7]= i-Ns+1#SE
		elif 1<=ix<=Ns-2 and iy==Ns-1:
			nbrarr[i][0]= i+1#East
			nbrarr[i][4]= i-1#West
			nbrarr[i][6]= i-Ns#South
			nbrarr[i][5]= i-Ns-1#SW
			nbrarr[i][7]= i-Ns+1#SE
			nbrarr[i][2]= i-Ns*(Ns-1)
			nbrarr[i][3]= i-Ns*(Ns-1)-1
			nbrarr[i][1]= i-Ns*(Ns-1)+1
		elif 1<=ix<=Ns-2 and iy==0:
			nbrarr[i][0]= i+1#East
			nbrarr[i][4]= i-1#West
			nbrarr[i][2]= i+Ns#North
			nbrarr[i][1]= i+Ns+1#NE
			nbrarr[i][3]= i+Ns-1#NW
			nbrarr[i][6]= i + Ns*(Ns-1)
			nbrarr[i][5]= i + Ns*(Ns-1)-1
			nbrarr[i][7]= i + Ns*(Ns-1)+1
		elif ix==0 and 1<=iy<=Ns-2:
			nbrarr[i][0]= i+1#East
			nbrarr[i][2]= i+Ns#North
			nbrarr[i][6]= i-Ns#South
			nbrarr[i][1]= i+Ns+1#NE
			nbrarr[i][7]= i-Ns+1#SE
			nbrarr[i][4]= i+Ns-1
			nbrarr[i][3]= i+(Ns-1)+Ns
			nbrarr[i][5]= i-1
		elif ix==Ns-1 and 1<=iy<=Ns-2:
			nbrarr[i][4]= i-1#West
			nbrarr[i][2]= i+Ns#North
			nbrarr[i][6]= i-Ns#South
			nbrarr[i][3]= i+Ns-1#NW
			nbrarr[i][5]= i-Ns-1#SW
			nbrarr[i][0]= i-Ns+1
			nbrarr[i][1]=i+1
			nbrarr[i][7]=i-Ns+1-Ns
		elif ix==0 and iy==0:
			nbrarr[i][0]= i+1#East
			nbrarr[i][4]= i+Ns-1#West
			nbrarr[i][2]= i+Ns#North
			nbrarr[i][6]= i+Ns*(Ns-1)#South
			nbrarr[i][1]= i+Ns+1#NE
			nbrarr[i][3]= i+Ns-1+Ns#NW
			nbrarr[i][5]= i+Ns-1+Ns*(Ns-1)#SW
			nbrarr[i][7]= i+1+Ns*(Ns-1)#SE
		elif ix==Ns-1 and iy==0:
			nbrarr[i][0]= i+1-Ns#East
			nbrarr[i][4]= i-1#West
			nbrarr[i][2]= i+Ns#North
			nbrarr[i][6]= i+Ns*(Ns-1)#South
			nbrarr[i][1]= i+1#NE
			nbrarr[i][3]= i+Ns-1#NW
			nbrarr[i][5]= i-1+Ns*(Ns-1)#SW
			nbrarr[i][7]= Ns*(Ns-1)#SE
		elif ix==0 and iy==Ns-1:
			nbrarr[i][0]= i+1#East
			nbrarr[i][4]= i+Ns-1#West
			nbrarr[i][2]= i-Ns*(Ns-1)#North
			nbrarr[i][6]= i-Ns#South
			nbrarr[i][1]= i-Ns*(Ns-1)+1#NE
			nbrarr[i][3]= Ns-1#NW
			nbrarr[i][5]= i-1#SW
			nbrarr[i][7]= i-Ns+1#SE			
		elif ix==Ns-1 and iy==Ns-1:
			nbrarr[i][0]= i-Ns+1#East
			nbrarr[i][4]= i-1#West
			nbrarr[i][2]= i-Ns*(Ns-1)#North
			nbrarr[i][6]= i-Ns#South
			nbrarr[i][1]= 0#NE
			nbrarr[i][3]= Ns-2#NW
			nbrarr[i][5]= i-Ns-1#SW
			nbrarr[i][7]= i-Ns-Ns+1#SE
	nbrarr[i][8]=i
	return nbrarr


nbrarr=nbr2D()
	
#@njit
def dist(x1,x2,y1,y2):
	dx=abs(x1-x2)
	dy=abs(y1-y2)
	
	dx=min(dx,L-dx)
	dy=min(dy,L-dy)
	return dx*dx+dy*dy


#@njit
def isvalid(xarr,yarr,pos,xnew,ynew,cl):
	x,y=xarr[pos],yarr[pos]
	ipos=(int(x/ds)+Ns*int(y/ds)) #cell number of pos 
	
	
	for i in range(9):
		for jpos in cl[nbrarr[ipos][i]]:
			dx=abs(xnew-xarr[jpos])
			dy=abs(ynew-yarr[jpos])
			
			dx=min(dx,L-dx)
			dy=min(dy,L-dy)
			if jpos!=pos and dx*dx+dy*dy < 4*r*r:
				return False
	return True

#@njit
def update(xarr,yarr,pxarr,pyarr,cl):
	for _ in range(N):
		pos=np.random.randint(N)

		theta=np.random.random()*2*pi
		dx=math.cos(theta)
		dy=math.sin(theta)

		xnew=(xarr[pos]+dx)%L
		ynew=(yarr[pos]+dy)%L

		if isvalid(xarr,yarr,pos,xnew,ynew,cl):
			thetap=np.random.random()*2*pi

			dpx=math.cos(thetap)
			dpy=math.sin(thetap)


			dp2=2*pxarr[pos]*dpx+2*pyarr[pos]*dpy+dpx*dpx+dpy*dpy

			if dp2<0 or np.random.random()<np.exp(-beta*dp2/2/m):
				
				cl[(int(xarr[pos]/ds)+Ns*int(yarr[pos]/ds))].remove(pos)
				
				xarr[pos]=xnew
				yarr[pos]=ynew
				
				cl[(int(xarr[pos]/ds)+Ns*int(yarr[pos]/ds))].append(pos)
				
				
				pxarr[pos]+=dpx
				pyarr[pos]+=dpy


#@njit
def gr(xarr,yarr,garr):
	for i in range(N):
		for j in range(N):
			if i!=j:
				ds=dist(xarr[i],xarr[j],yarr[i],yarr[j])**.5
				garr[int(ds/dl)]+=1
		
#@njit
def getss_data(xarr,yarr,pxarr,pyarr,cl):
	garr=np.zeros(1000)
	
	for i in range(1000):
		update(xarr,yarr,pxarr,pyarr,cl)
	for i in range(1000):
		update(xarr,yarr,pxarr,pyarr,cl)
		gr(xarr,yarr,garr)
	return garr/1000


if __name__=="__main__":
	xarr,yarr,pxarr,pyarr=init_config()
	cl=make_cell(xarr,yarr)
	
	print(f"eta={eta}")
	ti=time.time()
	garr=getss_data(xarr,yarr,pxarr,pyarr,cl)
	tf=time.time()
	print(f"t={tf-ti}s")
	
	
