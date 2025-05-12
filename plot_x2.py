import numpy as np
import matplotlib.pyplot as plt


rarr=[.12,.13,.14,.15,.16,.17,.18,.19,.2,0.21,0.215]
x=np.array(range(1,761),dtype=int)
for i in range(1,760):
    x[i]=int(x[i-1]*1.01+1)
#plt.xlim(1,12)


for r in rarr:
    y=np.loadtxt(f"x2L10.0N500r{r}T1.0.dat",max_rows=760)/500/10/10/2*3
    eta=np.pi*r*r/10/10*500
    plt.plot(x,y,label=f"{eta:.2f}")
plt.xscale('log')
plt.legend()
plt.show()

