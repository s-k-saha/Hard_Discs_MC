import numpy as np
import matplotlib.pyplot as plt


rarr=[.117,.118,.119,.12,.121,.122,.123,.124,.125]
xarr=np.array(range(1000))

plt.xlim(1,12)
for r in rarr:
    data=np.loadtxt(f"gr2L10.0N1500r{r}T1.0.dat",max_rows=1000)
    x=xarr/1000*10/2**.5/2/r
    y=data/x/2/np.pi
    eta=np.pi*r*r/10/10*1500
    plt.plot(x,y,label=f"{eta:.2f}")
plt.legend()
plt.show()

