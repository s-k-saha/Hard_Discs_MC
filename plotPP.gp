set terminal 'wxt'

p 'data.dat' u 1:($2*(1+2*$1*$3)/100*4*0.16*0.16) w lp

pause -1
