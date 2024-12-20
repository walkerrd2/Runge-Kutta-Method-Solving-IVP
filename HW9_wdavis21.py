import numpy as np
import matplotlib.pyplot as plt

def davis_rk4(f,a,b,alpha,n):
    h = (b - a) / n
    # Initialize arrays for t and w
    t = np.linspace(a, b, n + 1)
    w = np.zeros(n + 1)
    
    # Initial condition
    w[0] = alpha
    
    # Runge-Kutta 4th Order iteration
    for i in range(n):
        k1 = h*f(t[i], w[i])
        k2 = h*f(t[i] + h/2, w[i] + (k1/2))
        k3 = h*f(t[i] + h/2, w[i] + (k2/2))
        k4 = h*f(t[i] + h, w[i] + k3)
        
        w[i + 1] = w[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, w

def f(t,y):
    return np.e**(t)*np.cos(t)

a=0
b=5
alpha=1
h=0.01
n=int((b-a)/h)
tvals, wvals = davis_rk4(f,a,b,alpha,n)
print(tvals, wvals) #compare to Table 5.1, page 268

plt.plot(tvals, wvals, 'r-',label="approx sol") #a graph of the approximate solution w

#def y(t):
    #return (t**2)+((1/3)*np.e**(-5*t)) #this is the exact solution to the above IVP

#yvals = y(tvals)
#plt.plot(tvals, yvals, 'bo',label="exact sol") # graph of the exact solution y(t)

plt.xlabel('x values')
plt.ylabel('y values')
plt.title("RK4 Method")
plt.legend()
plt.grid()
plt.show()

