from re import L, M
import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pl
import cv2






def GetPwrSpcDensity(image):
    #https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Length=min(len(fourier_amplitudes),len(knrm))
    Abins, _, _ = stats.binned_statistic(knrm[0:Length], fourier_amplitudes[0:Length],
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    #return Abins
    pl.loglog(kvals, Abins)
    pl.xlabel("$k$")
    pl.ylabel("$P(k)$")
    pl.tight_layout()
    pl.savefig(r"C:\Working\testOutput\cloud_power_spectrum9.png", dpi = 300, bbox_inches = "tight")

image=cv2.imread(r"C:\Working\testOutput\plop9.jpg")
GetPwrSpcDensity(image)


ddd

if True is False:
        #notes for normalising vectors/make as unit vectors - to get dot product without magnitude
    #create two random integer vectors
    n=4
    v1=np.round(20*np.random.randn(4))
    v2=np.round(20*np.random.randn(4))
    #compute the lengths of the individual vectors, and magnitude of their dot products
    #compute magnitude
    v1m=np.sqrt(np.dot(v1,v1))
    v2m=np.sqrt(np.dot(v1,v1))
    dpm=np.abs(np.dot(v1,v2))
    #normalise the vectors - creating unit vectors
    #normalise dot product
    v1u=v1/v1m
    v2u=v2/v2m
    #compute magnitude of dot product of unit length vectors
    dpm=np.abs(np.dot(v1u,v2u))


    import matplotlib.pyplot as plt
    #2d input vector
    v=np.array([3,-2])
    #2*2 transformation matrix
    A=np.array([[1,-1],[2,1]])
    #output vector is Av (convert v to column)
    w=A@np.matrix.transpose(v)
    #plot vector
    plt.plot([0,v[0]],[0,v[1]],label='v')
    plt.plot([0,w[0]],[0,w[1]],label='Av')
    plt.grid()      
    plt.axis((-6,6,-6,6))
    plt.legend()
    plt.title("Rotation & stretching")
    plt.show()



    import matplotlib.pyplot as plt
    import math
    #2d input vector
    v=np.array([5,-2])
    #2*2 transformation matrix
    #rotation matrix
    theta = np.radians(45)

    #A= np.array(( (np.cos(theta), -np.sin(theta)),
    #               (np.sin(theta),  np.cos(theta)) ))
    A=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])

    #output vector is Av (convert v to column)
    w=A@np.matrix.transpose(v)
    #plot vector
    plt.plot([0,v[0]],[0,v[1]],label='v')
    plt.plot([0,w[0]],[0,w[1]],label='Av')
    plt.grid()
    plt.axis((-6,6,-6,6))
    plt.legend()
    plt.title("Rotation & stretching")
    plt.show()




    numbers=np.linspace(1,10,10)
    def square(m):
        return m*m 

    def even(n):
        return n%2==0

    #map maps the function to an input
    squares=list(map(square,numbers))
    #filter only maps if reutrn is true
    squares=list(filter(even,numbers))
    squares

    #lambda function
    squares=list(map(lambda x: x*x ,numbers))
    squares

    class teststatic():
        def __init__(self) -> None:
            pass

        #@staticmethod
        #def test(in):
        #    pass
        #    print(in)
        
        @classmethod
        def from_json(cls,filename):
            c=cls()
            return c
        
    #*ark **kwargs

    def My_func(*args,**kwargs):
        print("hello world",args,kwargs)

    My_func("abc","plop",1,2,3,key=123,plopplop="hehe")


import matplotlib.pyplot as plt
#generate xy coordsinates for a circle
x= np.linspace(-np.pi,np.pi,10)
xy=np.vstack((np.cos(x),np.sin(x))).T
print(np.shape(xy))
plt.ylim=(-5,5)
plt.xlim=(-5,5)
plt.plot(xy[:,0],xy[:,1],'o')
#2 by 2 matrix
T=np.array([[1,2],[2,1]])
#multiply matrix by coords
newxy=xy@T
#plt.axis('square')
plt.plot(newxy[:,0],newxy[:,1],'o')

plt.show()