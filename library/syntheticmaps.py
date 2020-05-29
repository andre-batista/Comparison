""" SYNTHETIC MAPS LIBRARY

    A library with functions which build synthetic contrast maps for use in
    foward simulations or inverse problems.
    
    The current options are:
    - Square
    - Equilateral triangle
    - Six-pointed star polygon (hexagram)
    - Ring with a circle in the middle
    - Two ellipsis
    - Two circles
    - A square, a equilateral triangle and a circle
    - Filled ring
    
    Common inputs:
    - I, J are the number of pixels in each direction.
    - dx, dy are the pixel sizel.
    - epsilon_rb and sigma_b are the background dielectric constants.
    - epsilon_robj and sigma_obj are the object dielectric constants.
    
    Obs.: All maps are built considering (0,0) as the center of the image. All
          lengths must be in the same unit (meters, centimeters etc).
     
    Author: Andre Costa Batista
            Universidade Federal de Minas Gerais
    
"""

import numpy as np

def build_square(I,J,dx,dy,l,epsilon_rb=1.,sigma_b=0.,epsilon_robj=1.,
                 sigma_obj=0.,center=np.array([0,0]),epsilon_r=None,
                 sigma=None,rotate=0):
    """ Build a square.
    
    - l is the length of the square.
    - center is and array with the position of the center of the square."""
    
    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    Lx, Ly = I*dx, J*dy
    x, y = np.meshgrid(np.linspace(-Lx/2,Lx/2,I,endpoint=False)+center[0],
                       np.linspace(-Ly/2,Ly/2,J,endpoint=False)+center[1])
    
    theta = np.deg2rad(rotate)
    xp = x*np.cos(theta) + y*np.sin(theta)
    yp = -x*np.sin(theta) + y*np.cos(theta)    
    
    epsilon_r[np.logical_and(np.logical_and(yp >= -l/2,
                                            yp <= l/2),
                             np.logical_and(xp >= -l/2,
                                            xp <= l/2))] = epsilon_robj
    
    sigma[np.logical_and(np.logical_and(yp >= -l/2,
                                        yp <= l/2),
                         np.logical_and(xp >= -l/2,
                                        xp <= l/2))] = sigma_obj
    
    return epsilon_r,sigma

def build_triangle(I,J,dx,dy,l,epsilon_rb=1.,sigma_b=.0,epsilon_robj=1.,
                   sigma_obj=.0,epsilon_r=None,sigma=None,center=[0,0],
                   rotate=0):
    """ Build an equilateral triangle at the center of the image.
    
    - l is the length of the triangle in sides."""

    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
        
    Lx, Ly = I*dx, J*dy
    
    xp, yp = np.meshgrid(np.arange(-Lx/2+dx/2,Lx/2,dx)+center[0],
                         np.arange(-Ly/2+dy/2,Ly/2,dy)+center[1])
    theta = np.deg2rad(rotate)
    x = xp*np.cos(theta) + yp*np.sin(theta)
    y = -xp*np.sin(theta) + yp*np.cos(theta) 
    
    epsilon_r[np.logical_and(np.logical_and(y>=-l/2,y<=2*x+l/2),
                             y<=-2*x+l/2)] = epsilon_robj
    sigma[np.logical_and(np.logical_and(y>=-l/2,y<=2*x-l/2),
                         y<=-2*x+l/2)] = sigma_obj

    return epsilon_r, sigma

def build_star(I,J,dx,dy,l,epsilon_rb=1.,sigma_b=.0,epsilon_robj=1.,
               sigma_obj=.0,epsilon_r=None,sigma=None,center=[0,0],
               rotate=0):
    """ Build a six-pointed star (hexagram) at the center of the image.
    
    - l is the length of the side of the two equilateral triangles which draws  
      the star."""

    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
        
    Lx, Ly = I*dx, J*dy
    
    xp, yp = np.meshgrid(np.arange(-Lx/2+dx/2,Lx/2,dx)+center[0],
                         np.arange(-Ly/2+dy/2,Ly/2,dy)+center[1])
    theta = np.deg2rad(rotate)
    x = xp*np.cos(theta) + yp*np.sin(theta)
    y = -xp*np.sin(theta) + yp*np.cos(theta) 
    
    epsilon_r[np.logical_and(np.logical_and(y>=-l/4,y<=3/2*x+l/2),
                             y<=-3/2*x+l/2)] = epsilon_robj
    sigma[np.logical_and(np.logical_and(y>=-l/4,y<=3/2*x+l/2),
                         y<=-3/2*x+l/2)] = sigma_obj
    
    epsilon_r[np.logical_and(np.logical_and(y<=l/4,y>=3/2*x-l/2),
                             y>=-3/2*x-l/2)] = epsilon_robj
    sigma[np.logical_and(np.logical_and(y<=l/4,y>=3/2*x-l/2),
                         y>=-3/2*x-l/2)] = sigma_obj

    return epsilon_r, sigma

def build_ring(I,J,dx,dy,ra,rb,rc,epsilon_rb=1.,sigma_b=.0,epsilon_robj=1.,
               sigma_obj=0.,delta=0.,epsilon_r=None,sigma=None,
               center=[0,0]):
    """ Build a ring with a circle in its middle.
    
    - ra and rb are the outer and inner radius of the ring.
    - rc is the radius of the circle.
    - delta is a real number which means the displacement of the figure in the
      image considering equal displacements in each axis."""

    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    xc = I*dx/2+delta
    yc = J*dy/2+delta

    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]-xc+center[0])**2+(y[j]-yc+center[1])**2)
            if r <= ra and r >= rb:
                if isinstance(epsilon_robj,float):
                    epsilon_r[j,i] = epsilon_robj
                else:
                    epsilon_r[j,i] = epsilon_robj[0]
                if isinstance(sigma_obj,float):
                    sigma[j,i] = sigma_obj
                else:
                    sigma[j,i] = sigma_obj[0]
            elif r <= rc:
                if isinstance(epsilon_robj,float):
                    epsilon_r[j,i] = epsilon_robj
                else:
                    epsilon_r[j,i] = epsilon_robj[1]
                if isinstance(sigma_obj,float):
                    sigma[j,i] = sigma_obj
                else:
                    sigma[j,i] = sigma_obj[1]
    return epsilon_r, sigma

def build_ellipses(I,J,dx,dy,la,lb,delta,epsilon_rb=1.,sigma_b=.0,
                   epsilon_robj=1.,sigma_obj=.0,epsilon_r=None,
                   sigma=None,center=[0,0]):
    """ Build two ellipses.
    
    - la is the semi-major axis.
    - lb is the semi-minor axis.
    - delta is the displacement between the center of the two ellipses."""

    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    xc = I*dx/2
    yc = J*dy/2

    for i in range(I):
        for j in range(J):

            if (x[i]+center[0]-(xc-delta))**2/la**2 + (y[j]+center[1]-yc)**2/lb**2 <= 1:
                if isinstance(epsilon_robj,float) == 1:
                    epsilon_r[j,i] = epsilon_robj
                else:
                    epsilon_r[j,i] = epsilon_robj[0]
                if isinstance(sigma_obj,float) == 1:
                    sigma[j,i] = sigma_obj
                else:
                    sigma[j,i] = sigma_obj[0]
            elif (x[i]+center[0]-(xc+delta))**2/la**2 + (y[j]+center[1]-yc)**2/lb**2 <= 1:
                if isinstance(epsilon_robj,float) == 1:
                    epsilon_r[j,i] = epsilon_robj
                else:
                    epsilon_r[j,i] = epsilon_robj[1]
                if isinstance(sigma_obj,float) == 1:
                    sigma[j,i] = sigma_obj
                else:
                    sigma[j,i] = sigma_obj[1]
    return epsilon_r, sigma

def build_2circles(I,J,dx,dy,ra,delta,epsilon_rb=1.,sigma_b=.0,
                   epsilon_robj=[1.,1.,1.], sigma_obj=[.0,.0,.0],
                   epsilon_r=None,sigma=None,center=[0,0]):
    """ Build two circles which may overlapping.

    Obs.: epsilon_robj and sigma_obj are arrays with the dielectric values of:
    (i) the left-circle, (ii) the right-circle, and (iii) the intersection.
    
    - ra is the radius of the two circles.
    - delta is the displacement of the two circles."""

    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xc1 = I*dx/2+delta
    yc1 = J*dy/2+delta
    
    xc2 = I*dx/2-delta
    yc2 = J*dy/2+delta

    for i in range(I):
        for j in range(J):

            r1 = np.sqrt((x[i]+center[0]-xc1)**2+(y[j]+center[1]-yc1)**2)
            r2 = np.sqrt((x[i]+center[0]-xc2)**2+(y[j]+center[1]-yc2)**2)
        
            if r1 <= ra and r2 <= ra:
                epsilon_r[j,i] = epsilon_robj[2]
                sigma[j,i] = sigma_obj[2]
            elif r1 <= ra:
                epsilon_r[j,i] = epsilon_robj[0]
                sigma[j,i] = sigma_obj[0]
            elif r2 <= ra:
                epsilon_r[j,i] = epsilon_robj[1]
                sigma[j,i] = sigma_obj[1]
    return epsilon_r, sigma

def build_3objects(I,J,dx,dy,ra,dela,lb,delb,lc,delc,epsilon_rb=1.,
                   sigma_b=.0,epsilon_robj=[1.,1.,1.],
                   sigma_obj=[.0,.0,.0],epsilon_r=None,sigma=None):
    """ Build three objects: a square, an equilateral triangle and a circle.
    
    Obs.: epsilon_robj and sigma_obj are arrays with the dielectric values of:
    (i) the circle, (ii) the square, and (iii) the triangle.
    
    - ra is the radius of the circle.
    - dela is an array with the displacement in each axis of the circle.
    - lb is the length of the square.
    - delb is an array with the displacement in each axis of the square.
    - lc is the length of the size of the triangle.
    - delc is an array with the displacement in each axis of the triangle."""
    
    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xca = I*dx/2+dela[0]
    yca = J*dy/2+dela[1]
    
    xcb = I*dx/2+delb[0]
    ycb = J*dy/2+delb[1]
    
    xcc = I*dx/2+delc[0]
    ycc = J*dy/2+delc[1]

    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]-xca)**2+(y[j]-yca)**2)
           
            if r <= ra:
                epsilon_r[j,i] = epsilon_robj[0]
                sigma[j,i] = sigma_obj[0]
                
            elif (x[i] >= xcb-lb/2 and x[i] <= xcb+lb/2 
                  and y[j] >= ycb-lb/2 and y[j] <= ycb+lb/2):
                
                epsilon_r[j,i] = epsilon_robj[1]
                sigma[j,i] = sigma_obj[1]
                
            elif (x[i] >= xcc-lc/2 and x[i] <= xcc+lc/2 
                  and y[j] >= ycc-lc/2 and y[j] <= ycc+lc/2):
                
                FLAG = False
                v = lc/2*np.sqrt(3)
                if y[j] >= ycc:
                    a = lc/2/v
                    b = ycc-lc/2/v*(xcc-v/2)
                    if y[j] <= a*x[i]+b:
                        FLAG = True
                else:
                    yp = ycc + ycc-y[j]
                    a = lc/2/v
                    b = ycc-lc/2/v*(xcc-v/2)
                    if yp <= a*x[i]+b:
                        FLAG = True
                if FLAG is True:
                    epsilon_r[j,i] = epsilon_robj[2]
                    sigma[j,i] = sigma_obj[2]
    return epsilon_r, sigma

def build_filledring(I,J,dx,dy,ra,rb,epsilon_rb=1.,sigma_b=.0,
                     epsilon_robj=[1.,1.],sigma_obj=[.0,.0],
                     epsilon_r=None,sigma=None,center=[0,0]):
    """ Build a filled ring at the center of the image.""

    Obs.: epsilon_robj and sigma_obj are arrays with the dielectric values of:
    (i) the ring, (ii) the inner.
    
    - ra is the outer radius of the ring.
    - rb is the inner radius of the ring."""

    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xc = I*dx/2
    yc = J*dy/2
    
    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]+center[0]-xc)**2+(y[j]+center[1]-yc)**2)
           
            if r <= ra and r >= rb:
                epsilon_r[j,i] = epsilon_robj[0]
                sigma[j,i] = sigma_obj[0]
            elif r <= rb:
                epsilon_r[j,i] = epsilon_robj[1]
                sigma[j,i] = sigma_obj[1]
    return epsilon_r, sigma

def build_sine(I,J,wavelength,epsilon_rmin,sigma_min,epsilon_rpeak,sigma_peak):
    
    x,y = np.meshgrid(np.linspace(-wavelength/4,wavelength/4,I,endpoint=False),
                      np.linspace(-wavelength/4,wavelength/4,J,endpoint=False))
    
    epsilon_r = epsilon_rmin + (epsilon_rpeak*np.cos(2*np.pi/wavelength*x)
                                * np.cos(2*np.pi/wavelength*y))
    
    sigma = sigma_min + (sigma_peak*np.cos(2*np.pi/wavelength*x)
                         * np.cos(2*np.pi/wavelength*y))
    
    return epsilon_r, sigma

def build_x(I,J,dx,dy,height,width,thick,epsilon_rb=1.,sigma_b=.0,
            epsilon_robj=1.,sigma_obj=.0,center=[0,0],epsilon_r=None,
            sigma=None):
    
    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    Lx, Ly = I*dx, J*dy
    x, y = np.meshgrid(np.linspace(-Lx/2,Lx/2,I,endpoint=False),
                       np.linspace(-Ly/2,Ly/2,J,endpoint=False))
    
    a = height/width
    
    for i in range(I):
        for j in range(J):
            if (x[j,i]-center[0] >= - width/2 and x[j,i]-center[0] <= width/2 and
                y[j,i]+center[1] >= -height/2 and y[j,i]+center[1] <= height/2 and
                ((y[j,i]+center[1] <= a*(x[j,i]-center[0] + thick/2) 
                  and y[j,i]+center[1] >= a*(x[j,i]-center[0]-thick/2))
                 or (y[j,i]+center[1] >= -a*(x[j,i]-center[0] + thick/2) 
                     and y[j,i]+center[1] <= -a*(x[j,i]-center[0]-thick/2)))):
                epsilon_r[j,i] = epsilon_robj
                sigma[j,i] = sigma_obj
                
    return epsilon_r, sigma

def build_cross(I,J,dx,dy,height,width,thick,epsilon_rb=1.,sigma_b=.0,
                epsilon_robj=1.,sigma_obj=.0,center=[0,0],epsilon_r=None,
                sigma=None):
    
    if epsilon_r is None:
        epsilon_r = epsilon_rb*np.ones((J,I))
    if sigma is None:
        sigma = sigma_b*np.ones((J,I))
    Lx, Ly = I*dx, J*dy
    x, y = np.meshgrid(np.linspace(-Lx/2,Lx/2,I,endpoint=False),
                       np.linspace(-Ly/2,Ly/2,J,endpoint=False))
    
    for i in range(I):
        for j in range(J):
            if ((x[j,i]+center[0] >= -width/2 and x[j,i]+center[0] <= width/2
                 and y[j,i]+center[1] >= -thick/2 and y[j,i]+center[1] <= thick/2)
                or (x[j,i]+center[0] >= -thick/2 and x[j,i]+center[0] <= thick/2
                and y[j,i]+center[1] >= -height/2 and y[j,i]+center[1] <= height/2)):
                epsilon_r[j,i] = epsilon_robj
                sigma[j,i] = sigma_obj

    return epsilon_r, sigma
