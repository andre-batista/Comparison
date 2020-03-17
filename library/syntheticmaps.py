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

def build_square(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,l,
                 center=np.array([0,0])):
    """ Build a square.
    
    - l is the length of the square.
    - center is and array with the position of the center of the square."""
    
    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    Lx, Ly = I*dx, J*dy
    x = np.linspace(-Lx/2,Lx/2,I,endpoint=False)
    y = np.linspace(-Ly/2,Ly/2,J,endpoint=False)
    
    epsilon_r[np.ix_(np.logical_and(x >= center[0]-l/2, x <= center[0]+l/2),
                     np.logical_and(y >= center[1]-l/2, y <= center[1]+l/2))] = epsilon_robj
    
    sigma[np.ix_(np.logical_and(x >= center[0]-l/2, x <= center[0]+l/2),
                 np.logical_and(y >= center[1]-l/2, y <= center[1]+l/2))] = sigma_obj
    
    return epsilon_r,sigma

def build_triangle(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,l):
    """ Build an equilateral triangle at the center of the image.
    
    - l is the length of the triangle in sides."""

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(dx,dx*(I+1),dx)
    y = np.arange(dy,dy*(J+1),dy)
    Lx, Ly = I*dx, J*dy

    for i in range(I):
        for j in range(J):

            if x[i] >= Lx/2-l/2 and x[i] <= Lx/2+l/2:
                a = x[i]-.5*(Lx-l)
                FLAG = False
                if y[j] < Ly/2:
                    b = y[j]-.5*(Ly-l)
                    v = -.5*a+l/2
                    if b >= v:
                        FLAG = True
                else:
                    b = y[j]-Lx/2
                    v = .5*a
                    if b <= v:
                        FLAG = True
                if FLAG is True:
                    epsilon_r[i,j] = epsilon_robj
                    sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_star(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,l):
    """ Build a six-pointed star (hexagram) at the center of the image.
    
    - l is the length of the side of the two equilateral triangles which draws  
      the star."""

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(dx,dx*(I+1),dx)
    y = np.arange(dy,dy*(J+1),dy)
    Lx, Ly = I*dx, J*dy
    xc = l/6

    for i in range(I):
        for j in range(J):

            if x[i]+xc >= Lx/2-l/2 and x[i]+xc <= Lx/2+l/2:
                a = x[i]+xc-.5*(Lx-l)
                FLAG = False
                if y[j] < Ly/2:
                    b = y[j]-.5*(Ly-l)
                    v = -.5*a+l/2
                    if b >= v:
                        FLAG = True
                else:
                    b = y[j]-Lx/2
                    v = .5*a
                    if b <= v:
                        FLAG = True
                if FLAG == True:
                    epsilon_r[i,j] = epsilon_robj
                    sigma[i,j] = sigma_obj

    for i in range(I):
        for j in range(J):

            if x[i]-xc >= Lx/2-l/2 and x[i]-xc <= Lx/2+l/2:
                a = x[i]-xc-.5*(Lx-l)
                FLAG = False
                if y[j] < Ly/2:
                    b = y(j)-.5*(Ly-l)
                    v = .5*a
                    if b >= v:
                        FLAG = True
                else:
                    b = y[j]-Lx/2
                    v = -.5*a+l/2
                    if b <= v:
                        FLAG = True
                if FLAG is True:
                    epsilon_r[i,j] = epsilon_robj
                    sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_ring(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,a,rb,rc,
               delta):
    """ Build a ring with a circle in its middle.
    
    - ra and rb are the outer and inner radius of the ring.
    - rc is the radius of the circle.
    - delta is a real number which means the displacement of the figure in the
      image considering equal displacements in each axis."""

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    xc = I*dx/2+delta
    yc = J*dy/2+delta

    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]-xc)**2+(y[j]-yc)**2)
            if r <= ra and r >= rb:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
            elif r <= rc:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_ellipses(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,la,lb,
                   delta):
    """ Build two ellipses.
    
    - la is the semi-major axis.
    - lb is the semi-minor axis.
    - delta is the displacement between the center of the two ellipses."""

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    xc = I*dx/2
    yc = J*dy/2

    for i in range(I):
        for j in range(J):

            if (x[i]-(xc-delta))**2/la**2 + (y[j]-yc)**2/lb**2 <= 1:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
            elif (x[i]-(xc+delta))**2/la**2 + (y[j]-yc)**2/lb**2 <= 1:
                epsilon_r[i,j] = epsilon_robj
                sigma[i,j] = sigma_obj
    return epsilon_r, sigma

def build_2circles(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,ra,
                   delta):
    """ Build two circles which may overlapping.

    Obs.: epsilon_robj and sigma_obj are arrays with the dielectric values of:
    (i) the left-circle, (ii) the right-circle, and (iii) the intersection.
    
    - ra is the radius of the two circles.
    - delta is the displacement of the two circles."""

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xc1 = I*dx/2+delta
    yc1 = J*dy/2+delta
    
    xc2 = I*dx/2-delta
    yc2 = J*dy/2+delta

    for i in range(I):
        for j in range(J):

            r1 = np.sqrt((x[i]-xc1)**2+(y[j]-yc1)**2)
            r2 = np.sqrt((x[i]-xc2)**2+(y[j]-yc2)**2)
        
            if r1 <= ra and r2 <= ra:
                epsilon_r[i,j] = epsilon_robj[2]
                sigma[i,j] = sigma_obj[2]
            elif r1 <= ra:
                epsilon_r[i,j] = epsilon_robj[0]
                sigma[i,j] = sigma_obj[0]
            elif r2 <= ra:
                epsilon_r[i,j] = epsilon_robj[1]
                sigma[i,j] = sigma_obj[1]
    return epsilon_r, sigma

def build_3objects(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,ra,dela,
                   lb,delb,lc,delc):
    """ Build three objects: a square, an equilateral triangle and a circle.
    
    Obs.: epsilon_robj and sigma_obj are arrays with the dielectric values of:
    (i) the circle, (ii) the square, and (iii) the triangle.
    
    - ra is the radius of the circle.
    - dela is an array with the displacement in each axis of the circle.
    - lb is the length of the square.
    - delb is an array with the displacement in each axis of the square.
    - lc is the length of the size of the triangle.
    - delc is an array with the displacement in each axis of the triangle."""
    
    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
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
                epsilon_r[i,j] = epsilon_robj[0]
                sigma[i,j] = sigma_obj[0]
                
            elif (x[i] >= xcb-lb/2 and x[i] <= xcb+lb/2 
                  and y[j] >= ycb-lb/2 and y[j] <= ycb+lb/2):
                
                epsilon_r[i,j] = epsilon_robj[1]
                sigma[i,j] = sigma_obj[1]
                
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
                    epsilon_r[i,j] = epsilon_robj[2]
                    sigma[i,j] = sigma_obj[2]
    return epsilon_r, sigma

def build_filledring(I,J,dx,dy,epsilon_rb,sigma_b,epsilon_robj,sigma_obj,ra,rb):
    """ Build a filled ring at the center of the image.""

    Obs.: epsilon_robj and sigma_obj are arrays with the dielectric values of:
    (i) the ring, (ii) the inner.
    
    - ra is the outer radius of the ring.
    - rb is the inner radius of the ring."""

    epsilon_r = epsilon_rb*np.ones((I,J))
    sigma = sigma_b*np.ones((I,J))
    x = np.arange(1,I+1)*dx
    y = np.arange(1,J+1)*dy
    
    xc = I*dx/2
    yc = J*dy/2

    for i in range(I):
        for j in range(J):

            r = np.sqrt((x[i]-xc)**2+(y[j]-yc)**2)
           
            if r <= ra and r >= rb:
                epsilon_r[i,j] = epsilon_robj[0]
                sigma[i,j] = sigma_obj[0]
            elif r <= rb:
                epsilon_r[i,j] = epsilon_robj[1]
                sigma[i,j] = sigma_obj[1]
    return epsilon_r, sigma