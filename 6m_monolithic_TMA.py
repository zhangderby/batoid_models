import numpy as np
import batoid as btd
import time
from copy import deepcopy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from scipy.interpolate import griddata
from datetime import date
import time
from itertools import chain, compress
import matplotlib.colors as mcolors

class model():

    def __init__(self,
                 wavelength=650e-9,
                 npix_pupil=512):
        
        self.wavelength = wavelength
        self.npix_pupil = npix_pupil

        # field bias
        self.field_bias = 0.17

        # M1 prescription
        self.M1_RoC = -16.256
        self.M1_conic = -0.995357
        self.M1_innerD = 1.38
        self.M1_outerD = 6.42

        # M1 position
        self.M1_dx = 0
        self.M1_dy = 0
        self.M1_dz = 0
        self.M1_tx = 0
        self.M1_ty = 0

        # M2 prescription
        self.M2_RoC = -1.660575
        self.M2_conic = -1.566503
        self.M2_outerD = 0.7147252

        # M2 position
        self.M2_dx = 0
        self.M2_dy = 0
        self.M2_dz = -7.400
        self.M2_tx = 0
        self.M2_ty = 0

        # M3 prescription
        self.M3_RoC = -1.830517
        self.M3_conic = -0.7180517
        self.M3_outerD = 0.920733

        # M3 position
        self.M3_dx = 0
        self.M3_dy = 0
        self.M3_dz = 0.05
        self.M3_tx = 0
        self.M3_ty = 0

        # M4 prescription
        self.M4_outerD = 0.0991468

        # M4 position
        self.M4_dx = 0
        self.M4_dy = 0
        self.M4_dz = -0.983629
        self.M4_tx = 0
        self.M4_ty = 0

        # detector prescription
        self.det_outerD = 0.9

        # detector position
        self.det_dx = 0
        self.det_dy = 0
        self.det_dz = 0.237581
        self.det_tx = 0
        self.det_ty = 0

        self.init_model()

    def init_model(self):

        # build batoid model
        M1coord = defineCoordinate(posX=self.M1_dx, posY=self.M1_dy, posZ=self.M1_dz, anglX=self.M1_tx, anglY=self.M1_ty).local()
        # M1 = optic('M1',type='Quadric', RoC=self.M1_RoC, conic=self.M1_conic, inDiam=self.M1_innerD, outDiam=self.M1_outerD).mirrorWithSpider(width=0.15, height=4)
        M1 = optic('M1',type='Quadric', RoC=self.M1_RoC, conic=self.M1_conic, outDiam=self.M1_outerD, coordSys=M1coord).mirror()

        M2coord = defineCoordinate(posX=self.M2_dx, posY=self.M2_dy, posZ=self.M2_dz, anglX=self.M2_tx, anglY=self.M2_ty).local()
        M2 = optic('M2', type='Quadric', RoC=self.M2_RoC, conic=self.M2_conic, outDiam=self.M2_outerD, coordSys=M2coord).mirror()

        M3coord = defineCoordinate(posX=self.M3_dx, posY=self.M3_dy, posZ=self.M3_dz, anglX=self.M3_tx, anglY=self.M3_ty).local()
        M3 = optic('M3', type='Quadric', RoC=self.M3_RoC, conic=self.M3_conic, outDiam=self.M3_outerD, coordSys=M3coord).mirror()

        M4coord = defineCoordinate(posX=self.M4_dx, posY=self.M4_dy, posZ=self.M4_dz, anglX=self.M4_tx, anglY=self.M4_ty).local()
        M4 = optic('M4', outDiam=self.M4_outerD, coordSys=M4coord).flatMirror()

        Dcoord = defineCoordinate(posX=self.det_dx, posY=self.det_dy, posZ=self.det_dz, anglX=self.det_tx, anglY=self.det_ty).local()
        D = optic('D', outDiam=self.det_outerD, coordSys=Dcoord).detector()

        self.osys = build.compoundOptic(M1, M2, M3, M4, D, pupilSize=6.46, backDist=7.5, EPcoord=defineCoordinate(posZ=0).local())
        
    def add_motion(self, 
                   M1_dx=0, M1_dy=0, M1_dz=0, M1_tx=0, M1_ty=0,
                   M2_dx=0, M2_dy=0, M2_dz=0, M2_tx=0, M2_ty=0,
                   M3_dx=0, M3_dy=0, M3_dz=0, M3_tx=0, M3_ty=0,
                   M4_dx=0, M4_dy=0, M4_dz=0, M4_tx=0, M4_ty=0,
                   det_dx=0, det_dy=0, det_dz=0, det_tx=0, det_ty=0,):

        # M1 position
        self.M1_dx += M1_dx
        self.M1_dy += M1_dy
        self.M1_dz += M1_dz
        self.M1_tx += M1_tx
        self.M1_ty += M1_ty

        # M2 position
        self.M2_dx += M2_dx
        self.M2_dy += M2_dy
        self.M2_dz += M2_dz
        self.M2_tx += M2_tx
        self.M2_ty += M2_ty

        # M3 position
        self.M3_dx += M3_dx
        self.M3_dy += M3_dy
        self.M3_dz += M3_dz
        self.M3_tx += M3_tx
        self.M3_ty += M3_ty

        # M4 position
        self.M4_dx += M4_dx
        self.M4_dy += M4_dy
        self.M4_dz += M4_dz
        self.M4_tx += M4_tx
        self.M4_ty += M4_ty

        # detector position
        self.det_dx += det_dx
        self.det_dy += det_dy
        self.det_dz += det_dz
        self.det_tx += det_tx
        self.det_ty += det_ty

        self.init_model()

    def reset_motion(self):

        # M1 position
        self.M1_dx = 0
        self.M1_dy = 0
        self.M1_dz = 0
        self.M1_tx = 0
        self.M1_ty = 0

        # M2 position
        self.M2_dx = 0
        self.M2_dy = 0
        self.M2_dz = -7.400
        self.M2_tx = 0
        self.M2_ty = 0

        # M3 position
        self.M3_dx = 0
        self.M3_dy = 0
        self.M3_dz = 0.05
        self.M3_tx = 0
        self.M3_ty = 0

        # M4 position
        self.M4_dx = 0
        self.M4_dy = 0
        self.M4_dz = -0.983629
        self.M4_tx = 0
        self.M4_ty = 0

        # detector position
        self.det_dx = 0
        self.det_dy = 0
        self.det_dz = 0.237581
        self.det_tx = 0
        self.det_ty = 0

        self.init_model()
        
    def get_opd(self, fieldX=0, fieldY=0, TT_remove=True, plot=True):
        
        # apply field bias
        fieldY += self.field_bias

        # get opd data
        infoTelescope = details(self.osys, fieldX=fieldX, fieldY=fieldY, wavelength=self.wavelength)
        self.wf = infoTelescope.wavefront(npx=self.npix_pupil)

        # remove tip/tilt if desired
        if TT_remove:
            corrWF, removed = correction.removeTiltWavefront(self.wf, self.wavelength, tiltX=True, tiltY=True)
            # self.wf = corrWF
        
        # plot if desired
        if plot:
            plotting.wavefront(self.wf, self.wavelength, [fieldX, fieldY-self.field_bias])
            print(f'Tilt Removed:  X = {np.round(removed[0], decimals=4)}, Y = {np.round(removed[1], decimals=4)} waves\n')

        # what is this stupid masked array object
        what =  self.wf.array 
        the = what.data * ~what.mask
        heck = the * self.wavelength

        heck[heck==0] = np.nan

        return heck
        


        
class rotation:
    
    def __init__(self, anglX=0, anglY=0, anglZ=0):
        self.anglX = anglX
        self.anglY = anglY
        self.anglZ = anglZ
    
    def Rx(self):
        theta = np.deg2rad(self.anglX)
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    
    def Ry(self):
        theta = np.deg2rad(self.anglY)
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
    
    def Rz(self):
        theta = np.deg2rad(self.anglZ)
        return np.array([[np.cos(theta), -np.sin(theta),0 ],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    
class defineCoordinate:
    
    def __init__(self, posX=0, posY=0, posZ=0, anglX=0, anglY=0, anglZ=0):
        self.posX = posX
        self.posY = posY
        self.posZ = posZ
        self.anglX = anglX
        self.anglY = anglY
        self.anglZ = anglZ
    
    def local(self):
        x, y, z = self.posX, self.posY, self.posZ
        R = rotation(self.anglX, self.anglY, self.anglZ)
        Rx, Ry, Rz = R.Rx(), R.Ry(), R.Rz()
        return btd.CoordSys(np.array([x, y, z]), np.matmul(Rx, Ry, Rz))
    
class optic:

    def __init__(self, name,  
                 type='Plane', RoC=-1, conic=0, inDiam=0, outDiam=1, indx = 1.0,
                 coordSys=defineCoordinate().local()
                ):
        self.name = name
        self.type = type
        self.RoC = RoC
        self.conic = conic
        self.inDiam = inDiam
        self.outDiam = outDiam
        self.indx = indx
        self.coordSys = coordSys

    def obscMirror(self):
        inDiam, outDiam = self.inDiam, self.outDiam
        if inDiam == 0:
            return btd.ObscAnnulus(outDiam/2, outDiam/2+10)
        else:
            return btd.ObscUnion(btd.ObscCircle(inDiam/2), btd.ObscAnnulus(outDiam/2, outDiam/2+10))

    def obscMirrorWithSpider(self, width, height):
        obsc = btd.ObscUnion(self.obscMirror(), 
                             btd.ObscRectangle(width=width, height=height, 
                                               x=height/2, y=0.0, theta=np.pi/2),
                             btd.ObscRectangle(width=width, height=height, 
                                               x=-height/2*np.sin(np.pi/6), y=-4/2*np.cos(np.pi/6), theta=-np.pi/6),
                             btd.ObscRectangle(width=width, height=height, 
                                               x=-height/2*np.sin(5*np.pi/6), y=-4/2*np.cos(5*np.pi/6), theta=-5*np.pi/6))
        return obsc
            
    def flatMirror(self):
        type = self.type
        if type == 'Plane':
            obj = btd.Mirror(name=self.name,
                             surface=btd.Plane(), 
                             obscuration=self.obscMirror(),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=btd.ConstMedium(self.indx),
                             outMedium=btd.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj    

    def mirror(self):
        type = self.type 
        if type == 'Quadric':
            obj = btd.Mirror(name=self.name,
                             surface=btd.Quadric(self.RoC, self.conic), 
                             obscuration=self.obscMirror(),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=btd.ConstMedium(self.indx),
                             outMedium=btd.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj 

    def mirrorWithSpider(self, width=0, height=0):
        type = self.type 
        if type == 'Quadric':
            obj = btd.Mirror(name=self.name,
                             surface=btd.Quadric(self.RoC, self.conic), 
                             obscuration=self.obscMirrorWithSpider(width, height),
                             inDiam=self.inDiam,
                             outDiam=self.outDiam,
                             inMedium=btd.ConstMedium(self.indx),
                             outMedium=btd.ConstMedium(self.indx),
                             coordSys=self.coordSys,
                             skip=False)
            return obj 

    def detector(self):
        type = self.type
        if type == 'Plane':
            obj = btd.Detector(btd.Plane(), 
                               coordSys=self.coordSys, 
                               name=self.name,
                               inDiam=self.inDiam,
                               outDiam=self.outDiam)
            return obj

class build:

    def stopSurface(coordSys=defineCoordinate().local()):
        return btd.Interface(btd.Plane(), coordSys=coordSys)
    
    def compoundOptic(*args, pupilSize, backDist=40, EPcoord):
        optics = [item for item in args]
        sys = btd.CompoundOptic(optics,
                        backDist=backDist, 
                        stopSurface=build.stopSurface(coordSys=EPcoord),
                        pupilSize = pupilSize)
        return sys
    
class ray:
    
    def __init__(self,
                 opticalSys, fieldX=0, fieldY=0, wavelength=633e-9, nbOfRays=1
                ):
        self.opticalSys = opticalSys
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.wavelength = wavelength
        self.nbOfRays = nbOfRays

    def fromPupil(self):
        N, size = self.nbOfRays, self.opticalSys.pupilSize
        th = 2 * np.pi * np.random.rand(int(N))
        cth, sth = np.cos(th), np.sin(th)
        d = (size) * np.random.rand(int(N))
        x, y = (d/2) * cth, (d/2) * sth
        
        fieldX, fieldY = self.fieldX, self.fieldY
        rays = btd.RayVector.fromStop(x=x, y=y, 
                                      optic=self.opticalSys,
                                      wavelength=self.wavelength,
                                      dirCos=np.array([fieldX*np.pi/180, fieldY*np.pi/180, 1.]),
                                      flux=1.)
        return rays

    def deleteVignettedRays(rv):
        which = np.where(np.invert(rv.vignetted))[0].tolist()
        return rv[which]
    
class details:

    def __init__(self, 
                 optic, fieldX=0, fieldY=0, wavelength=633e-9
                ):
        self.optic = optic
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.wavelength = wavelength

    def workingFNumber(self):
        opt, wl = self.optic, self.wavelength
        D = opt.pupilSize
        ray = btd.RayVector.fromStop(x=0, y=D/2,
                                     optic=opt,
                                     wavelength=wl,
                                     dirCos=np.array([0., 0., 1.]),
                                     flux=1.)

        opt.trace(ray, reverse=False)
        return 1/(2*np.sin(np.arccos(ray.vz[0])))

    def airyRadius(self):
        return 1.22 * self.wavelength * self.workingFNumber() * 1e6

    def wavefront(self, npx=64):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        Rref = btd.analysis.exitPupilPos(optic=opt, wavelength=wl)[-1]
        WF = btd.analysis.wavefront(optic=opt,
                                    theta_x=fieldX*np.pi/180, theta_y=fieldY*np.pi/180,
                                    wavelength=wl,
                                    nx=npx,
                                    sphereRadius=Rref,
                                    projection='zemax',
                                    reference='chief')
        return WF

    def wavefrontZnk(self, npx=64, jmax=12):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        Rref = btd.analysis.exitPupilPos(optic=opt, wavelength=wl)[-1]
        Znk = btd.analysis.zernike(optic=opt, 
                                   theta_x=fieldX*np.pi/180, theta_y=fieldY*np.pi/180, 
                                   wavelength=wl,
                                   nx=npx, 
                                   sphereRadius=Rref, 
                                   projection='zemax',
                                   reference='chief', 
                                   jmax=jmax, eps=0.0)
        Noll_Znk_names = ["", "1", "4^(1/2) (P) * COS (A)", "4^(1/2) (P) * SIN (A)", "3^(1/2) (20^2-1)", 
                          "6^(1/2) (p^2) * SIN (2A)", "6^(1/2) (p^2) * COS (2A)", "8^(1/2) (3p^3 - 2p) * SIN (A)",
                          "8^(1/2) (3p^3 - 2p) * COS (A)", "8^(1/2) (p^3) * SIN (3A)", "8^(1/2) (p^3) * COS (3A)", 
                          "5^(1/2) (6p^4-6p^2+1)", "10^(1/2) (4p^4-3p^2) * COS (2A)", "10^(1/2) (4p^4 - 3p^2) * SIN (2A)"]
        #for i in range(1, len(Znk)):
         #   if i<=9:
          #      if Znk[i] >= 0:
           #         print(f'{"Z":<4}{i:<7}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}')
            #    else:
             #       print(f'{"Z":<4}{i:<6}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}')
            #else:
             #   if Znk[i] >= 0:
              #      print(f'{"Z":<3}{i:<8}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}') 
               # else:
                #    print(f'{"Z":<3}{i:<7}{Znk[i]:.9f}{"":<2}{":":<7}{Noll_Znk_names[i]}')
        return Znk

    def focalLength(self):
        return btd.analysis.focalLength(optic=self.optic,
                                        theta_x=0, theta_y=0,
                                        wavelength=self.wavelength,
                                        projection='zemax')

    def posChiefRayAtImgPlane(self):
        anglX, anglY = self.fieldX*np.pi/180, self.fieldY*np.pi/180
        chiefRay = btd.RayVector.fromStop(x=0, y=0,
                                          optic=self.optic,
                                          wavelength=self.wavelength,
                                          dirCos=np.array([anglX, anglY, 1.]),
                                          flux=1.)
        self.optic.trace(chiefRay, reverse=False)
        return chiefRay.r[0]

    def huygensPSF(self, npx=32):
        opt, wl = self.optic, self.wavelength
        fieldX, fieldY = self.fieldX, self.fieldY
        WF = details(opt, fieldX=fieldX, fieldY=fieldY, wavelength=wl).wavefront(npx=1024)
        corrWF, _ = correction.removeTiltWavefront(WF, wl, tiltX=False, tiltY=True)
        RMS = corrWF.std()
        StehlRatio = np.exp(-(2*np.pi*RMS)**2)
        PSF = btd.analysis.huygensPSF(optic=opt,
                                      wavelength=wl,
                                      theta_x=fieldX*np.pi/180, theta_y=fieldY*np.pi/180,
                                      nx=npx,
                                      projection='zemax',
                                      reference='chief')
        PSF = PSF.array * StehlRatio/np.max(PSF.array)
        return PSF, StehlRatio
        

class correction:

    def removeTiltWavefront(WF, wl, tiltX=True, tiltY=True):
        if isinstance(WF, btd.lattice.Lattice):
            WF = WF.array
            
        rows, cols = WF.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x, y = np.meshgrid(x, y)

        X = np.vstack((x.flatten(), y.flatten())).T
        Z = WF.flatten()

        coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
        a, b = coeffs
        if tiltX and tiltY:
            fitted_plane = a * x + b * y
            removed = np.abs([a, b]) * wl * 1e9
        elif tiltX and not tiltY:
            fitted_plane = a * x
            removed = np.abs([a, 0.0]) * wl * 1e9
        elif not tiltX and tiltY:
            fitted_plane = b * y
            removed = np.abs([0.0, b]) * wl * 1e9

        corrWF = WF - fitted_plane.reshape(rows, cols)
        avg = (np.max(corrWF) + np.min(corrWF)) / 2

        return corrWF - avg, removed
    

class plotting:

    def sag3D(optic):
        inDiam, outDiam = optic.inDiam, optic.outDiam
        th = 2 * np.pi * np.random.rand(int(1e6))
        cth, sth = np.cos(th), np.sin(th)
        d = (outDiam - inDiam) * np.random.rand(int(1e6)) + inDiam
        x = (d/2) * cth
        y = (d/2) * sth
        z = optic.surface.sag(x, y) * 1000
        
        xi = np.linspace(x.min(), x.max(), 64)
        yi = np.linspace(y.min(), y.max(), 64)
        X,Y = np.meshgrid(xi, yi)
        Z = griddata((x,y), z, (X,Y), method='nearest')
        
        fig = go.Figure(data=[go.Surface(x=xi,y=yi,z=Z,
                        colorscale='Spectral',
                        reversescale=True,
                        colorbar=dict(thickness=30,
                                      tickvals=np.linspace(np.min(Z), np.max(Z), 10),
                                      title='Surface Sag (mm)'))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          scene=dict(zaxis=dict(range=[np.min(Z), 0],title='sag (mm)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')))
        fig.show()

    def spotDiagram(rvObj, fieldBias, rvImg, wl, scale, AiryR):
        ## Centroid method
        cx = np.sum(rvImg.x)/len(rvImg.x)
        cy = np.sum(rvImg.y)/len(rvImg.y)

        fieldsObj = rvObj.v[0]
        anglX, anglY = np.round(fieldsObj[0]*180/np.pi, decimals=3), np.round(fieldsObj[1]*180/np.pi-fieldBias, decimals=3)
        
        fig, ax = plt.subplots()
        ax.scatter((rvImg.x - cx) * 1e6, (rvImg.y - cy) * 1e6, s=0.25 ,c='blue', marker='.')
        
        if AiryR is not None:
            ax.add_patch(plt.Circle((0, 0), 11.97, color='k', fill=False))
        ax.set_title(f'OBJ: {anglX}, {anglY} (deg)', fontsize=14)
        ax.set_xlabel(f'IMA: {np.round(cx * 1000, decimals=3)}, {np.round(cy * 1000, decimals=3)} mm', fontsize=14)
        if scale is not None:
            ax.set_xlim(-scale/2, scale/2)
            ax.set_ylim(-scale/2, scale/2)
            ax.set_ylabel(f'{float(scale)}', fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both',       # changes apply to the both axis
                       which='both',      # both major and minor ticks are affected
                       bottom=False,
                       top=False,
                       left=False,
                       right=False,
                       labelbottom=False,
                       labelleft=False)
        ax.grid()
        ax.set_axisbelow(True)
        fig.canvas.header_visible = False
        plt.subplots_adjust(left=-0.1)
        plt.show()

        print(f'{date.today().strftime("%m/%d/%Y")}')
        print(f'Units are µm. Airy Radius: {np.round(AiryR, decimals=3)} µm. Wavelength: {wl * 1e6} µm.')
        Rms_R = np.sqrt(np.sum((rvImg.x - cx)**2)/len(rvImg.x) + np.sum((rvImg.y - cy)**2)/len(rvImg.y)) * 1e6
        print(f'RMS radius :    {np.round(Rms_R, decimals=3)}')
        R = np.sqrt((rvImg.x - cx)**2 + (rvImg.y - cy)**2)
        Geo_R = np.max(R) * 1e6
        print(f'GEO radius :    {np.round(Geo_R, decimals=3)}')
        print(f'Scale bar  :   {float(scale)}    Reference : Centroid\n')

    def wavefront(WF, wl, fields):
        if isinstance(WF, btd.lattice.Lattice):
            WF = -WF.array

        fig, ax = plt.subplots()
        WFax = ax.imshow(WF, cmap=plotting.customDivergingColormap(), extent=[-1., 1., -1., 1.])
        ax.set_facecolor('#2822bb')
        ax.set_title('Wavefront Map', fontsize=14)
        ax.set_xlabel('X-Pupil (Rel. Units)', fontsize=14)
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_ylabel('Y-Pupil (Rel. Units)', fontsize=14)
        ax.set_yticks([-1.0, 0.0, 1.0])
        cbar = fig.colorbar(WFax, label='waves')
        cbar.set_ticks(np.round(np.linspace(WF.min(), WF.max(), 11), decimals=4))
        fig.canvas.header_visible = False
        plt.show()

        print(f'{date.today().strftime("%m/%d/%Y")}')
        print(f'{float(wl * 1e6)} µm at {float(fields[0])}, {float(fields[1])} (deg)')
        RMS = WF.std()
        PV = np.abs(WF.max()-WF.min())
        print('Peak to valley =', np.round(PV, decimals=4), 'waves, RMS =', np.round(RMS, decimals=4), 'waves')


    def customDivergingColormap():
        nbOfBits = 512
        N = nbOfBits//7
        s = 1.2
        jet_colors = plt.cm.get_cmap('jet')
        lower_colors = jet_colors(np.linspace(0.10, 1/3, int(s*N)))
        upper_colors = jet_colors(np.linspace(2/3, 0.95, int(1.5*s*N)))
        middle_colors = jet_colors(np.linspace(1/3, 2/3, int(nbOfBits-2.5*s*N)))
        custom_colors = np.vstack((lower_colors, middle_colors, upper_colors))
        return mcolors.ListedColormap(custom_colors)

    def psf3D(PSF, pxSize, thresh=0):
        #PSF = np.log(PSF.array)
        #plotting.thresholding(PSF, thresh=thresh)
        N = len(PSF[0])
        mid = N//2
        x = np.linspace(-mid*pxSize, mid*pxSize, N)*1e6
        x, y = np.meshgrid(x, x)
        
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=PSF,
                                         colorscale='Spectral',
                                         reversescale=True,
                                         colorbar=dict(thickness=30,
                                                       tickvals=np.round(np.linspace(PSF.min(), PSF.max(), 10), decimals=1)
                                                      )
                                        )
                             ]
                       )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                          width=700,
                          height=500
                         )
        fig.show()
    
    def psf2D(PSF, pxSize, thresh=0):
        #PSF = np.log(PSF)
        #plotting.thresholding(PSF, thresh=thresh)
        N = len(PSF[0])
        mid = N//2
        pxSize *= 1e6
        
        fig, ax = plt.subplots()
        PSFax = ax.imshow(PSF, cmap='gray', extent=[-mid*pxSize,mid*pxSize,-mid*pxSize,mid*pxSize])
        ax.set_title(f'Log Huygens PSF', fontsize=14)
        ax.set_xlabel('µm', fontsize=14)
        ax.set_ylabel('µm', fontsize=14)
        cbar = fig.colorbar(PSFax)
        cbar.set_ticks(np.round(np.linspace(PSF.min(), PSF.max(), 10), decimals=1))
        fig.canvas.header_visible = False
        plt.show()

    def psfCrossSection(PSF, pxSize, axis='Y'):
        #PSF = np.sqrt(PSF)

        N = len(PSF[0])
        mid = N//2
        x = np.linspace(-mid*pxSize, mid*pxSize, N)*1e6
        
        fig, ax = plt.subplots()
        if axis == 'Y':
            CS = PSF[:, mid]
            ax.set_title(f'Sqrt of Cross Section of PSF at X={mid}', fontsize=14)
        elif axis == 'X':
            CS = PSF[mid, :]
            ax.set_title(f'Sqrt of Cross Section of PSF at Y={mid}', fontsize=14)
        ax.plot(x, CS)
        ax.set_xlabel('µm', fontsize=14)
        fig.canvas.header_visible = False
        plt.show()

    def thresholding(data, thresh=0):
        if thresh != 0:
            lim = thresh * np.max(data)
            data[data<lim] = 0
        return data
   
class layout:

    def sag2DY(optic, inDiam, outDiam, offset):
        N = 0.5e2
        if inDiam != 0:
            y1 = np.linspace(-outDiam, -inDiam, int(N/2))/2
            x1 = np.zeros(len(y1))
            z1 = (optic.surface.sag(x1, y1) + offset) 
            y2 = np.linspace(inDiam, outDiam, int(N/2))/2
            x2 = np.zeros(len(y2))
            z2 = (optic.surface.sag(x2, y2) + offset)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            y = np.linspace(-outDiam, outDiam, int(N))/2
            x = np.zeros(len(y))
            z = (optic.surface.sag(x, y) + offset)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df
        
    def sag2DX(optic, inDiam, outDiam, offset):
        N = 0.5e2
        if inDiam != 0:
            x1 = np.linspace(-outDiam, -inDiam, int(N/2))/2
            y1 = np.zeros(len(x1))
            z1 = (optic.surface.sag(x1, y1) + offset)
            x2 = np.linspace(inDiam, outDiam, int(N/2))/2
            y2 = np.zeros(len(x2))
            z2 = (optic.surface.sag(x2, y2) + offset)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            x = np.linspace(-outDiam, outDiam, int(N))/2
            y = np.zeros(len(x))
            z = (optic.surface.sag(x, y) + offset)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df 

    def sag2Dcircle(optic, inDiam, outDiam, offset):
        N = 0.5e2
        if inDiam != 0:
            th = np.linspace(0, 2*np.pi, int(N))
            x1 = outDiam/2 * np.cos(th)
            y1 = outDiam/2 * np.sin(th)
            z1 = (optic.surface.sag(x1, y1) + offset)
            x2 = inDiam/2 * np.cos(th)
            y2 = inDiam/2 * np.sin(th)
            z2 = (optic.surface.sag(x2, y2) + offset)
            df1 = pd.DataFrame({"x":x1, "y":y1, "z":z1})
            df2 = pd.DataFrame({"x":x2, "y":y2, "z":z2})
            return df1, df2
        else:
            th = np.linspace(0, 2*np.pi, int(N))
            x = outDiam/2 * np.cos(th)
            y = outDiam/2 * np.sin(th)
            z = (optic.surface.sag(x, y) + offset)
            df = pd.DataFrame({"x":x, "y":y, "z":z})
            return df

    def getMirrorLines(opt, inner, outter):
        if inner != 0:
            df1Y, df2Y = layout.sag2DY(opt, inner, outter, opt.coordSys.origin[-1])
            df1X, df2X = layout.sag2DX(opt, inner, outter, opt.coordSys.origin[-1])
            df1c, df2c = layout.sag2Dcircle(opt, inner, outter, opt.coordSys.origin[-1])
            return df1Y, df2Y, df1X, df2X, df1c, df2c
        else: 
            dfY = layout.sag2DY(opt, 0, outter, opt.coordSys.origin[-1])
            dfX = layout.sag2DX(opt, 0, outter, opt.coordSys.origin[-1])
            dfc = layout.sag2Dcircle(opt, 0, outter, opt.coordSys.origin[-1])
            return dfY, dfX, dfc

    def combineRays(rayIn, rayOut):
        Rin = rayIn.r[0]
        Rout = rayOut.r[0]
        return np.array([[Rin[0], Rout[0]], [Rin[1], Rout[1]], [Rin[2], Rout[2]]])

    def combineLine(Lend, Lbeg, offsetZ=0):
        return np.array([[Lend[0, 1], Lbeg[0, 1]], [Lend[1, 1],Lbeg[1, 1]], [Lend[2, 1], Lbeg[2, 1]+offsetZ]])
    
    def traceThrough(opt, rayIn):
        ray = rayIn.copy()
        opt.trace(ray, reverse=False)
        return ray, layout.combineRays(rayIn, ray)

    def drawLines(optic, WL, x=0, y=0, anglX=0, anglY=0):
        ray = btd.RayVector.fromStop(x=x, y=y,
                                     optic=optic,
                                     wavelength=WL,
                                     dirCos=np.array([anglX*np.pi/180, anglY*np.pi/180, 1.]),
                                     flux=1.)

        args = optic.items
        N = len(args)
        wholeRayTracing = [0] * N

        ray, line_n = layout.traceThrough(args[0], ray)
        wholeRayTracing[0] = line_n
        for i in range(1, N):
            opt = args[i]
            ray, line = layout.traceThrough(opt, ray)
            line_n = layout.combineLine(line_n, line, offsetZ=opt.coordSys.origin[-1])
            wholeRayTracing[i] = line_n
        return wholeRayTracing

    def getAllLines(optic):
        N = len(optic.items)
        allLines = [0]* N
        for i in range(N):
            opt = optic.items[i]
            allLines[i] = layout.getMirrorLines(opt, opt.inDiam, opt.outDiam)
        return chain(*allLines)

    def bunchOfRays(optic, WL, inDiam, outDiam, anglX, anglY, nbOfRays):
        if inDiam != 0:
            y = np.concatenate((np.linspace(-outDiam, -inDiam, int(nbOfRays/2)), np.linspace(inDiam, outDiam, int(nbOfRays/2)))) / 2
        else:
            y = np.linspace(-outDiam, outDiam, int(nbOfRays)) / 2
        rays = [0]*len(y)
        for i in range(len(y)):
            rays[i] = layout.drawLines(optic, WL=WL, x=0, y=y[i], anglX=anglX, anglY=anglY)
        return rays

    def visualized2D(nbOfRaysPerField, *args):
        colors = np.array([['indigo'], ['blue'], ['green'], ['yellow'], ['orange'], ['red']])
        c, i = 0, 0
        fig = px.line(width=1000, height=700)
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                fig.add_trace(px.line(arg, x='z', y='y', color_discrete_sequence=['black'],).data[0])
            else:
                for j in range(0, len(arg)): # start at 1 => no ray from inf
                    if c > len(colors):
                        c = 0
                    line = arg[j]
                    df = pd.DataFrame({"x":line[0], "y":line[1], "z":line[2]})
                    if (i < nbOfRaysPerField) or (nbOfRaysPerField == 0):
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[0]).data[0]) #colors[c]
                    elif ((i >= nbOfRaysPerField) and (i < 2*nbOfRaysPerField)) and (nbOfRaysPerField != 0):
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[2]).data[0]) #colors[c]
                    else:
                        fig.add_trace(px.line(df, x='z', y='y', color_discrete_sequence=colors[-1]).data[0]) #colors[c]
                c += 1
                i += 1
        
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          scene=dict(xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')),
                          paper_bgcolor='rgb(255,255,255)',
                          plot_bgcolor='rgb(255,255,255)')
        fig.show()

    def visualized3D(nbOfRaysPerField, *args):
        colors = np.array([['indigo'], ['blue'], ['green'], ['yellow'], ['orange'], ['red']])
        c, i = 0, 0
        fig = px.line_3d(width=1000, height=700)
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                fig.add_trace(px.line_3d(arg, x='x', y='y', z='z', color_discrete_sequence=['black'],).data[0])
            else:
                for j in range(0, len(arg)): # start at 1 => no ray from inf
                    if c > len(colors):
                        c = 0
                    line = arg[j]
                    df = pd.DataFrame({"x":line[0], "y":line[1], "z":line[2]})
                    if (i < nbOfRaysPerField) or (nbOfRaysPerField == 0):
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[0]).data[0]) #colors[c]
                    elif ((i >= nbOfRaysPerField) and (i < 2*nbOfRaysPerField)) and (nbOfRaysPerField != 0):
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[2]).data[0]) #colors[c]
                    else:
                        fig.add_trace(px.line_3d(df, x='x', y='y', z='z', 
                                                 color_discrete_sequence=colors[-1]).data[0]) #colors[c]
                c += 1
                i += 1
        camera_params = dict(up=dict(x=0,y=1,z=0),
                             center=dict(x=0,y=0,z=0),
                             eye=dict(x=-1.5,y=0,z=0.455))
    
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          scene=dict(zaxis=dict(title='z (meters)'),
                                     xaxis=dict(title='x (meters)'),
                                     yaxis=dict(title='y (meters)')),
                          scene_camera=camera_params)
        fig.show()