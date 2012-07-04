from igakit.nurbs import NURBS
import numpy as np
import sys

def WritePetIGAGeometry(geom,fname):
    iga_id = 1211299
    vec_id = 1211214
    def write_integer(fh, data):
        np.array(data, dtype='>i4').tofile(fh)
    def write_scalar(fh, data):
        np.array(data, dtype='>f8').tofile(fh)
    fh = open(fname,'wb')
    descr = +1
    dim = geom.dim
    write_integer(fh,[iga_id,descr,dim])
    for i in range(geom.dim):
        p = geom.degree[i]
        U = geom.knots[i]
        m = len(U)
        write_integer(fh,[p,m])
        write_scalar(fh,U)
    if descr:
        nsd = 3
        Cw = geom.control#[...,range(dim)+[3]]
        Cw = np.rollaxis(Cw, -1).transpose().copy()
        write_integer(fh,[nsd,vec_id,Cw.size])
        write_scalar(fh,Cw)
        fh.flush()
        fh.close()


N = 10
p = 2
if len(sys.argv) == 3:
    N = int(sys.argv[1])
    p = int(sys.argv[2])

U = [0,0,     1,1]
V = [0,0,0, 1,1,1]
C = np.zeros((2,3,4))
val = np.sqrt(2)*0.5
C[0,0,:] = [  0,-100,  0,1]
C[1,0,:] = [100,-100,  0,1]
C[0,1,:] = [  0,-100,100,1]
C[1,1,:] = [100,-100,100,1]
C[0,2,:] = [  0,   0,100,1]
C[1,2,:] = [100,   0,100,1]
C[:,1,:] *= val

geom = NURBS([U,V],C)
geom.elevate(max(p-1,0),max(p-2,0))

h = 1./N
insert = np.linspace(h,1.-h,N-1)
geom = geom.refine(insert,insert)

WritePetIGAGeometry(geom,"shell.dat")

if False:
    from igakit.plot import plt
    plt.figure()
    plt.cpoint(geom)
    plt.cwire(geom)
    plt.kwire(geom)
    plt.surface(geom)
    plt.show()

