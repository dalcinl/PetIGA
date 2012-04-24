from igakit.nurbs import NURBS
import numpy as np
import sys

def WritePetIGAGeometry(geom,fname):
    def write_integer(fh, data):
        np.array(data, dtype='>i4').tofile(fh)
    def write_scalar(fh, data):
        np.array(data, dtype='>f8').tofile(fh)
    iga_id = 1211299
    vec_id = 1211214
    kind = 2
    dim  = geom.dim
    fh = open(fname,'wb')
    write_integer(fh,[iga_id, kind, dim])
    for i in range(geom.dim):
        periodic = 0
        p = geom.degree[i]
        U = geom.knots[i]
        m = len(U)
        write_integer(fh,[periodic,p,m])
        write_scalar(fh,U)
    if kind:
        Cw = geom.control#[...,range(dim)+[3]]
        Cw = np.rollaxis(Cw, -1).transpose().copy()
        write_integer(fh,[vec_id,Cw.size])
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

geom = NURBS(C,[U,V])
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

