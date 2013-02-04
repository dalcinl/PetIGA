import numpy as np
from subprocess import Popen,PIPE
import re,time

def ParseOutput(lines):
    """
    Extracts L2 error, assembly and solve time, and number of
    iterations from the lines of a PETSC log file.
    """
    err = 0.
    ta = 0.
    p2 = r"L2\serror\s=\s(.*)"
    p3 = r"IGAFormSystem(.*)"
    for line in lines:
        m = re.search(p2,line)
        if m:
            err = float(m.group(1))
        m = re.search(p3,line)
        if m:
            ta = float(m.group(1).split()[2])
    return err,ta

t0 = time.time()
for dim in [1,2,3]:
    for p in [1,2,3,4]:
        h = []
        e = []
        for N in [4,6,8,10,12,14]:
            h.append(1./N)
            cmd = "./ConvTest -iga_dim %d -iga_degree %d -iga_elements %d -log_summary -pc_type lu" % (dim,p,N)
            op = Popen(cmd,shell=True,stdout=PIPE)
            output,errors = op.communicate()
            if errors == None:
                E,t = ParseOutput(output.split('\n'))
                e.append(E)
        r,junk = np.polyfit(np.log10(h),np.log10(e),1)
        r = int(round(r))
        assert p+1==r
dt = time.time()-t0
assert abs(dt-143.299206018)/143.299206018 < 0.1
print "time %.3f (%.3f)" % (dt,143.299206018)
