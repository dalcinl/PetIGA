"""
PetIGA convergence test
"""
from __future__ import print_function
import sys, argparse
from subprocess import Popen, PIPE

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', type=int, nargs='+', dest='dimension', default=[1,2])
parser.add_argument('-n', type=int, nargs='+', dest='element', default=[48,64])
parser.add_argument('-p', type=int, nargs='+', dest='degree', default=[1,2,3,4])
parser.add_argument('-k', type=int, nargs='+', dest='continuity', default=None)
parser.add_argument('-collocation', action='store_true', dest='collocation')
parser.add_argument('args', nargs=argparse.REMAINDER, help='extra args for program')

options = parser.parse_args(sys.argv[1:])

try:
    unichr
except NameError:
    unichr = chr
CHECK = unichr(0x2705)
CROSS = unichr(0x274C)

def parseoutput(output):
    a = b = None
    for line in output.split(b'\n'):
        line = line.strip()
        if not line: continue
        bits = line.split()
        if bits[0] == b'Error:':
            a, b = float(bits[3]), float(bits[6])
        else:
            print(line)
    return a, b

def generate(opts):
    for d in sorted(set(opts.dimension)):
        if d < 1: continue
        if d > 3: continue
        for p in sorted(set(opts.degree)):
            if p < 1: continue
            if p > 9: continue
            if opts.collocation:
                if p < 2: continue
                continuity = [p-1]
            elif options.continuity:
                continuity = []
                for k in options.continuity:
                    if k < 0: k = p+k
                    if k < 0: continue
                    if k > p-1: continue
                    continuity.append(k)
            else:
                continuity = range(p)
            for k in sorted(set(continuity)):
                yield d, p, k

def computerate(h, e):
    try:
        import numpy as np
        r, _ = np.polyfit(np.log10(h), np.log10(e), 1)
    except ImportError:
        from math import log10
        de = log10(e[-2]/e[-1])
        dh = log10(h[-2]/h[-1])
        r = de/dh
    return r

def checkrate(expected, actual, tol=0.075):
    return expected - actual < tol

ok = True
for d, p, k in generate(options):
    N = sorted(set(options.element[:]))
    if len(N) < 1:
        N.insert([64,32,16][dim-1])
    if len(N) < 2:
        N.insert(0, N[0]-N[0]//4)
    h, errorL2, errorH1 = [], [], []
    for n in N:
        h.append(1.0/n)
        command = "./ConvTest -d %d -n %d -p %d -k %d" % (d, n, p, k)
        if options.collocation: command += ' -iga_collocation'
        if options.args: command = ' '.join([command] + options.args)
        pipe = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = pipe.communicate()
        assert not stderr
        L2, H1 = parseoutput(stdout)
        errorL2.append(L2)
        errorH1.append(H1)
    rL2 = computerate(h, errorL2)
    rH1 = computerate(h, errorH1)
    if options.collocation:
        q = p-1 if p % 2 else p
        mL2 = CHECK if checkrate(q, rL2) else CROSS
        mH1 = CHECK if checkrate(q, rH1) else CROSS
    else:
        mL2 = CHECK if checkrate(p+1, rL2) else CROSS
        mH1 = CHECK if checkrate(p,   rH1) else CROSS
    if CROSS in [mL2, mH1]: ok = False

    print(u"d={:d} n={:s} p={:d} k={:d}".format(d, repr(N).replace(" ",""), p, k),
          u"Rates: L2={:.2f}{:s} H1={:.2f}{:s}".format(rL2, mL2, rH1, mH1),
          u"Errors: L2={:.2e} H1={:.2e}".format(L2, H1),
          sep=' - ')

sys.exit(0 if ok else 1)
