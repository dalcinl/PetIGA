import sys, os
VERBOSE = True

for dim in [1,2,3]:
    iga_dim = "-iga_dim %d" % dim
    for dof in [1,2,3]:
        iga_dof = "-iga_dof %d" % dof
        for mat_type in ['aij', 'baij', 'sbaij', 'is']:
            iga_mat_type = "-iga_mat_type %s" % mat_type
            for periodic in [0,1]:
                iga_periodic = "-iga_periodic %d" % periodic
                for elements in [4,6,8]:#[1,2,3]:
                    iga_elements = "-iga_elements %d" % elements
                    for degree in range(1, 4):
                        iga_degree = "-iga_degree %d" % degree
                        for continuity in range(0, degree):
                            iga_continuity = "-iga_continuity %d" % continuity
                            for mpiexec in ["mpiexec -n %d" % 1,
                                            "mpiexec -n %d" % 2**dim]:
                                #
                                IGACreate = "./IGACreate"
                                options = [
                                    iga_dim,
                                    iga_dof,
                                    iga_mat_type,
                                    iga_periodic,
                                    iga_elements,
                                    iga_degree,
                                    iga_continuity,
                                    "-malloc_debug",
                                    "-malloc_dump",
                                    ]
                                if mat_type == 'is':
                                    options.append('-pc_type jacobi')
                                #
                                cmd = " ".join([mpiexec, IGACreate] + options)
                                if VERBOSE: print (cmd)
                                ret = os.system(cmd)
                                if ret:
                                    print ("Error!")
                                    sys.exit(ret)
