#if !defined(PETIGAPROBE_H)
#define PETIGAPROBE_H

#include "petiga.h"

typedef struct _n_IGAProbe *IGAProbe;

struct _n_IGAProbe {
  PetscInt refct;
  /**/
  IGA       iga;
  Vec       gvec;
  Vec       lvec;
  /**/
  PetscInt  order;
  PetscBool collective;
  PetscBool offprocess;
  /**/
  PetscInt  dim;
  PetscInt  dof;
  PetscInt  p[3];
  PetscReal *U[3];
  PetscInt  n[3];
  PetscInt  s[3];
  PetscInt  w[3];
  const PetscReal   *arrayW;
  const PetscReal   *arrayX;
  const PetscScalar *arrayA;
  /**/
  PetscInt    nen;
  PetscInt    *map;
  PetscReal   *W;
  PetscReal   *X;
  PetscScalar *A;
  /**/
  PetscReal point[3];
  PetscInt  ID[3];
  PetscReal *BD[3];
  PetscReal *basis[5]; /*0: [nen] */
                       /*1: [nen][dim] */
                       /*2: [nen][dim][dim] */
                       /*3: [nen][dim][dim][dim] */
                       /*4: [nen][dim][dim][dim][dim] */
  PetscReal *detX;
  PetscReal *mapX[5]; /*0: [nsd] */
                      /*1: [nsd][dim] */
                      /*2: [nsd][dim][dim] */
                      /*3: [nsd][dim][dim][dim] */
                      /*4: [nsd][dim][dim][dim][dim] */
  PetscReal *mapU[5]; /*0: [dim] */
                      /*1: [dim][nsd] */
                      /*2: [dim][nsd][nsd] */
                      /*3: [dim][nsd][nsd][nsd] */
                      /*4: [dim][nsd][nsd][nsd][nsd] */
  PetscReal *shape[5]; /*0: [nen] */
                       /*1: [nen][nsd] */
                       /*2: [nen][nsd][nsd] */
                       /*3: [nen][nsd][nsd][nsd] */
                       /*4: [nen][nsd][nsd][nsd][nsd] */
};

PETSC_EXTERN PetscErrorCode IGAProbeCreate(IGA iga,Vec A,IGAProbe *prb);
PETSC_EXTERN PetscErrorCode IGAProbeDestroy(IGAProbe *prb);
PETSC_EXTERN PetscErrorCode IGAProbeReference(IGAProbe prb);

PETSC_EXTERN PetscErrorCode IGAProbeSetOrder(IGAProbe prb,PetscInt order);
PETSC_EXTERN PetscErrorCode IGAProbeSetCollective(IGAProbe prb,PetscBool collective);

PETSC_EXTERN PetscErrorCode IGAProbeSetVec(IGAProbe prb,Vec A);
PETSC_EXTERN PetscErrorCode IGAProbeSetPoint(IGAProbe prb,const PetscReal u[]);

PETSC_EXTERN PetscErrorCode IGAProbeGeomMap  (IGAProbe prb,PetscReal x[]);
PETSC_EXTERN PetscErrorCode IGAProbeEvaluate (IGAProbe prb,PetscInt der,PetscScalar A[]);
PETSC_EXTERN PetscErrorCode IGAProbeFormValue(IGAProbe prb,PetscScalar A[]);
PETSC_EXTERN PetscErrorCode IGAProbeFormGrad (IGAProbe prb,PetscScalar A[]);
PETSC_EXTERN PetscErrorCode IGAProbeFormHess (IGAProbe prb,PetscScalar A[]);
PETSC_EXTERN PetscErrorCode IGAProbeFormDer3 (IGAProbe prb,PetscScalar A[]);
PETSC_EXTERN PetscErrorCode IGAProbeFormDer4 (IGAProbe prb,PetscScalar A[]);

#endif/*PETIGAPROBE_H*/
