#if !defined(PETIGA_H)
#define PETIGA_H

/*
#include "petscconf.h"
#undef  PETSC_STATIC_INLINE
#define PETSC_STATIC_INLINE static __inline
*/

#include <petsc.h>
#include <petscdmda.h>
PETSC_EXTERN_CXX_BEGIN

#ifndef PetscMalloc1
#define PetscMalloc1(m1,t1,r1) (PetscMalloc((m1)*sizeof(t1),(r1)))
#endif
 
/* ---------------------------------------------------------------- */

typedef struct _n_IGAAxis     *IGAAxis;
typedef struct _n_IGARule     *IGARule;
typedef struct _n_IGABasis    *IGABasis;
typedef struct _n_IGABoundary *IGABoundary;

typedef struct _n_IGAElement  *IGAElement;
typedef struct _n_IGAPoint    *IGAPoint;

/* ---------------------------------------------------------------- */

struct _n_IGAAxis {
  PetscInt   refct;
  /**/
  PetscBool  periodic; /* periodicity */
  PetscInt   p; /* polynomial order   */
  PetscInt   m; /* last knot index    */
  PetscReal *U; /* knot vector        */
};
extern PetscErrorCode IGAAxisCreate(IGAAxis *axis);
extern PetscErrorCode IGAAxisDestroy(IGAAxis *axis);
extern PetscErrorCode IGAAxisReference(IGAAxis axis);
extern PetscErrorCode IGAAxisCopy(IGAAxis base,IGAAxis axis);
extern PetscErrorCode IGAAxisDuplicate(IGAAxis base,IGAAxis *axis);
extern PetscErrorCode IGAAxisSetPeriodic(IGAAxis axis,PetscBool periodic);
extern PetscErrorCode IGAAxisGetPeriodic(IGAAxis axis,PetscBool *periodic);
extern PetscErrorCode IGAAxisSetOrder(IGAAxis axis,PetscInt p);
extern PetscErrorCode IGAAxisGetOrder(IGAAxis axis,PetscInt *p);
extern PetscErrorCode IGAAxisSetKnots(IGAAxis axis,PetscInt m,PetscReal U[]);
extern PetscErrorCode IGAAxisGetKnots(IGAAxis axis,PetscInt *m,PetscReal *U[]);
extern PetscErrorCode IGAAxisInitUniform(IGAAxis axis,PetscInt p,PetscInt C,
                                         PetscInt E,PetscReal Ui,PetscReal Uf);
extern PetscErrorCode IGAAxisCheck(IGAAxis axis);

struct _n_IGARule {
  PetscInt   refct;
  /**/
  PetscInt  nqp;      /* number of quadrature points */
  PetscReal *point;   /* [nqp] quadrature points  */
  PetscReal *weight;  /* [nqp] quadrature weights */
};
extern PetscErrorCode IGARuleCreate(IGARule *rule);
extern PetscErrorCode IGARuleDestroy(IGARule *rule);
extern PetscErrorCode IGARuleCopy(IGARule base,IGARule rule);
extern PetscErrorCode IGARuleReference(IGARule rule);
extern PetscErrorCode IGARuleDuplicate(IGARule base,IGARule *rule);
extern PetscErrorCode IGARuleInit(IGARule rule,PetscInt q);
extern PetscErrorCode IGARuleSetRule(IGARule rule,PetscInt q,const PetscReal x[],const PetscReal w[]);
extern PetscErrorCode IGARuleGetRule(IGARule rule,PetscInt *q,PetscReal *x[],PetscReal *w[]);

struct _n_IGABasis {
  PetscInt   refct;
  /**/
  PetscInt  nel;      /* number of elements */
  PetscInt  nqp;      /* number of quadrature points */
  PetscInt  nen;      /* number of local basis functions */
  PetscInt  p,d;      /* polynomial order, last derivative index */

  PetscReal *detJ;   /* [nel]              */
  PetscReal *weight; /* [nqp]              */
  PetscReal *point;  /* [nel][nqp]         */
  PetscReal *value;  /* [nel][nqp][nen][d] */
  /**/
  PetscInt  nnp;      /* number of global basis functions */
  PetscInt  *span;    /* [nel] span index                 */
  PetscInt  *offset;  /* [nel] basis offset               */
};
extern PetscErrorCode IGABasisCreate(IGABasis *basis);
extern PetscErrorCode IGABasisDestroy(IGABasis *basis);
extern PetscErrorCode IGABasisReference(IGABasis basis);
extern PetscErrorCode IGABasisInit(IGABasis basis,IGAAxis axis,IGARule rule,PetscInt d);

struct _n_IGABoundary {
  PetscInt   refct;
  /**/
  PetscInt    dof;
  PetscInt    nbc;
  PetscInt    *field;
  PetscScalar *value;
};
extern PetscErrorCode IGABoundaryCreate(IGABoundary *boundary);
extern PetscErrorCode IGABoundaryDestroy(IGABoundary *boundary);
extern PetscErrorCode IGABoundaryReference(IGABoundary boundary);
extern PetscErrorCode IGABoundaryInit(IGABoundary boundary,PetscInt dof);
extern PetscErrorCode IGABoundarySetValue(IGABoundary boundary,PetscInt field,PetscScalar value);

/* ---------------------------------------------------------------- */

typedef PetscErrorCode (*IGAUserSystem)    (IGAPoint point,PetscScalar *K,PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAUserFunction)  (IGAPoint point,const PetscScalar *U,PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAUserJacobian)  (IGAPoint point,const PetscScalar *U,PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAUserIFunction) (IGAPoint point,PetscReal dt,PetscReal a,
                                            PetscReal t,const PetscScalar *U,const PetscScalar *V,PetscScalar *F,
                                            void *ctx);
typedef PetscErrorCode (*IGAUserIJacobian) (IGAPoint point,PetscReal dt,PetscReal a,
                                            PetscReal t,const PetscScalar *U,const PetscScalar *V,PetscScalar *J,
                                            void *ctx);

typedef struct _IGAUserOps *IGAUserOps;
struct _IGAUserOps {
  /**/
  IGAUserSystem     System;
  void              *SysCtx;
  /**/
  IGAUserFunction   Function;
  void              *FunCtx;
  IGAUserJacobian   Jacobian;
  void              *JacCtx;
  /**/
  IGAUserIFunction  IFunction;
  void              *IFunCtx;
  IGAUserIJacobian  IJacobian;
  void              *IJacCtx;
};

/* ---------------------------------------------------------------- */

typedef struct _p_IGA *IGA;

typedef struct _IGAOps *IGAOps;
struct _IGAOps {
  PetscErrorCode (*create)(IGA);
  PetscErrorCode (*destroy)(IGA);
  PetscErrorCode (*view)(IGA);
  PetscErrorCode (*setup)(IGA);
  PetscErrorCode (*setfromoptions)(IGA);
};

struct _p_IGA {
  PETSCHEADER(struct _IGAOps);
  IGAUserOps userops;
  VecType    vectype;
  MatType    mattype;

  PetscBool setup;
  PetscInt  dim;
  PetscInt  dof;
  
  IGAAxis  axis[3];
  IGARule  rule[3];
  IGABasis basis[3];
  IGABoundary boundary[3][2];
  
  DM dm_geom;
  PetscBool rational;
  PetscBool geometry;
  DM dm_dof;

  IGAElement iterator;
};

extern PetscClassId IGA_CLASSID;

extern PetscErrorCode IGAInitializePackage(const char path[]);

extern PetscErrorCode IGACreate(MPI_Comm comm,IGA *iga);
extern PetscErrorCode IGADestroy(IGA *iga);
extern PetscErrorCode IGAView(IGA iga,PetscViewer viewer);
extern PetscErrorCode IGASetFromOptions(IGA iga);

extern PetscErrorCode IGASetDim(IGA iga,PetscInt dim);
extern PetscErrorCode IGAGetDim(IGA iga,PetscInt *dim);
extern PetscErrorCode IGASetDof(IGA iga,PetscInt dof);
extern PetscErrorCode IGAGetDof(IGA iga,PetscInt *dof);

extern PetscErrorCode IGAGetAxis(IGA iga,PetscInt i,IGAAxis *axis);
extern PetscErrorCode IGAGetRule(IGA iga,PetscInt i,IGARule *rule);
extern PetscErrorCode IGAGetBoundary(IGA iga,PetscInt i,PetscInt side,IGABoundary *boundary);

extern PetscErrorCode IGASetUp(IGA iga);

extern PetscErrorCode IGAGetComm(IGA iga,MPI_Comm *comm);
extern PetscErrorCode IGAGetDofDM(IGA iga,DM *dm_dof);
extern PetscErrorCode IGAGetGeomDM(IGA iga,DM *dm_geom);

extern PetscErrorCode IGASetVecType(IGA iga,const VecType vectype);
extern PetscErrorCode IGASetMatType(IGA iga,const MatType mattype);

extern PetscErrorCode IGACreateVec(IGA iga,Vec *vec);
extern PetscErrorCode IGACreateMat(IGA iga,Mat *mat);

extern PetscErrorCode IGAGetLocalVec(IGA iga,Vec *gvec);
extern PetscErrorCode IGARestoreLocalVec(IGA iga,Vec *lvec);
extern PetscErrorCode IGAGetGlobalVec(IGA iga,Vec *gvec);
extern PetscErrorCode IGARestoreGlobalVec(IGA iga,Vec *gvec);
extern PetscErrorCode IGAGlobalToLocal(IGA iga,Vec gvec,Vec lvec);
extern PetscErrorCode IGALocalToGlobal(IGA iga,Vec lvec,Vec gvec,InsertMode addv);

extern PetscErrorCode IGAGetElement(IGA iga,IGAElement *element);

extern PetscErrorCode IGASetUserSystem   (IGA iga,IGAUserSystem    System,   void *SysCtx);
extern PetscErrorCode IGASetUserFunction (IGA iga,IGAUserFunction  Function, void *FunCtx);
extern PetscErrorCode IGASetUserJacobian (IGA iga,IGAUserJacobian  Jacobian, void *JacCtx);
extern PetscErrorCode IGASetUserIFunction(IGA iga,IGAUserIFunction IFunction,void *FunCtx);
extern PetscErrorCode IGASetUserIJacobian(IGA iga,IGAUserIJacobian IJacobian,void *JacCtx);

/* ---------------------------------------------------------------- */

struct _n_IGAElement {
  PetscInt   refct;
  /**/
  PetscInt count;
  PetscInt index;
  PetscInt range[3][2];
  PetscInt ID[3];

  PetscInt nqp;
  PetscInt nen;
  PetscInt dof;
  PetscInt dim;

  PetscInt start[3];
  PetscInt width[3];
  PetscInt *mapping;   /*   [nen] */

  PetscInt     nfix;
  PetscInt    *ifix;
  PetscScalar *vfix;
  PetscScalar *xfix;
  
  PetscReal *point;    /*   [nqp][dim]                */
  PetscReal *weight;   /*   [nqp]                     */
  PetscReal *detJ;     /*   [nqp]                     */
  PetscReal *jacobian; /*   [nqp][dim][dim]           */
  PetscReal *shape[4]; /*0: [nqp][nen]                */
                       /*1: [nqp][nen][dim]           */
                       /*2: [nqp][nen][dim][dim]      */
                       /*3: [nqp][nen][dim][dim][dim] */
  
  IGA      parent;
  IGAPoint iterator;

  PetscInt    nvec;
  PetscScalar *wvec[8];
  PetscInt    nmat;
  PetscScalar *wmat[4];

};
extern PetscErrorCode IGAElementCreate(IGAElement *element);
extern PetscErrorCode IGAElementDestroy(IGAElement *element);
extern PetscErrorCode IGAElementSetUp(IGAElement element);

extern PetscErrorCode IGAElementBegin(IGAElement element);
extern PetscBool      IGAElementNext(IGAElement element);

extern PetscErrorCode IGAElementBuildFix(IGAElement element);
extern PetscErrorCode IGAElementBuildMap(IGAElement element);
extern PetscErrorCode IGAElementBuildQuadrature(IGAElement element);
extern PetscErrorCode IGAElementBuildShapeFuns(IGAElement element);

extern PetscErrorCode IGAElementGetIndex(IGAElement element,PetscInt *index);
extern PetscErrorCode IGAElementGetInfo(IGAElement element,PetscInt *nqp,PetscInt *nen,PetscInt *dof,PetscInt *dim);
extern PetscErrorCode IGAElementGetPoint(IGAElement element,IGAPoint *point);

extern PetscErrorCode IGAElementGetWorkVec(IGAElement element,PetscScalar *V[]);
extern PetscErrorCode IGAElementGetWorkMat(IGAElement element,PetscScalar *M[]);

extern PetscErrorCode IGAElementGetValues(IGAElement element,const PetscScalar U[],PetscScalar u[]);

extern PetscErrorCode IGAElementFixValues(IGAElement element,PetscScalar U[]);
extern PetscErrorCode IGAElementFixFunction(IGAElement element,PetscScalar F[]);
extern PetscErrorCode IGAElementFixJacobian(IGAElement element,PetscScalar J[]);
extern PetscErrorCode IGAElementFixSystem(IGAElement element,PetscScalar K[],PetscScalar F[]);

extern PetscErrorCode IGAElementAssembleVec(IGAElement element,PetscScalar F[],Vec vec);
extern PetscErrorCode IGAElementAssembleMat(IGAElement element,PetscScalar K[],Mat mat);

/* ---------------------------------------------------------------- */

struct _n_IGAPoint {
  PetscInt refct;
  /**/
  PetscInt count;
  PetscInt index;

  PetscInt nen;
  PetscInt dof;
  PetscInt dim;

  PetscReal *point;    /*   [dim] */
  PetscReal weight;    /*   []    */
  PetscReal detJ;      /*   []    */
  PetscReal *jacobian; /*   [dim][dim] */
  PetscReal *shape[4]; /*0: [nen]      */
                       /*1: [nen][dim] */
                       /*2: [nen][dim][dim] */
                       /*3: [nen][dim][dim][dim] */
  
  IGAElement  parent;

  PetscInt    nvec;
  PetscScalar *wvec[8];
  PetscInt    nmat;
  PetscScalar *wmat[4];
};
extern PetscErrorCode IGAPointCreate(IGAPoint *point);
extern PetscErrorCode IGAPointDestroy(IGAPoint *point);
extern PetscErrorCode IGAPointSetUp(IGAPoint point);

extern PetscErrorCode IGAPointBegin(IGAPoint point);
extern PetscBool      IGAPointNext(IGAPoint point);

extern PetscErrorCode IGAPointGetIndex(IGAPoint point,PetscInt *index);
extern PetscErrorCode IGAPointGetInfo(IGAPoint point,PetscInt *nen,PetscInt *dof,PetscInt *dim);

extern PetscErrorCode IGAPointGetWorkVec(IGAPoint point,PetscScalar *V[]);
extern PetscErrorCode IGAPointGetWorkMat(IGAPoint point,PetscScalar *M[]);

extern PetscErrorCode IGAPointInterpolate(IGAPoint point,PetscInt ider,const PetscScalar U[],PetscScalar u[]);
extern PetscErrorCode IGAPointAddVec(IGAPoint point,const PetscScalar f[],PetscScalar F[]);
extern PetscErrorCode IGAPointAddMat(IGAPoint point,const PetscScalar k[],PetscScalar K[]);

/* ---------------------------------------------------------------- */

extern PetscErrorCode IGACreateKSP(IGA iga,KSP *ksp);
extern PetscErrorCode IGAFormSystem(IGA iga,Mat A,Vec B,
                                    IGAUserSystem System,void *ctx);

extern PetscErrorCode IGACreateSNES(IGA iga,SNES *snes);
extern PetscErrorCode IGAFormFunction(IGA iga,Vec U,Vec F,
                                      IGAUserFunction Function,void *ctx);
extern PetscErrorCode IGAFormJacobian(IGA iga,Vec U,Mat J,
                                      IGAUserJacobian Jacobian,void *ctx);

extern PetscErrorCode IGACreateTS(IGA iga,TS *ts);
extern PetscErrorCode IGAFormIFunction(IGA iga,PetscReal dt,PetscReal a,
                                       PetscReal t,Vec U,Vec V,Vec F,
                                       IGAUserIFunction IFunction,void *ctx);
extern PetscErrorCode IGAFormIJacobian(IGA iga,PetscReal dt,PetscReal a,
                                       PetscReal t,Vec U,Vec V,Mat J,
                                       IGAUserIJacobian IJacobian,void *ctx);

/* ---------------------------------------------------------------- */

PETSC_EXTERN_CXX_END
#endif/*PETIGA_H*/
