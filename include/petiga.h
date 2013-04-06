#if !defined(PETIGA_H)
#define PETIGA_H

/*
#include <petscconf.h>
#undef  PETSC_STATIC_INLINE
#define PETSC_STATIC_INLINE static __inline
*/

#include <petsc.h>
#if PETSC_VERSION_(3,2,0)
#include <private/petscimpl.h>
#else
#include <petsc-private/petscimpl.h>
#endif
#include <petscts2.h>

#if PETSC_VERSION_(3,2,0)
PETSC_EXTERN_CXX_BEGIN
#define PETSC_EXTERN extern
#endif

typedef ISLocalToGlobalMapping LGMap;
#define LGMap LGMap

/* ---------------------------------------------------------------- */

typedef struct _n_IGAAxis     *IGAAxis;
typedef struct _n_IGABoundary *IGABoundary;
typedef struct _n_IGARule     *IGARule;
typedef struct _n_IGABasis    *IGABasis;
typedef struct _n_IGAElement  *IGAElement;
typedef struct _n_IGAPoint    *IGAPoint;

/* ---------------------------------------------------------------- */

struct _n_IGAAxis {
  PetscInt refct;
  /**/
  PetscInt   p; /* polynomial order    */
  PetscInt   m; /* last knot index     */
  PetscReal *U; /* knot vector         */
  /**/
  PetscBool  periodic; /* periodicity  */
  PetscInt   nnp,nel;  /* spans, bases */
  PetscInt   *span;    /* span indices */
};
PETSC_EXTERN PetscErrorCode IGAAxisCreate(IGAAxis *axis);
PETSC_EXTERN PetscErrorCode IGAAxisDestroy(IGAAxis *axis);
PETSC_EXTERN PetscErrorCode IGAAxisReset(IGAAxis axis);
PETSC_EXTERN PetscErrorCode IGAAxisReference(IGAAxis axis);
PETSC_EXTERN PetscErrorCode IGAAxisCopy(IGAAxis base,IGAAxis axis);
PETSC_EXTERN PetscErrorCode IGAAxisDuplicate(IGAAxis base,IGAAxis *axis);
PETSC_EXTERN PetscErrorCode IGAAxisSetPeriodic(IGAAxis axis,PetscBool periodic);
PETSC_EXTERN PetscErrorCode IGAAxisGetPeriodic(IGAAxis axis,PetscBool *periodic);
PETSC_EXTERN PetscErrorCode IGAAxisSetDegree(IGAAxis axis,PetscInt p);
PETSC_EXTERN PetscErrorCode IGAAxisGetDegree(IGAAxis axis,PetscInt *p);
PETSC_EXTERN PetscErrorCode IGAAxisSetKnots(IGAAxis axis,PetscInt m,PetscReal U[]);
PETSC_EXTERN PetscErrorCode IGAAxisGetKnots(IGAAxis axis,PetscInt *m,PetscReal *U[]);
PETSC_EXTERN PetscErrorCode IGAAxisGetLimits(IGAAxis axis,PetscReal *Ui,PetscReal *Uf);
PETSC_EXTERN PetscErrorCode IGAAxisGetSizes(IGAAxis axis,PetscInt *nel,PetscInt *nnp);
PETSC_EXTERN PetscErrorCode IGAAxisInitBreaks(IGAAxis axis,PetscInt nu,const PetscReal u[],PetscInt C);
PETSC_EXTERN PetscErrorCode IGAAxisInitUniform(IGAAxis axis,PetscInt N,PetscReal Ui,PetscReal Uf,PetscInt C);
PETSC_EXTERN PetscErrorCode IGAAxisSetUp(IGAAxis axis);

struct _n_IGARule {
  PetscInt refct;
  /**/
  PetscInt  nqp;      /* number of quadrature points */
  PetscReal *point;   /* [nqp] quadrature points  */
  PetscReal *weight;  /* [nqp] quadrature weights */
};
PETSC_EXTERN PetscErrorCode IGARuleCreate(IGARule *rule);
PETSC_EXTERN PetscErrorCode IGARuleDestroy(IGARule *rule);
PETSC_EXTERN PetscErrorCode IGARuleReset(IGARule rule);
PETSC_EXTERN PetscErrorCode IGARuleReference(IGARule rule);
PETSC_EXTERN PetscErrorCode IGARuleCopy(IGARule base,IGARule rule);
PETSC_EXTERN PetscErrorCode IGARuleDuplicate(IGARule base,IGARule *rule);
PETSC_EXTERN PetscErrorCode IGARuleInit(IGARule rule,PetscInt q);
PETSC_EXTERN PetscErrorCode IGARuleSetRule(IGARule rule,PetscInt q,const PetscReal x[],const PetscReal w[]);
PETSC_EXTERN PetscErrorCode IGARuleGetRule(IGARule rule,PetscInt *q,PetscReal *x[],PetscReal *w[]);

typedef enum {
  IGA_BASIS_BSPLINE=0,
  IGA_BASIS_BERNSTEIN,
  IGA_BASIS_LAGRANGE
} IGABasisType;

PETSC_EXTERN const char *const IGABasisTypes[];

struct _n_IGABasis {
  PetscInt refct;
  /**/
  IGABasisType type;  /* basis type */
  /**/
  PetscInt  nel;      /* number of elements */
  PetscInt  nqp;      /* number of quadrature points */
  PetscInt  nen;      /* number of local basis functions */
  PetscInt  p,d;      /* polynomial order, last derivative index */

  PetscInt  *offset;  /* [nel] basis offset   */
  PetscReal *detJ;    /* [nel]                */
  PetscReal *weight;  /* [nqp]                */
  PetscReal *point;   /* [nel][nqp]           */
  PetscReal *value;   /* [nel][nqp][nen][d+1] */

  PetscInt   bnd_offset[2];
  PetscReal  bnd_detJ[2];
  PetscReal  bnd_weight[2];
  PetscReal  bnd_point[2];
  PetscReal *bnd_value[2]; /* [nen][d+1] */
};
PETSC_EXTERN PetscErrorCode IGABasisCreate(IGABasis *basis);
PETSC_EXTERN PetscErrorCode IGABasisDestroy(IGABasis *basis);
PETSC_EXTERN PetscErrorCode IGABasisReset(IGABasis basis);
PETSC_EXTERN PetscErrorCode IGABasisReference(IGABasis basis);
PETSC_EXTERN PetscErrorCode IGABasisSetType(IGABasis basis,IGABasisType type);
PETSC_EXTERN PetscErrorCode IGABasisInitQuadrature (IGABasis basis,IGAAxis axis,IGARule rule,PetscInt order);
PETSC_EXTERN PetscErrorCode IGABasisInitCollocation(IGABasis basis,IGAAxis axis,PetscInt order);

/* ---------------------------------------------------------------- */

typedef PetscErrorCode (*IGAUserScalar)    (IGAPoint point,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx);
typedef PetscErrorCode (*IGAUserSystem)    (IGAPoint point,PetscScalar *K,PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAUserFunction)  (IGAPoint point,const PetscScalar *U,PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAUserJacobian)  (IGAPoint point,const PetscScalar *U,PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAUserIFunction) (IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAUserIJacobian) (IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAUserIFunction2)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *A,
                                            PetscReal v,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAUserIJacobian2)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *A,
                                            PetscReal v,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAUserIEFunction)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscReal t0,const PetscScalar *U0,
                                            PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAUserIEJacobian)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscReal t0,const PetscScalar *U0,
                                            PetscScalar *J,void *ctx);

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
  IGAUserIFunction2 IFunction2;
  void              *IFunCtx;
  IGAUserIJacobian  IJacobian;
  IGAUserIJacobian2 IJacobian2;
  void              *IJacCtx;
  /**/
  IGAUserIEFunction IEFunction;
  void              *IEFunCtx;
  IGAUserIEJacobian IEJacobian;
  void              *IEJacCtx;
};

/* ---------------------------------------------------------------- */

struct _n_IGABoundary {
  PetscInt refct;
  /**/
  PetscInt    dof;
  /**/
  PetscInt    count;
  PetscInt    *field;
  PetscScalar *value;
  /**/
  PetscInt    nload;
  PetscInt    *iload;
  PetscScalar *vload;
  /**/
  IGAUserOps userops;
};
PETSC_EXTERN PetscErrorCode IGABoundaryCreate(IGABoundary *boundary);
PETSC_EXTERN PetscErrorCode IGABoundaryDestroy(IGABoundary *boundary);
PETSC_EXTERN PetscErrorCode IGABoundaryReset(IGABoundary boundary);
PETSC_EXTERN PetscErrorCode IGABoundaryReference(IGABoundary boundary);
PETSC_EXTERN PetscErrorCode IGABoundaryInit(IGABoundary boundary,PetscInt dof);
PETSC_EXTERN PetscErrorCode IGABoundaryClear(IGABoundary boundary);
PETSC_EXTERN PetscErrorCode IGABoundarySetValue(IGABoundary boundary,PetscInt field,PetscScalar value);
PETSC_EXTERN PetscErrorCode IGABoundarySetLoad (IGABoundary boundary,PetscInt field,PetscScalar value);

PETSC_EXTERN PetscErrorCode IGABoundarySetUserSystem    (IGABoundary boundary,IGAUserSystem     System,    void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserFunction  (IGABoundary boundary,IGAUserFunction   Function,  void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserJacobian  (IGABoundary boundary,IGAUserJacobian   Jacobian,  void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserIFunction (IGABoundary boundary,IGAUserIFunction  IFunction, void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserIJacobian (IGABoundary boundary,IGAUserIJacobian  IJacobian, void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserIFunction2(IGABoundary boundary,IGAUserIFunction2 IFunction, void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserIJacobian2(IGABoundary boundary,IGAUserIJacobian2 IJacobian, void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserIEFunction(IGABoundary boundary,IGAUserIEFunction IEFunction,void *ctx);
PETSC_EXTERN PetscErrorCode IGABoundarySetUserIEJacobian(IGABoundary boundary,IGAUserIEJacobian IEJacobian,void *ctx);


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
  PetscBool  setup;
  PetscInt   setupstage;

  IGAUserOps userops;
  VecType    vectype;
  MatType    mattype;
  char       **fieldname;

  PetscInt  dim;   /* parametric dimension of the function space*/
  PetscInt  dof;   /* number of degrees of freedom per node */
  PetscInt  order; /* maximum derivative order */

  IGAAxis     axis[3];
  IGARule     rule[3];
  IGABoundary boundary[3][2];

  IGABasis    basis[3];
  IGAElement  iterator;

  PetscInt    geometry;
  PetscBool   rational;
  PetscInt    property;
  PetscReal   *geometryX;
  PetscReal   *rationalW;
  PetscScalar *propertyA;

  PetscInt  proc_sizes[3];
  PetscInt  proc_ranks[3];

  PetscInt  elem_sizes[3];
  PetscInt  elem_start[3];
  PetscInt  elem_width[3];
  DM        elem_dm;

  PetscInt  geom_sizes[3];
  PetscInt  geom_lstart[3];
  PetscInt  geom_lwidth[3];
  PetscInt  geom_gstart[3];
  PetscInt  geom_gwidth[3];
  DM        geom_dm;

  PetscInt  node_sizes[3];
  PetscInt  node_lstart[3];
  PetscInt  node_lwidth[3];
  PetscInt  node_gstart[3];
  PetscInt  node_gwidth[3];
  DM        node_dm;

  AO         ao,aob;
  LGMap      lgmap,lgmapb;
  VecScatter g2l,l2g;
  PetscInt   nwork;
  Vec        vwork[16];
  Vec        natural;
  VecScatter n2g,g2n;

  /* stuff added for collocation */
  PetscBool   collocation;
  IGABasis    node_basis[3];
  IGAElement  node_iterator;
};

PETSC_EXTERN PetscClassId IGA_CLASSID;
#define IGA_FILE_CLASSID 1211299

PETSC_EXTERN PetscErrorCode IGAInitializePackage(const char path[]);
PETSC_EXTERN PetscErrorCode IGAFinalizePackage(void);
PETSC_EXTERN PetscErrorCode IGARegisterAll(const char path[]);

PETSC_EXTERN PetscErrorCode IGACreate(MPI_Comm comm,IGA *iga);
PETSC_EXTERN PetscErrorCode IGADestroy(IGA *iga);
PETSC_EXTERN PetscErrorCode IGAReset(IGA iga);
PETSC_EXTERN PetscErrorCode IGASetUp(IGA iga);
PETSC_EXTERN PetscErrorCode IGAView(IGA iga,PetscViewer viewer);

PETSC_EXTERN PetscErrorCode IGAGetOptionsPrefix(IGA iga,const char *prefix[]);
PETSC_EXTERN PetscErrorCode IGASetOptionsPrefix(IGA iga,const char prefix[]);
PETSC_EXTERN PetscErrorCode IGAPrependOptionsPrefix(IGA iga,const char prefix[]);
PETSC_EXTERN PetscErrorCode IGAAppendOptionsPrefix(IGA iga,const char prefix[]);
PETSC_EXTERN PetscErrorCode IGASetFromOptions(IGA iga);

PETSC_EXTERN PetscErrorCode IGALoad(IGA iga,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode IGASave(IGA iga,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode IGARead(IGA iga,const char filename[]);
PETSC_EXTERN PetscErrorCode IGAWrite(IGA iga,const char filename[]);

PETSC_EXTERN PetscErrorCode IGASetGeometryDim(IGA iga,PetscInt dim);
PETSC_EXTERN PetscErrorCode IGAGetGeometryDim(IGA iga,PetscInt *dim);
PETSC_EXTERN PetscErrorCode IGALoadGeometry(IGA iga,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode IGASaveGeometry(IGA iga,PetscViewer viewer);

PETSC_EXTERN PetscErrorCode IGASetPropertyDim(IGA iga,PetscInt dim);
PETSC_EXTERN PetscErrorCode IGAGetPropertyDim(IGA iga,PetscInt *dim);
PETSC_EXTERN PetscErrorCode IGALoadProperty(IGA iga,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode IGASaveProperty(IGA iga,PetscViewer viewer);

PETSC_EXTERN PetscErrorCode IGALoadVec(IGA iga,Vec vec,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode IGASaveVec(IGA iga,Vec vec,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode IGAReadVec(IGA iga,Vec vec,const char filename[]);
PETSC_EXTERN PetscErrorCode IGAWriteVec(IGA iga,Vec vec,const char filename[]);

PETSC_EXTERN PetscErrorCode IGASetDim(IGA iga,PetscInt dim);
PETSC_EXTERN PetscErrorCode IGAGetDim(IGA iga,PetscInt *dim);
PETSC_EXTERN PetscErrorCode IGASetDof(IGA iga,PetscInt dof);
PETSC_EXTERN PetscErrorCode IGAGetDof(IGA iga,PetscInt *dof);
PETSC_EXTERN PetscErrorCode IGASetFieldName(IGA iga,PetscInt field,const char name[]);
PETSC_EXTERN PetscErrorCode IGAGetFieldName(IGA iga,PetscInt field,const char *name[]);
PETSC_EXTERN PetscErrorCode IGASetOrder(IGA iga,PetscInt order);
PETSC_EXTERN PetscErrorCode IGASetProcessors(IGA iga,PetscInt i,PetscInt processors);
PETSC_EXTERN PetscErrorCode IGASetUseCollocation(IGA iga,PetscBool collocation);

PETSC_EXTERN PetscErrorCode IGAGetAxis(IGA iga,PetscInt i,IGAAxis *axis);
PETSC_EXTERN PetscErrorCode IGAGetRule(IGA iga,PetscInt i,IGARule *rule);
PETSC_EXTERN PetscErrorCode IGAGetBasis(IGA iga,PetscInt i,IGABasis *basis);
PETSC_EXTERN PetscErrorCode IGAGetBoundary(IGA iga,PetscInt i,PetscInt side,IGABoundary *boundary);

PETSC_EXTERN PetscErrorCode IGAGetComm(IGA iga,MPI_Comm *comm);

PETSC_EXTERN PetscErrorCode IGACreateElemDM(IGA iga,PetscInt bs,DM *dm);
PETSC_EXTERN PetscErrorCode IGACreateGeomDM(IGA iga,PetscInt bs,DM *dm);
PETSC_EXTERN PetscErrorCode IGACreateNodeDM(IGA iga,PetscInt bs,DM *dm);

PETSC_EXTERN PetscErrorCode IGAGetElemDM(IGA iga,DM *dm);
PETSC_EXTERN PetscErrorCode IGAGetGeomDM(IGA iga,DM *dm);
PETSC_EXTERN PetscErrorCode IGAGetNodeDM(IGA iga,DM *dm);

PETSC_EXTERN PetscErrorCode IGASetVecType(IGA iga,const VecType vectype);
PETSC_EXTERN PetscErrorCode IGASetMatType(IGA iga,const MatType mattype);

PETSC_EXTERN PetscErrorCode IGACreateVec(IGA iga,Vec *vec);
PETSC_EXTERN PetscErrorCode IGACreateMat(IGA iga,Mat *mat);

PETSC_EXTERN PetscErrorCode IGACreateLocalVec(IGA iga, Vec *lvec);
PETSC_EXTERN PetscErrorCode IGAGetLocalVec(IGA iga,Vec *lvec);
PETSC_EXTERN PetscErrorCode IGARestoreLocalVec(IGA iga,Vec *lvec);
PETSC_EXTERN PetscErrorCode IGAGlobalToLocal(IGA iga,Vec gvec,Vec lvec);
PETSC_EXTERN PetscErrorCode IGALocalToGlobal(IGA iga,Vec lvec,Vec gvec,InsertMode addv);

PETSC_EXTERN PetscErrorCode IGAGetNaturalVec(IGA iga,Vec *nvec);
PETSC_EXTERN PetscErrorCode IGANaturalToGlobal(IGA iga,Vec nvec,Vec gvec);
PETSC_EXTERN PetscErrorCode IGAGlobalToNatural(IGA iga,Vec gvec,Vec nvec);

PETSC_EXTERN PetscErrorCode IGAGetLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[]);
PETSC_EXTERN PetscErrorCode IGARestoreLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[]);

PETSC_EXTERN PetscErrorCode IGASetUserSystem    (IGA iga,IGAUserSystem     System,    void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserFunction  (IGA iga,IGAUserFunction   Function,  void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserJacobian  (IGA iga,IGAUserJacobian   Jacobian,  void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserIFunction (IGA iga,IGAUserIFunction  IFunction, void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserIJacobian (IGA iga,IGAUserIJacobian  IJacobian, void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserIFunction2(IGA iga,IGAUserIFunction2 IFunction, void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserIJacobian2(IGA iga,IGAUserIJacobian2 IJacobian, void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserIEFunction(IGA iga,IGAUserIEFunction IEFunction,void *ctx);
PETSC_EXTERN PetscErrorCode IGASetUserIEJacobian(IGA iga,IGAUserIEJacobian IEJacobian,void *ctx);

PETSC_EXTERN PetscErrorCode IGAGetMeshInformation(IGA iga,PetscReal *hmin,PetscReal *hmax,PetscReal *havg,PetscReal *hstd);

/* ---------------------------------------------------------------- */

struct _n_IGAElement {
  PetscInt refct;
  /**/
  PetscInt  start[3];
  PetscInt  width[3];
  PetscInt  sizes[3];
  PetscInt  ID[3];
  PetscBool atboundary;
  PetscInt  atboundary_id;
  /**/
  PetscInt count;
  PetscInt index;
  /**/
  PetscInt nqp;
  PetscInt neq;
  PetscInt nen;
  PetscInt dof;
  PetscInt dim;
  PetscInt nsd;
  PetscInt npd;

  IGABasis *BD;

  PetscInt    *mapping;   /*[nen]      */
  PetscBool   geometry;
  PetscReal   *geometryX; /*[nen][nsd] */
  PetscBool   rational;
  PetscReal   *rationalW; /*[nen]      */
  PetscBool   property;
  PetscScalar *propertyA; /*[nen][npd] */

  PetscReal *weight;   /*   [nqp]                     */
  PetscReal *detJac;   /*   [nqp]                     */

  PetscReal *point;    /*   [nqp][dim]                */
  PetscReal *basis[4]; /*0: [nqp][nen]                */
                       /*1: [nqp][nen][dim]           */
                       /*2: [nqp][nen][dim][dim]      */
                       /*3: [nqp][nen][dim][dim][dim] */

  PetscReal *detX;     /*   [nqp]                     */
  PetscReal *gradX[2]; /*0: [nqp][nsd][dim]           */
                       /*1: [nqp][dim][nsd]           */
  PetscReal *shape[4]; /*0: [nqp][nen]                */
                       /*1: [nqp][nen][nsd]           */
                       /*2: [nqp][nen][nsd][nsd]      */
                       /*3: [nqp][nen][nsd][nsd][nsd] */

  PetscReal *normal;   /*   [nqp][dim]                */


  IGA      parent;
  IGAPoint iterator;

  PetscBool   collocation;

  PetscInt    *rowmap;
  PetscInt    *colmap;

  PetscInt     nfix;
  PetscInt    *ifix;
  PetscScalar *vfix;
  PetscScalar *ufix;

  PetscInt     nflux;
  PetscInt    *iflux;
  PetscScalar *vflux;

  PetscInt    nval;
  PetscScalar *wval[8];
  PetscInt    nvec;
  PetscScalar *wvec[8];
  PetscInt    nmat;
  PetscScalar *wmat[4];

};

PETSC_EXTERN PetscErrorCode IGAElementCreate(IGAElement *element);
PETSC_EXTERN PetscErrorCode IGAElementDestroy(IGAElement *element);
PETSC_EXTERN PetscErrorCode IGAElementReference(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementReset(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementInit(IGAElement element,IGA iga);

PETSC_EXTERN PetscErrorCode IGAGetElement(IGA iga,IGAElement *element);
PETSC_EXTERN PetscErrorCode IGABeginElement(IGA iga,IGAElement *element);
PETSC_EXTERN PetscBool      IGANextElement(IGA iga,IGAElement element);
PETSC_EXTERN PetscErrorCode IGAEndElement(IGA iga,IGAElement *element);
PETSC_EXTERN PetscBool      IGAElementNextUserOps(IGAElement element,IGAUserOps *userops);
PETSC_EXTERN PetscErrorCode IGAElementGetPoint(IGAElement element,IGAPoint *point);
PETSC_EXTERN PetscErrorCode IGAElementBeginPoint(IGAElement element,IGAPoint *point);
PETSC_EXTERN PetscBool      IGAElementNextPoint(IGAElement element,IGAPoint point);
PETSC_EXTERN PetscErrorCode IGAElementEndPoint(IGAElement element,IGAPoint *point);

PETSC_EXTERN PetscErrorCode IGAElementBuildMapping(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildGeometry(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildProperty(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildQuadrature(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildShapeFuns(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildQuadratureAtBoundary(IGAElement element,PetscInt dir,PetscInt side);
PETSC_EXTERN PetscErrorCode IGAElementBuildShapeFunsAtBoundary(IGAElement element,PetscInt dir,PetscInt side);

PETSC_EXTERN PetscErrorCode IGAElementGetParent(IGAElement element,IGA *parent);
PETSC_EXTERN PetscErrorCode IGAElementGetIndex(IGAElement element,PetscInt *index);
PETSC_EXTERN PetscErrorCode IGAElementGetCount(IGAElement element,PetscInt *count);

PETSC_EXTERN PetscErrorCode IGAElementGetSizes(IGAElement element,PetscInt *neq,PetscInt *nen,PetscInt *dof);
PETSC_EXTERN PetscErrorCode IGAElementGetMapping(IGAElement element,PetscInt *nen,const PetscInt *mapping[]);

PETSC_EXTERN PetscErrorCode IGAElementGetWorkVal(IGAElement element,PetscScalar *U[]);
PETSC_EXTERN PetscErrorCode IGAElementGetWorkVec(IGAElement element,PetscScalar *V[]);
PETSC_EXTERN PetscErrorCode IGAElementGetWorkMat(IGAElement element,PetscScalar *M[]);

PETSC_EXTERN PetscErrorCode IGAElementGetValues(IGAElement element,const PetscScalar U[],PetscScalar u[]);

PETSC_EXTERN PetscErrorCode IGAElementBuildFix(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementFixValues(IGAElement element,PetscScalar U[]);
PETSC_EXTERN PetscErrorCode IGAElementFixSystem(IGAElement element,PetscScalar K[],PetscScalar F[]);
PETSC_EXTERN PetscErrorCode IGAElementFixFunction(IGAElement element,PetscScalar F[]);
PETSC_EXTERN PetscErrorCode IGAElementFixJacobian(IGAElement element,PetscScalar J[]);

PETSC_EXTERN PetscErrorCode IGAElementAssembleVec(IGAElement element,const PetscScalar F[],Vec vec);
PETSC_EXTERN PetscErrorCode IGAElementAssembleMat(IGAElement element,const PetscScalar K[],Mat mat);

PETSC_EXTERN PetscErrorCode IGAElementCharacteristicSize(IGAElement element,PetscReal *h);

/* ---------------------------------------------------------------- */

struct _n_IGAPoint {
  PetscInt refct;
  /**/
  PetscInt count;
  PetscInt index;
  /**/
  PetscInt neq;
  PetscInt nen;
  PetscInt dof;
  PetscInt dim;
  PetscInt nsd;
  PetscInt npd;

  PetscReal   *geometry;/*  [nen][nsd] */
  PetscScalar *property;/*  [nen][npd] */

  PetscReal *weight;   /*   [1]   */
  PetscReal *detJac;   /*   [1]   */

  PetscReal *point;    /*   [dim] */
  PetscReal *basis[4]; /*0: [nen] */
                       /*1: [nen][dim] */
                       /*2: [nen][dim][dim] */
                       /*3: [nen][dim][dim][dim] */

  PetscReal *detX;     /*   [1] */
  PetscReal *gradX[2]; /*0: [nsd][dim] */
                       /*1: [dim][nsd] */
  PetscReal *shape[4]; /*0: [nen]  */
                       /*1: [nen][nsd] */
                       /*2: [nen][nsd][nsd] */
                       /*3: [nen][nsd][nsd][nsd] */

  PetscReal *normal;   /*   [dim] */

  IGAElement parent;

  PetscInt    nvec;
  PetscScalar *wvec[8];
  PetscInt    nmat;
  PetscScalar *wmat[4];
};
PETSC_EXTERN PetscErrorCode IGAPointCreate(IGAPoint *point);
PETSC_EXTERN PetscErrorCode IGAPointDestroy(IGAPoint *point);
PETSC_EXTERN PetscErrorCode IGAPointReference(IGAPoint point);
PETSC_EXTERN PetscErrorCode IGAPointReset(IGAPoint point);
PETSC_EXTERN PetscErrorCode IGAPointInit(IGAPoint point,IGAElement element);

PETSC_EXTERN PetscErrorCode IGAPointGetParent(IGAPoint point,IGAElement *element);
PETSC_EXTERN PetscErrorCode IGAPointGetIndex(IGAPoint point,PetscInt *index);
PETSC_EXTERN PetscErrorCode IGAPointGetCount(IGAPoint point,PetscInt *count);
PETSC_EXTERN PetscErrorCode IGAPointGetSizes(IGAPoint point,PetscInt *neq,PetscInt *nen,PetscInt *dof);
PETSC_EXTERN PetscErrorCode IGAPointGetDims(IGAPoint point,PetscInt *dim,PetscInt *nsd,PetscInt *npd);
PETSC_EXTERN PetscErrorCode IGAPointGetQuadrature(IGAPoint point,PetscReal *weigth,PetscReal *detJac);
PETSC_EXTERN PetscErrorCode IGAPointGetBasisFuns(IGAPoint point,PetscInt der,const PetscReal *basisfuns[]);
PETSC_EXTERN PetscErrorCode IGAPointGetShapeFuns(IGAPoint point,PetscInt der,const PetscReal *shapefuns[]);

PETSC_EXTERN PetscErrorCode IGAPointInterpolate(IGAPoint point,PetscInt ider,const PetscScalar U[],PetscScalar u[]);

PETSC_EXTERN PetscErrorCode IGAPointFormPoint    (IGAPoint p,PetscReal x[]);
PETSC_EXTERN PetscErrorCode IGAPointFormGradMap  (IGAPoint p,PetscReal map[],PetscReal inv[]);
PETSC_EXTERN PetscErrorCode IGAPointFormShapeFuns(IGAPoint p,PetscInt der,PetscReal N[]);

PETSC_EXTERN PetscErrorCode IGAPointFormGeomMap(IGAPoint p,PetscReal x[]);
PETSC_EXTERN PetscErrorCode IGAPointFormGradGeomMap(IGAPoint p,PetscReal F[]);
PETSC_EXTERN PetscErrorCode IGAPointFormInvGradGeomMap(IGAPoint p,PetscReal G[]);

PETSC_EXTERN PetscErrorCode IGAPointFormValue(IGAPoint p,const PetscScalar U[],PetscScalar u[]);
PETSC_EXTERN PetscErrorCode IGAPointFormGrad (IGAPoint p,const PetscScalar U[],PetscScalar u[]);
PETSC_EXTERN PetscErrorCode IGAPointFormHess (IGAPoint p,const PetscScalar U[],PetscScalar u[]);
PETSC_EXTERN PetscErrorCode IGAPointFormDel2 (IGAPoint p,const PetscScalar U[],PetscScalar u[]);
PETSC_EXTERN PetscErrorCode IGAPointFormDer3 (IGAPoint p,const PetscScalar U[],PetscScalar u[]);

PETSC_EXTERN PetscErrorCode IGAPointGetWorkVec(IGAPoint point,PetscScalar *V[]);
PETSC_EXTERN PetscErrorCode IGAPointGetWorkMat(IGAPoint point,PetscScalar *M[]);

PETSC_EXTERN PetscErrorCode IGAPointAddArray(IGAPoint point,PetscInt n,const PetscScalar a[],PetscScalar A[]);
PETSC_EXTERN PetscErrorCode IGAPointAddVec(IGAPoint point,const PetscScalar f[],PetscScalar F[]);
PETSC_EXTERN PetscErrorCode IGAPointAddMat(IGAPoint point,const PetscScalar k[],PetscScalar K[]);

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode IGAFormScalar(IGA iga,Vec U,PetscInt n,PetscScalar S[],
                                          IGAUserScalar Scalar,void *ctx);

#define PCEBE "ebe"
#define PCBBB "bbb"
PETSC_EXTERN PetscErrorCode IGACreateKSP(IGA iga,KSP *ksp);
PETSC_EXTERN PetscErrorCode IGAComputeSystem(IGA iga,Mat A,Vec B);

PETSC_EXTERN PetscErrorCode IGACreateSNES(IGA iga,SNES *snes);
PETSC_EXTERN PetscErrorCode IGAComputeFunction(IGA iga,Vec U,Vec F);
PETSC_EXTERN PetscErrorCode IGAComputeJacobian(IGA iga,Vec U,Mat J);

PETSC_EXTERN PetscErrorCode IGACreateTS(IGA iga,TS *ts);
PETSC_EXTERN PetscErrorCode IGAComputeIFunction(IGA iga,PetscReal dt,
                                                PetscReal a,Vec V,
                                                PetscReal t,Vec U,
                                                Vec F);
PETSC_EXTERN PetscErrorCode IGAComputeIJacobian(IGA iga,PetscReal dt,
                                                PetscReal a,Vec V,
                                                PetscReal t,Vec U,
                                                Mat J);
PETSC_EXTERN PetscErrorCode IGAComputeIEFunction(IGA iga,PetscReal dt,
                                                 PetscReal a, Vec V,
                                                 PetscReal t, Vec U,
                                                 PetscReal t0,Vec U0,
                                                 Vec F);
PETSC_EXTERN PetscErrorCode IGAComputeIEJacobian(IGA iga,PetscReal dt,
                                                 PetscReal a,Vec V,
                                                 PetscReal t,Vec U,
                                                 PetscReal t0,Vec U0,
                                                 Mat J);

PETSC_EXTERN PetscErrorCode IGACreateTS2(IGA iga, TS *ts);
PETSC_EXTERN PetscErrorCode IGAComputeIFunction2(IGA iga,PetscReal dt,
                                                 PetscReal a,Vec vecA,
                                                 PetscReal v,Vec vecV,
                                                 PetscReal t,Vec vecU,
                                                 Vec vecF);
PETSC_EXTERN PetscErrorCode IGAComputeIJacobian2(IGA iga,PetscReal dt,
                                                 PetscReal a,Vec vecA,
                                                 PetscReal v,Vec vecV,
                                                 PetscReal t,Vec vecU,
                                                 Mat matJ);

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode IGAPreparePCBDDC(IGA iga,PC pc);
PETSC_EXTERN PetscErrorCode IGASetOptionsHandlerPC(PC pc);
PETSC_EXTERN PetscErrorCode IGASetOptionsHandlerKSP(KSP ksp);
PETSC_EXTERN PetscErrorCode IGASetOptionsHandlerSNES(SNES snes);
PETSC_EXTERN PetscErrorCode IGASetOptionsHandlerTS(TS ts);

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscBool IGALocateElement(IGA iga,PetscReal *pnt,IGAElement element);
PETSC_EXTERN PetscErrorCode IGAPointEval(IGA iga,IGAPoint point);

/* ---------------------------------------------------------------- */

#if defined(PETSC_USE_DEBUG)
#define IGACheckSetUp(iga,arg) do {                                      \
    if (PetscUnlikely(!(iga)->setup))                                    \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,                 \
               "Must call IGASetUp() on argument %D \"%s\" before %s()", \
               (arg),#iga,PETSC_FUNCTION_NAME);                          \
    } while (0)
#else
#define IGACheckSetUp(iga,arg) do {} while (0)
#endif

#if defined(PETSC_USE_DEBUG)
#define IGACheckUserOp(iga,arg,UserOp) do {             \
    if (!iga->userops->UserOp)                          \
      SETERRQ4(((PetscObject)iga)->comm,PETSC_ERR_USER, \
               "Must call IGASetUser%s() "              \
               "on argument %D \"%s\" before %s()",     \
               #UserOp,(arg),#iga,PETSC_FUNCTION_NAME); \
    } while (0)
#else
#define IGACheckUserOp(iga,arg,UserOp) do {} while (0)
#endif

/* ---------------------------------------------------------------- */

#ifndef PETSC_VERSION_EQ
#define PETSC_VERSION_EQ(MAJOR,MINOR,SUBMINOR) \
  (PETSC_VERSION_(MAJOR,MINOR,SUBMINOR))
#endif
#ifndef PETSC_VERSION_LT
#define PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR) \
  (PETSC_VERSION_MAJOR < (MAJOR) ||            \
   (PETSC_VERSION_MAJOR == (MAJOR) &&          \
    (PETSC_VERSION_MINOR < (MINOR) ||          \
     (PETSC_VERSION_MINOR == (MINOR) &&        \
      (PETSC_VERSION_SUBMINOR < (SUBMINOR))))))
#define PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR) \
  (PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR) ||   \
   PETSC_VERSION_EQ(MAJOR,MINOR,SUBMINOR))
#endif
#ifndef PETSC_VERSION_GT
#define PETSC_VERSION_GT(MAJOR,MINOR,SUBMINOR) \
  (!PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR))
#define PETSC_VERSION_GE(MAJOR,MINOR,SUBMINOR) \
  (!PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR))
#endif

#ifndef PetscMalloc1
#define PetscMalloc1(m1,t1,r1) (PetscMalloc((m1)*sizeof(t1),(r1)))
#endif

#ifndef PetscValidRealPointer
#define PetscValidRealPointer PetscValidDoublePointer
#endif

#if PETSC_VERSION_LE(3,2,0)
#define PetscObjectTypeCompare PetscTypeCompare
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,2,0)
PETSC_EXTERN_CXX_END
#endif

#endif/*PETIGA_H*/
