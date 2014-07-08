#if !defined(PETIGA_H)
#define PETIGA_H

/*
#include <petscconf.h>
#undef  PETSC_STATIC_INLINE
#define PETSC_STATIC_INLINE static __inline
*/

#include <petsc.h>
#include <petsc-private/petscimpl.h>
#include <petscts2.h>

typedef ISLocalToGlobalMapping LGMap;
#define LGMap LGMap

/* ---------------------------------------------------------------- */

typedef struct _p_IGA         *IGA;
typedef struct _n_IGAAxis     *IGAAxis;
typedef struct _n_IGARule     *IGARule;
typedef struct _n_IGABasis    *IGABasis;
typedef struct _n_IGAForm     *IGAForm;
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
  PetscInt   nnp,nel;  /* bases, spans */
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
PETSC_EXTERN PetscErrorCode IGAAxisSetKnots(IGAAxis axis,PetscInt m,const PetscReal U[]);
PETSC_EXTERN PetscErrorCode IGAAxisGetKnots(IGAAxis axis,PetscInt *m,PetscReal *U[]);
PETSC_EXTERN PetscErrorCode IGAAxisGetLimits(IGAAxis axis,PetscReal *Ui,PetscReal *Uf);
PETSC_EXTERN PetscErrorCode IGAAxisGetSizes(IGAAxis axis,PetscInt *nel,PetscInt *nnp);
PETSC_EXTERN PetscErrorCode IGAAxisGetSpans(IGAAxis axis,PetscInt *nel,PetscInt *spans[]);
PETSC_EXTERN PetscErrorCode IGAAxisInit(IGAAxis axis,PetscInt p,PetscInt m,const PetscReal U[]);
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
  IGA_BASIS_LAGRANGE,
  IGA_BASIS_HIERARCHICAL
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

typedef PetscErrorCode (*IGAFormScalar)(IGAPoint point,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx);
typedef PetscErrorCode (*IGAFormVector)(IGAPoint point,PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAFormMatrix)(IGAPoint point,PetscScalar *K,void *ctx);
typedef PetscErrorCode (*IGAFormSystem)(IGAPoint point,PetscScalar *K,PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAFormFunction)(IGAPoint point,const PetscScalar *U,PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAFormJacobian)(IGAPoint point,const PetscScalar *U,PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAFormIFunction)(IGAPoint point,PetscReal dt,
                                           PetscReal a,const PetscScalar *V,
                                           PetscReal t,const PetscScalar *U,
                                           PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAFormIJacobian)(IGAPoint point,PetscReal dt,
                                           PetscReal a,const PetscScalar *V,
                                           PetscReal t,const PetscScalar *U,
                                           PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAFormIFunction2)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *A,
                                            PetscReal v,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAFormIJacobian2)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *A,
                                            PetscReal v,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAFormIEFunction)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscReal t0,const PetscScalar *U0,
                                            PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAFormIEJacobian)(IGAPoint point,PetscReal dt,
                                            PetscReal a,const PetscScalar *V,
                                            PetscReal t,const PetscScalar *U,
                                            PetscReal t0,const PetscScalar *U0,
                                            PetscScalar *J,void *ctx);
typedef PetscErrorCode (*IGAFormRHSFunction)(IGAPoint point,PetscReal dt,
                                             PetscReal t,const PetscScalar *U,
                                             PetscScalar *F,void *ctx);
typedef PetscErrorCode (*IGAFormRHSJacobian)(IGAPoint point,PetscReal dt,
                                             PetscReal t,const PetscScalar *U,
                                             PetscScalar *J,void *ctx);

typedef struct _IGAFormBC *IGAFormBC;
struct _IGAFormBC {
  PetscInt    count;
  PetscInt    field[64];
  PetscScalar value[64];
};

typedef struct _IGAFormOps *IGAFormOps;
struct _IGAFormOps {
  /**/
  IGAFormVector     Vector;
  void              *VecCtx;
  IGAFormMatrix     Matrix;
  void              *MatCtx;
  IGAFormSystem     System;
  void              *SysCtx;
  /**/
  IGAFormFunction   Function;
  void              *FunCtx;
  IGAFormJacobian   Jacobian;
  void              *JacCtx;
  /**/
  IGAFormIFunction  IFunction;
  IGAFormIFunction2 IFunction2;
  void              *IFunCtx;
  IGAFormIJacobian  IJacobian;
  IGAFormIJacobian2 IJacobian2;
  void              *IJacCtx;
  /**/
  IGAFormIEFunction IEFunction;
  void              *IEFunCtx;
  IGAFormIEJacobian IEJacobian;
  void              *IEJacCtx;
  /**/
  IGAFormRHSFunction RHSFunction;
  void               *RHSFunCtx;
  IGAFormRHSJacobian RHSJacobian;
  void               *RHSJacCtx;
};

struct _n_IGAForm {
  PetscInt refct;
  /**/
  IGAFormOps ops;
  PetscInt   dof;
  IGAFormBC  value[3][2];
  IGAFormBC  load [3][2];
  PetscBool  visit[3][2];
};

PETSC_EXTERN PetscErrorCode IGAGetForm(IGA iga,IGAForm *form);
PETSC_EXTERN PetscErrorCode IGASetForm(IGA iga,IGAForm form);

PETSC_EXTERN PetscErrorCode IGAFormCreate(IGAForm *form);
PETSC_EXTERN PetscErrorCode IGAFormDestroy(IGAForm *form);
PETSC_EXTERN PetscErrorCode IGAFormReset(IGAForm form);
PETSC_EXTERN PetscErrorCode IGAFormReference(IGAForm form);

PETSC_EXTERN PetscErrorCode IGAFormSetBoundaryValue(IGAForm form,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value);
PETSC_EXTERN PetscErrorCode IGAFormSetBoundaryLoad (IGAForm form,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value);
PETSC_EXTERN PetscErrorCode IGAFormSetBoundaryForm (IGAForm form,PetscInt axis,PetscInt side,PetscBool flag);
PETSC_EXTERN PetscErrorCode IGAFormClearBoundary   (IGAForm form,PetscInt axis,PetscInt side);

PETSC_EXTERN PetscErrorCode IGAFormSetVector     (IGAForm form,IGAFormVector      Vector,     void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetMatrix     (IGAForm form,IGAFormMatrix      Matrix,     void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetSystem     (IGAForm form,IGAFormSystem      System,     void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetFunction   (IGAForm form,IGAFormFunction    Function,   void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetJacobian   (IGAForm form,IGAFormJacobian    Jacobian,   void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetIFunction  (IGAForm form,IGAFormIFunction   IFunction,  void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetIJacobian  (IGAForm form,IGAFormIJacobian   IJacobian,  void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetIFunction2 (IGAForm form,IGAFormIFunction2  IFunction,  void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetIJacobian2 (IGAForm form,IGAFormIJacobian2  IJacobian,  void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetIEFunction (IGAForm form,IGAFormIEFunction  IEFunction, void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetIEJacobian (IGAForm form,IGAFormIEJacobian  IEJacobian, void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetRHSFunction(IGAForm form,IGAFormRHSFunction RHSFunction,void *ctx);
PETSC_EXTERN PetscErrorCode IGAFormSetRHSJacobian(IGAForm form,IGAFormRHSJacobian RHSJacobian,void *ctx);

PETSC_EXTERN PetscErrorCode IGASetFixTable(IGA iga,Vec table);
PETSC_EXTERN PetscErrorCode IGASetBoundaryValue(IGA iga,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value);
PETSC_EXTERN PetscErrorCode IGASetBoundaryLoad (IGA iga,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value);
PETSC_EXTERN PetscErrorCode IGASetBoundaryForm (IGA iga,PetscInt axis,PetscInt side,PetscBool flag);

PETSC_EXTERN PetscErrorCode IGASetFormVector     (IGA iga,IGAFormVector      Vector,     void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormMatrix     (IGA iga,IGAFormMatrix      Matrix,     void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormSystem     (IGA iga,IGAFormSystem      System,     void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormFunction   (IGA iga,IGAFormFunction    Function,   void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormJacobian   (IGA iga,IGAFormJacobian    Jacobian,   void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormIFunction  (IGA iga,IGAFormIFunction   IFunction,  void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormIJacobian  (IGA iga,IGAFormIJacobian   IJacobian,  void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormIFunction2 (IGA iga,IGAFormIFunction2  IFunction,  void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormIJacobian2 (IGA iga,IGAFormIJacobian2  IJacobian,  void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormIEFunction (IGA iga,IGAFormIEFunction  IEFunction, void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormIEJacobian (IGA iga,IGAFormIEJacobian  IEJacobian, void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormRHSFunction(IGA iga,IGAFormRHSFunction RHSFunction,void *ctx);
PETSC_EXTERN PetscErrorCode IGASetFormRHSJacobian(IGA iga,IGAFormRHSJacobian RHSJacobian,void *ctx);

/* ---------------------------------------------------------------- */

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
  PetscBool  collocation;

  VecType    vectype;
  MatType    mattype;
  char       **fieldname;

  PetscInt  dim;   /* parametric dimension of the function space*/
  PetscInt  dof;   /* number of degrees of freedom per node */
  PetscInt  order; /* maximum derivative order */

  IGAAxis     axis[3];
  IGARule     rule[3];
  IGABasis    basis[3];
  IGAForm     form;

  IGAElement  iterator;

  PetscBool   rational;
  PetscInt    geometry;
  PetscInt    property;
  PetscBool   fixtable;
  PetscReal   *rationalW;
  PetscReal   *geometryX;
  PetscScalar *propertyA;
  PetscScalar *fixtableU;

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

  AO          ao;
  LGMap       lgmap;
  PetscLayout map;
  VecScatter  g2l,l2g,l2l;
  PetscInt    nwork;
  Vec         vwork[16];
  Vec         natural;
  VecScatter  n2g,g2n;
};

PETSC_EXTERN PetscClassId IGA_CLASSID;
#define IGA_FILE_CLASSID 1211299

PETSC_EXTERN PetscErrorCode IGAInitializePackage(void);
PETSC_EXTERN PetscErrorCode IGAFinalizePackage(void);
PETSC_EXTERN PetscErrorCode IGARegisterAll(void);

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
PETSC_EXTERN PetscErrorCode IGAViewFromOptions(IGA iga,const char prefix[],const char option[]);
PETSC_EXTERN PetscErrorCode IGAOptionsAlias(const char name[],const char defval[],const char alias[]);

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
PETSC_EXTERN PetscErrorCode IGAGetOrder(IGA iga,PetscInt *order);
PETSC_EXTERN PetscErrorCode IGASetProcessors(IGA iga,PetscInt i,PetscInt processors);
PETSC_EXTERN PetscErrorCode IGASetBasisType(IGA iga,PetscInt i,IGABasisType type);
PETSC_EXTERN PetscErrorCode IGASetQuadrature(IGA iga,PetscInt i,PetscInt q);
PETSC_EXTERN PetscErrorCode IGASetUseCollocation(IGA iga,PetscBool collocation);

PETSC_EXTERN PetscErrorCode IGAGetAxis(IGA iga,PetscInt i,IGAAxis *axis);
PETSC_EXTERN PetscErrorCode IGAGetRule(IGA iga,PetscInt i,IGARule *rule);
PETSC_EXTERN PetscErrorCode IGAGetBasis(IGA iga,PetscInt i,IGABasis *basis);

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

PETSC_EXTERN PetscErrorCode IGACreateCoordinates(IGA iga,Vec *coords);
PETSC_EXTERN PetscErrorCode IGACreateRigidBody(IGA iga,MatNullSpace *nsp);

PETSC_EXTERN PetscErrorCode IGACreateLocalVec(IGA iga, Vec *lvec);
PETSC_EXTERN PetscErrorCode IGAGetLocalVec(IGA iga,Vec *lvec);
PETSC_EXTERN PetscErrorCode IGARestoreLocalVec(IGA iga,Vec *lvec);
PETSC_EXTERN PetscErrorCode IGAGlobalToLocalBegin(IGA iga,Vec gvec,Vec lvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGAGlobalToLocalEnd  (IGA iga,Vec gvec,Vec lvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGAGlobalToLocal     (IGA iga,Vec gvec,Vec lvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGALocalToGlobalBegin(IGA iga,Vec lvec,Vec gvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGALocalToGlobalEnd  (IGA iga,Vec lvec,Vec gvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGALocalToGlobal     (IGA iga,Vec lvec,Vec gvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGALocalToLocalBegin (IGA iga,Vec gvec,Vec lvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGALocalToLocalEnd   (IGA iga,Vec gvec,Vec lvec,InsertMode addv);
PETSC_EXTERN PetscErrorCode IGALocalToLocal      (IGA iga,Vec gvec,Vec lvec,InsertMode addv);

PETSC_EXTERN PetscErrorCode IGAGetNaturalVec(IGA iga,Vec *nvec);
PETSC_EXTERN PetscErrorCode IGANaturalToGlobal(IGA iga,Vec nvec,Vec gvec);
PETSC_EXTERN PetscErrorCode IGAGlobalToNatural(IGA iga,Vec gvec,Vec nvec);

PETSC_EXTERN PetscErrorCode IGAGetLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[]);
PETSC_EXTERN PetscErrorCode IGARestoreLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[]);

PETSC_EXTERN PetscErrorCode IGAClone(IGA iga,PetscInt dof,IGA *newiga);

#undef  DMIGA
#define DMIGA "iga"
PETSC_EXTERN PetscErrorCode IGACreateWrapperDM(IGA iga,DM *dm);
PETSC_EXTERN PetscErrorCode DMIGASetIGA(DM dm,IGA iga);
PETSC_EXTERN PetscErrorCode DMIGAGetIGA(DM dm,IGA *iga);

/* ---------------------------------------------------------------- */

struct _n_IGAElement {
  PetscInt refct;
  /**/
  PetscInt  start[3];
  PetscInt  width[3];
  PetscInt  sizes[3];
  PetscInt  ID[3];
  /**/
  PetscBool atboundary;
  PetscInt  boundary_id;
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

  PetscInt    *mapping;   /*[nen]      */
  PetscInt    *rowmap;    /*[neq]      */
  PetscInt    *colmap;    /*[nen]      */
  PetscBool   rational;
  PetscReal   *rationalW; /*[nen]      */
  PetscBool   geometry;
  PetscReal   *geometryX; /*[nen][nsd] */
  PetscBool   property;
  PetscScalar *propertyA; /*[nen][npd] */

  PetscInt  nqp;

  PetscReal *point;    /*   [nqp][dim]                */
  PetscReal *weight;   /*   [nqp]                     */
  PetscReal *detJac;   /*   [nqp]                     */

  PetscReal *basis[4]; /*0: [nqp][nen]                */
                       /*1: [nqp][nen][dim]           */
                       /*2: [nqp][nen][dim][dim]      */
                       /*3: [nqp][nen][dim][dim][dim] */

  PetscReal *detX;     /*   [nqp]                     */
  PetscReal *gradX[2]; /*0: [nqp][nsd][dim]           */
                       /*1: [nqp][dim][nsd]           */
  PetscReal *hessX[2]; /*0: [nqp][nsd][dim][dim]      */
                       /*1: [nqp][dim][nsd][nsd]      */
  PetscReal *der3X[2]; /*0: [nqp][nsd][dim][dim][dim] */
                       /*1: [nqp][dim][nsd][nsd][nsd] */
  PetscReal *detS;     /*   [nqp]                     */
  PetscReal *normal;   /*   [nqp][dim]                */

  PetscReal *shape[4]; /*0: [nqp][nen]                */
                       /*1: [nqp][nen][nsd]           */
                       /*2: [nqp][nen][nsd][nsd]      */
                       /*3: [nqp][nen][nsd][nsd][nsd] */

  IGA      parent;
  IGAPoint iterator;

  PetscBool   collocation;

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
PETSC_EXTERN PetscBool      IGAElementNextForm(IGAElement element,PetscBool visit[][2]);
PETSC_EXTERN PetscErrorCode IGAElementGetPoint(IGAElement element,IGAPoint *point);
PETSC_EXTERN PetscErrorCode IGAElementBeginPoint(IGAElement element,IGAPoint *point);
PETSC_EXTERN PetscBool      IGAElementNextPoint(IGAElement element,IGAPoint point);
PETSC_EXTERN PetscErrorCode IGAElementEndPoint(IGAElement element,IGAPoint *point);

PETSC_EXTERN PetscErrorCode IGAElementBuildClosure(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildQuadrature(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildShapeFuns (IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementBuildQuadratureAtBoundary(IGAElement element,PetscInt dir,PetscInt side);
PETSC_EXTERN PetscErrorCode IGAElementBuildShapeFunsAtBoundary (IGAElement element,PetscInt dir,PetscInt side);

PETSC_EXTERN PetscErrorCode IGAElementGetParent(IGAElement element,IGA *parent);
PETSC_EXTERN PetscErrorCode IGAElementGetIndex(IGAElement element,PetscInt *index);
PETSC_EXTERN PetscErrorCode IGAElementGetCount(IGAElement element,PetscInt *count);
PETSC_EXTERN PetscErrorCode IGAElementGetSizes(IGAElement element,PetscInt *neq,PetscInt *nen,PetscInt *dof);
PETSC_EXTERN PetscErrorCode IGAElementGetClosure(IGAElement element,PetscInt *nen,const PetscInt *mapping[]);
PETSC_EXTERN PetscErrorCode IGAElementGetIndices(IGAElement element,
                                                 PetscInt *neq,const PetscInt *rowmap[],
                                                 PetscInt *nen,const PetscInt *colmap[]);

PETSC_EXTERN PetscErrorCode IGAElementGetWorkVec(IGAElement element,PetscScalar *V[]);
PETSC_EXTERN PetscErrorCode IGAElementGetWorkMat(IGAElement element,PetscScalar *M[]);
PETSC_EXTERN PetscErrorCode IGAElementGetValues(IGAElement element,const PetscScalar arrayU[],PetscScalar *U[]);

PETSC_EXTERN PetscErrorCode IGAElementBuildFix(IGAElement element);
PETSC_EXTERN PetscErrorCode IGAElementFixValues(IGAElement element,PetscScalar U[]);
PETSC_EXTERN PetscErrorCode IGAElementFixSystem(IGAElement element,PetscScalar K[],PetscScalar F[]);
PETSC_EXTERN PetscErrorCode IGAElementFixFunction(IGAElement element,PetscScalar F[]);
PETSC_EXTERN PetscErrorCode IGAElementFixJacobian(IGAElement element,PetscScalar J[]);

PETSC_EXTERN PetscErrorCode IGAElementAssembleVec(IGAElement element,const PetscScalar F[],Vec vec);
PETSC_EXTERN PetscErrorCode IGAElementAssembleMat(IGAElement element,const PetscScalar K[],Mat mat);

/* ---------------------------------------------------------------- */

struct _n_IGAPoint {
  PetscInt refct;
  /**/
  PetscBool atboundary;
  PetscInt  boundary_id;
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

  PetscReal   *rational;/*  [nen]      */
  PetscReal   *geometry;/*  [nen][nsd] */
  PetscScalar *property;/*  [nen][npd] */

  PetscReal *point;    /*   [dim] */
  PetscReal *weight;   /*   [1]   */
  PetscReal *detJac;   /*   [1]   */

  PetscReal *basis[4]; /*0: [nen] */
                       /*1: [nen][dim] */
                       /*2: [nen][dim][dim] */
                       /*3: [nen][dim][dim][dim] */

  PetscReal *detX;     /*   [1] */
  PetscReal *gradX[2]; /*0: [nsd][dim] */
                       /*1: [dim][nsd] */
  PetscReal *hessX[2]; /*0: [nsd][dim][dim] */
                       /*1: [dim][nsd][nsd] */
  PetscReal *der3X[2]; /*0: [nsd][dim][dim][dim] */
                       /*1: [dim][nsd][nsd][nsd] */
  PetscReal *detS;     /*   [1] */
  PetscReal *normal;   /*   [dim] */

  PetscReal *shape[4]; /*0: [nen]  */
                       /*1: [nen][nsd] */
                       /*2: [nen][nsd][nsd] */
                       /*3: [nen][nsd][nsd][nsd] */

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
PETSC_EXTERN PetscErrorCode IGAPointAtBoundary(IGAPoint point,PetscInt *axis,PetscInt *side);
PETSC_EXTERN PetscErrorCode IGAPointGetSizes(IGAPoint point,PetscInt *neq,PetscInt *nen,PetscInt *dof);
PETSC_EXTERN PetscErrorCode IGAPointGetDims(IGAPoint point,PetscInt *dim,PetscInt *nsd,PetscInt *npd);
PETSC_EXTERN PetscErrorCode IGAPointGetQuadrature(IGAPoint point,PetscReal *weigth,PetscReal *detJac);
PETSC_EXTERN PetscErrorCode IGAPointGetBasisFuns(IGAPoint point,PetscInt der,const PetscReal *basisfuns[]);
PETSC_EXTERN PetscErrorCode IGAPointGetShapeFuns(IGAPoint point,PetscInt der,const PetscReal *shapefuns[]);

PETSC_EXTERN PetscErrorCode IGAPointInterpolate(IGAPoint point,PetscInt ider,const PetscScalar U[],PetscScalar u[]);

PETSC_EXTERN PetscErrorCode IGAPointFormPoint(IGAPoint p,PetscReal x[]);
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

PETSC_EXTERN PetscLogEvent IGA_FormScalar;
PETSC_EXTERN PetscLogEvent IGA_FormVector;
PETSC_EXTERN PetscLogEvent IGA_FormMatrix;
PETSC_EXTERN PetscLogEvent IGA_FormSystem;
PETSC_EXTERN PetscLogEvent IGA_FormFunction;
PETSC_EXTERN PetscLogEvent IGA_FormJacobian;
PETSC_EXTERN PetscLogEvent IGA_FormIFunction;
PETSC_EXTERN PetscLogEvent IGA_FormIJacobian;

PETSC_EXTERN PetscErrorCode IGAComputeScalar(IGA iga,Vec U,
                                             PetscInt n,PetscScalar S[],
                                             IGAFormScalar Scalar,void *ctx);

#define PCIGAEBE "igaebe"
#define PCIGABBB "igabbb"

PETSC_EXTERN PetscErrorCode IGACreateKSP(IGA iga,KSP *ksp);
PETSC_EXTERN PetscErrorCode IGAComputeVector(IGA iga,Vec B);
PETSC_EXTERN PetscErrorCode IGAComputeMatrix(IGA iga,Mat A);
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
                                                 PetscReal a, Vec V,
                                                 PetscReal t, Vec U,
                                                 PetscReal t0,Vec U0,
                                                 Mat J);
PETSC_EXTERN PetscErrorCode IGAComputeRHSFunction(IGA iga,PetscReal dt,
                                                  PetscReal t,Vec U,
                                                  Vec F);
PETSC_EXTERN PetscErrorCode IGAComputeRHSJacobian(IGA iga,PetscReal dt,
                                                  PetscReal t,Vec U,
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

#if defined(PETSC_USE_DEBUG)
#define IGACheckSetUp(iga,arg) do {                                      \
    if (PetscUnlikely(!(iga)->setup))                                    \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,                 \
               "Must call IGASetUp() on argument %D \"%s\" before %s()", \
               (arg),#iga,PETSC_FUNCTION_NAME);                          \
    } while (0)
#define IGACheckSetUpStage(iga,arg,stg) do {                             \
    if (PetscUnlikely((iga)->setupstage<(stg))) IGACheckSetUp(iga,arg);  \
    } while (0)
#else
#define IGACheckSetUp(iga,arg)          do {} while (0)
#define IGACheckSetUpStage(iga,arg,stg) do {} while (0)
#endif
#define IGACheckSetUpStage1(iga,arg) IGACheckSetUpStage(iga,arg,1)
#define IGACheckSetUpStage2(iga,arg) IGACheckSetUpStage(iga,arg,2)

#if defined(PETSC_USE_DEBUG)
#define IGACheckFormOp(iga,arg,FormOp) do {             \
    if (!iga->form->ops->FormOp)                        \
      SETERRQ4(((PetscObject)iga)->comm,PETSC_ERR_USER, \
               "Must call IGASetForm%s() "              \
               "on argument %D \"%s\" before %s()",     \
               #FormOp,(arg),#iga,PETSC_FUNCTION_NAME); \
    } while (0)
#else
#define IGACheckFormOp(iga,arg,FormOp) do {} while (0)
#endif

/* ---------------------------------------------------------------- */

#ifndef PetscValidRealPointer
#define PetscValidRealPointer PetscValidDoublePointer
#endif

#ifndef PetscMalloc1
#define PetscMalloc1(m1,r1) \
  PetscMalloc((m1)*sizeof(**(r1)),r1)
#endif

#ifndef PetscCalloc1
#define PetscCalloc1(m1,r1) \
  (PetscMalloc1((m1),r1) || PetscMemzero(*(r1),(m1)*sizeof(**(r1))))
#endif

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(3,3,0)

#undef  PETSC_VERSION_LT
#define PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR)    \
  (PETSC_VERSION_RELEASE == 1 &&                  \
   (PETSC_VERSION_MAJOR < (MAJOR) ||              \
    (PETSC_VERSION_MAJOR == (MAJOR) &&            \
     (PETSC_VERSION_MINOR < (MINOR) ||            \
      (PETSC_VERSION_MINOR == (MINOR) &&          \
       (PETSC_VERSION_SUBMINOR < (SUBMINOR)))))))

#undef  PETSC_VERSION_LE
#define PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR) \
  (PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR) ||   \
   PETSC_VERSION_(MAJOR,MINOR,SUBMINOR))

#undef  PETSC_VERSION_GE
#define PETSC_VERSION_GE(MAJOR,MINOR,SUBMINOR) \
  (0 == PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR))

#undef  PETSC_VERSION_GT
#define PETSC_VERSION_GT(MAJOR,MINOR,SUBMINOR) \
  (0 == PETSC_VERSION_LE(MAJOR,MINOR,SUBMINOR))

#endif

/* ---------------------------------------------------------------- */

#endif/*PETIGA_H*/
