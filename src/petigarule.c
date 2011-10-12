#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGARuleCreate"
PetscErrorCode IGARuleCreate(IGARule *rule)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  ierr = PetscNew(struct _n_IGARule,rule);CHKERRQ(ierr);
  (*rule)->refct = 1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleDestroy"
PetscErrorCode IGARuleDestroy(IGARule *_rule)
{
  IGARule        rule;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_rule,1);
  rule = *_rule; *_rule = 0;
  if (!rule) PetscFunctionReturn(0);
  if (--rule->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
  ierr = PetscFree(rule);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleReference"
PetscErrorCode IGARuleReference(IGARule rule)
{
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  rule->refct++;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleCopy"
PetscErrorCode IGARuleCopy(IGARule base,IGARule rule)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(base,1);
  PetscValidPointer(rule,2);
  rule->nqp = base->nqp;
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  if (base->point && base->nqp > 0) {
    ierr = PetscMalloc1(base->nqp,PetscReal,&rule->point);CHKERRQ(ierr);
    ierr = PetscMemcpy(rule->point,base->point,base->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  }
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
  if (base->weight && base->nqp > 0) {
    ierr = PetscMalloc1(base->nqp,PetscReal,&rule->weight);CHKERRQ(ierr);
    ierr = PetscMemcpy(rule->weight,base->weight,base->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleDuplicate"
PetscErrorCode IGARuleDuplicate(IGARule base,IGARule *rule)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(base,1);
  PetscValidPointer(rule,2);
  ierr = PetscNew(struct _n_IGARule,rule);CHKERRQ(ierr);
  ierr = IGARuleCopy(base,*rule);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GaussRule(PetscInt q, PetscReal X[], PetscReal W[]);

#undef  __FUNCT__
#define __FUNCT__ "IGARuleInit"
PetscErrorCode IGARuleInit(IGARule rule,PetscInt nqp)
{
  PetscReal      *point,*weight;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  if (nqp < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of quadrature points must be grather than zero, got %D",nqp);
  ierr = PetscMalloc1(nqp,PetscReal,&point);CHKERRQ(ierr);
  ierr = PetscMalloc1(nqp,PetscReal,&weight);CHKERRQ(ierr);
  if (GaussRule(nqp,point,weight) != 0)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of quadrature points %D not implemented",nqp);
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
  rule->nqp = nqp;
  rule->point = point;
  rule->weight = weight;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleSetRule"
PetscErrorCode IGARuleSetRule(IGARule rule,PetscInt q,const PetscReal x[],const PetscReal w[])
{
  PetscReal      *xx,*ww;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  PetscValidPointer(x,3);
  PetscValidPointer(w,4);
  if (q < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of quadrature points must be grather than zero, got %D",q);
  ierr = PetscMalloc1(q,PetscReal,&xx);CHKERRQ(ierr);
  ierr = PetscMalloc1(q,PetscReal,&ww);CHKERRQ(ierr);
  ierr = PetscMemcpy(xx,x,q*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(ww,w,q*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
  rule->nqp = q;
  rule->point = xx;
  rule->weight = ww;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleGetRule"
PetscErrorCode IGARuleGetRule(IGARule rule,PetscInt *q,PetscReal *x[],PetscReal *w[])
{
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  if (q) PetscValidPointer(q,2);
  if (x) PetscValidPointer(x,3);
  if (w) PetscValidPointer(w,4);
  if (q) *q = rule->nqp;
  if (x) *x = rule->point;
  if (w) *w = rule->weight;
  PetscFunctionReturn(0);
}

static PetscErrorCode GaussRule(PetscInt q, PetscReal X[], PetscReal W[])
{
  switch (q)  {
  case (1): /* p = 1 */
    X[0] = 0.0;
    W[0] = 2.0;
    break;
  case (2): /* p = 3 */
    X[0] = -0.5773502691896257645091487805019576; /* 1/sqrt(3) */
    X[1] = -X[0];
    W[0] =  1.0;
    W[1] =  W[0];
    break;
  case (3): /* p = 5 */
    X[0] = -0.7745966692414833770358530799564799; /* sqrt(3/5) */
    X[1] =  0.0;
    X[2] = -X[0];
    W[0] =  0.5555555555555555555555555555555556; /* 5/9 */
    W[1] =  0.8888888888888888888888888888888889; /* 8/9 */
    W[2] =  W[0];
    break;
  case (4): /* p = 7 */
    X[0] = -0.8611363115940525752239464888928094; /* sqrt((3+2*sqrt(6/5))/7) */
    X[1] = -0.3399810435848562648026657591032448; /* sqrt((3-2*sqrt(6/5))/7) */
    X[2] = -X[1];
    X[3] = -X[0];
    W[0] =  0.3478548451374538573730639492219994; /* (18-sqrt(30))/36 */
    W[1] =  0.6521451548625461426269360507780006; /* (18+sqrt(30))/36 */
    W[2] =  W[1];
    W[3] =  W[0];
    break;
  case (5): /* p = 9 */
    X[0] = -0.9061798459386639927976268782993929; /* 1/3*sqrt(5+2*sqrt(10/7)) */
    X[1] = -0.5384693101056830910363144207002086; /* 1/3*sqrt(5-2*sqrt(10/7)) */
    X[2] =  0.0;
    X[3] = -X[1];
    X[4] = -X[0];
    W[0] =  0.2369268850561890875142640407199173; /* (322-13*sqrt(70))/900 */
    W[1] =  0.4786286704993664680412915148356382; /* (322+13*sqrt(70))/900 */
    W[2] =  0.5688888888888888888888888888888889; /* 128/225 */
    W[3] =  W[1];
    W[4] =  W[0];
    break;
  default:
    return -1;
  }
  return 0;
}
