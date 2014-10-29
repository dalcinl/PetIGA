#include "petiga.h"

const char *const IGARuleTypes[] = {
  "LEGENDRE",
  "LOBATTO",
  "USER",
  /* */
  "IGARuleType","IGA_RULE_",NULL};

#undef  __FUNCT__
#define __FUNCT__ "IGARuleCreate"
PetscErrorCode IGARuleCreate(IGARule *rule)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  ierr = PetscCalloc1(1,rule);CHKERRQ(ierr);
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
  rule = *_rule; *_rule = NULL;
  if (!rule) PetscFunctionReturn(0);
  if (--rule->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
  ierr = PetscFree(rule);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleReset"
PetscErrorCode IGARuleReset(IGARule rule)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!rule) PetscFunctionReturn(0);
  PetscValidPointer(rule,1);
  rule->nqp = 0;
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
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
  if (base == rule) PetscFunctionReturn(0);

  rule->nqp = base->nqp;
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  if (base->point && base->nqp > 0) {
    ierr = PetscMalloc1(base->nqp,&rule->point);CHKERRQ(ierr);
    ierr = PetscMemcpy(rule->point,base->point,base->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  }
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
  if (base->weight && base->nqp > 0) {
    ierr = PetscMalloc1(base->nqp,&rule->weight);CHKERRQ(ierr);
    ierr = PetscMemcpy(rule->weight,base->weight,base->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  }
  rule->type = base->type;
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
  ierr = PetscCalloc1(1,rule);CHKERRQ(ierr);
  (*rule)->refct = 1;
  ierr = IGARuleCopy(base,*rule);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARuleSetType"
PetscErrorCode IGARuleSetType(IGARule rule,IGARuleType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  if (rule->type != type) {ierr = IGARuleReset(rule);CHKERRQ(ierr);}
  rule->type = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode IGA_Rule_GaussLegendre (PetscInt q,PetscReal X[],PetscReal W[]);
static PetscErrorCode IGA_Rule_GaussLobatto  (PetscInt q,PetscReal X[],PetscReal W[]);

#undef  __FUNCT__
#define __FUNCT__ "IGARuleInit"
PetscErrorCode IGARuleInit(IGARule rule,PetscInt nqp)
{
  PetscReal      *point,*weight;
  PetscErrorCode (*ComputeRule)(PetscInt,PetscReal[],PetscReal[]) = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  if (nqp < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of quadrature points must be greater than zero, got %D",nqp);

  ierr = PetscCalloc1(nqp,&point);CHKERRQ(ierr);
  ierr = PetscCalloc1(nqp,&weight);CHKERRQ(ierr);
  switch (rule->type) {
  case IGA_RULE_LEGENDRE:
    ComputeRule = IGA_Rule_GaussLegendre; break;
  case IGA_RULE_LOBATTO:
    ComputeRule = IGA_Rule_GaussLobatto; break;
  case IGA_RULE_USER:
    ComputeRule = NULL; break;
  }
  if (ComputeRule && ComputeRule(nqp,point,weight) != 0)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of quadrature points %D not implemented",nqp);

  ierr = IGARuleReset(rule);CHKERRQ(ierr);
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
             "Number of quadrature points must be greater than zero, got %D",q);
  ierr = PetscMalloc1(q,&xx);CHKERRQ(ierr);
  ierr = PetscMalloc1(q,&ww);CHKERRQ(ierr);
  ierr = PetscMemcpy(xx,x,q*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(ww,w,q*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscFree(rule->point);CHKERRQ(ierr);
  ierr = PetscFree(rule->weight);CHKERRQ(ierr);
  rule->nqp = q;
  rule->point = xx;
  rule->weight = ww;
  rule->type = IGA_RULE_USER;
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

#if   defined(PETSC_USE_REAL_SINGLE)
#define Q(constant) constant##f
#elif defined(PETSC_USE_REAL_DOUBLE)
#define Q(constant) constant
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#define Q(constant) constant##L
#elif   defined(PETSC_USE_REAL___FLOAT128)
#define Q(constant) constant##Q
#endif

static PetscErrorCode IGA_Rule_GaussLegendre(PetscInt q,PetscReal X[],PetscReal W[])
{
  switch (q)  {
  case (1): /* p = 1 */
    X[0] =  Q(0.0);
    W[0] =  Q(2.0);
    break;
  case (2): /* p = 3 */
    X[0] = -Q(0.577350269189625764509148780501957455); /* 1/sqrt(3) */
    X[1] = -X[0];
    W[0] =  Q(1.0);
    W[1] =  W[0];
    break;
  case (3): /* p = 5 */
    X[0] = -Q(0.774596669241483377035853079956479922); /* sqrt(3/5) */
    X[1] =  Q(0.0);
    X[2] = -X[0];
    W[0] =  Q(0.555555555555555555555555555555555556); /* 5/9 */
    W[1] =  Q(0.888888888888888888888888888888888889); /* 8/9 */
    W[2] =  W[0];
    break;
  case (4): /* p = 7 */
    X[0] = -Q(0.861136311594052575223946488892809506); /* sqrt((3+2*sqrt(6/5))/7) */
    X[1] = -Q(0.339981043584856264802665759103244686); /* sqrt((3-2*sqrt(6/5))/7) */
    X[2] = -X[1];
    X[3] = -X[0];
    W[0] =  Q(0.347854845137453857373063949221999408); /* (18-sqrt(30))/36 */
    W[1] =  Q(0.652145154862546142626936050778000592); /* (18+sqrt(30))/36 */
    W[2] =  W[1];
    W[3] =  W[0];
    break;
  case (5): /* p = 9 */
    X[0] = -Q(0.906179845938663992797626878299392962); /* 1/3*sqrt(5+2*sqrt(10/7)) */
    X[1] = -Q(0.538469310105683091036314420700208806); /* 1/3*sqrt(5-2*sqrt(10/7)) */
    X[2] =  Q(0.0);
    X[3] = -X[1];
    X[4] = -X[0];
    W[0] =  Q(0.236926885056189087514264040719917362); /* (322-13*sqrt(70))/900 */
    W[1] =  Q(0.478628670499366468041291514835638193); /* (322+13*sqrt(70))/900 */
    W[2] =  Q(0.568888888888888888888888888888888889); /* 128/225 */
    W[3] =  W[1];
    W[4] =  W[0];
    break;
  case (6): /* p = 11 */
    X[0] = -Q(0.9324695142031520278123015544939946); /* << NumericalDifferentialEquationAnalysis` */
    X[1] = -Q(0.6612093864662645136613995950199053); /* GaussianQuadratureWeights[6, -1, 1, 37]   */
    X[2] = -Q(0.2386191860831969086305017216807119);
    X[3] = -X[2];
    X[4] = -X[1];
    X[5] = -X[0];
    W[0] =  Q(0.171324492379170345040296142172732894);
    W[1] =  Q(0.360761573048138607569833513837716112);
    W[2] =  Q(0.467913934572691047389870343989550995);
    W[3] =  W[2];
    W[4] =  W[1];
    W[5] =  W[0];
    break;
  case (7): /* p = 13 */
    X[0] = -Q(0.9491079123427585245261896840478513); /* << NumericalDifferentialEquationAnalysis` */
    X[1] = -Q(0.7415311855993944398638647732807884); /* GaussianQuadratureWeights[7, -1, 1, 37]   */
    X[2] = -Q(0.4058451513773971669066064120769615);
    X[3] =  Q(0.0);
    X[4] = -X[2];
    X[5] = -X[1];
    X[6] = -X[0];
    W[0] =  Q(0.129484966168869693270611432679082018);
    W[1] =  Q(0.279705391489276667901467771423779582);
    W[2] =  Q(0.381830050505118944950369775488975134);
    W[3] =  Q(0.417959183673469387755102040816326531);
    W[4] =  W[2];
    W[5] =  W[1];
    W[6] =  W[0];
    break;
  case (8): /* p = 15 */
    X[0] = -Q(0.9602898564975362316835608685694730); /* << NumericalDifferentialEquationAnalysis` */
    X[1] = -Q(0.7966664774136267395915539364758304); /* GaussianQuadratureWeights[8, -1, 1, 37]   */
    X[2] = -Q(0.5255324099163289858177390491892463);
    X[3] = -Q(0.1834346424956498049394761423601840);
    X[4] = -X[3];
    X[5] = -X[2];
    X[6] = -X[1];
    X[7] = -X[0];
    W[0] =  Q(0.101228536290376259152531354309962190);
    W[1] =  Q(0.222381034453374470544355994426240884);
    W[2] =  Q(0.313706645877887287337962201986601313);
    W[3] =  Q(0.362683783378361982965150449277195612);
    W[4] =  W[3];
    W[5] =  W[2];
    W[6] =  W[1];
    W[7] =  W[0];
    break;
  case (9): /* p = 17 */
    X[0] = -Q(0.9681602395076260898355762029036729); /* << NumericalDifferentialEquationAnalysis` */
    X[1] = -Q(0.8360311073266357942994297880697349); /* GaussianQuadratureWeights[9, -1, 1, 37]   */
    X[2] = -Q(0.6133714327005903973087020393414742);
    X[3] = -Q(0.3242534234038089290385380146433366);
    X[4] =  Q(0.0);
    X[5] = -X[3];
    X[6] = -X[2];
    X[7] = -X[1];
    X[8] = -X[0];
    W[0] =  Q(0.081274388361574411971892158110523651);
    W[1] =  Q(0.180648160694857404058472031242912810);
    W[2] =  Q(0.260610696402935462318742869418632850);
    W[3] =  Q(0.312347077040002840068630406584443666);
    W[4] =  Q(0.330239355001259763164525069286974049);
    W[5] =  W[3];
    W[6] =  W[2];
    W[7] =  W[1];
    W[8] =  W[0];
    break;
  default:
    return -1;
  }
  return 0;
}

static PetscErrorCode IGA_Rule_GaussLobatto(PetscInt q,PetscReal X[],PetscReal W[])
{
  switch (q)  {
  case (2): /* p = 1 */
    X[0] = -Q(1.0);
    X[1] = -X[0];
    W[0] =  Q(1.0);
    W[1] =  W[0];
    break;
  case (3): /* p = 3 */
    X[0] = -Q(1.0);
    X[1] =  Q(0.0);
    X[2] = -X[0];
    W[0] =  Q(0.333333333333333333333333333333333333); /* 1/3 */
    W[1] =  Q(1.333333333333333333333333333333333333); /* 4/3 */
    W[2] =  W[0];
    break;
  case (4): /* p = 5 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.447213595499957939281834733746255247); /* 1/sqrt(5) */
    X[2] = -X[1];
    X[3] = -X[0];
    W[0] =  Q(0.166666666666666666666666666666666667); /* 1/6 */
    W[1] =  Q(0.833333333333333333333333333333333333); /* 5/6 */
    W[2] =  W[1];
    W[3] =  W[0];
    break;
  case (5): /* p = 7 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.654653670707977143798292456246858356); /* sqrt(3/7) */
    X[2] =  Q(0.0);
    X[3] = -X[1];
    X[4] = -X[0];
    W[0] =  Q(0.100000000000000000000000000000000000); /*  1/10 */
    W[1] =  Q(0.544444444444444444444444444444444444); /* 49/90 */
    W[2] =  Q(0.711111111111111111111111111111111111); /* 32/45 */
    W[3] =  W[1];
    W[4] =  W[0];
    break;
  case (6): /* p = 9 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.765055323929464692851002973959338150); /* sqrt((7+2*sqrt(7))/21) */
    X[2] = -Q(0.285231516480645096314150994040879072); /* sqrt((7-2*sqrt(7))/21) */
    X[3] = -X[2];
    X[4] = -X[1];
    X[5] = -X[0];
    W[0] =  Q(0.066666666666666666666666666666666667); /* 1/15 */
    W[1] =  Q(0.378474956297846980316612808212024652); /* (14-sqrt(7))/30 */
    W[2] =  Q(0.554858377035486353016720525121308681); /* (14+sqrt(7))/30 */
    W[3] =  W[2];
    W[4] =  W[1];
    W[5] =  W[0];
    break;
  case (7): /* p = 11 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.830223896278566929872032213967465140);
    X[2] = -Q(0.468848793470714213803771881908766329);
    X[3] =  Q(0.0);
    X[4] = -X[2];
    X[5] = -X[1];
    X[6] = -X[0];
    W[0] =  Q(0.047619047619047619047619047619047619);
    W[1] =  Q(0.276826047361565948010700406290066293);
    W[2] =  Q(0.431745381209862623417871022281362278);
    W[3] =  Q(0.487619047619047619047619047619047619);
    W[4] =  W[2]; 
    W[5] =  W[1];
    W[6] =  W[0];
    break;
  case (8): /* p = 13 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.871740148509606615337445761220663438);
    X[2] = -Q(0.591700181433142302144510731397953190);
    X[3] = -Q(0.209299217902478868768657260345351255);
    X[4] = -X[3];
    X[5] = -X[2];
    X[6] = -X[1];
    X[7] = -X[0];
    W[0] =  Q(0.035714285714285714285714285714285714);
    W[1] =  Q(0.210704227143506039382992065775756324);
    W[2] =  Q(0.341122692483504364764240677107748172);
    W[3] =  Q(0.412458794658703881567052971402209789);
    W[4] =  W[3];
    W[5] =  W[2];
    W[6] =  W[1];
    W[7] =  W[0];
    break;
  case (9): /* p = 15 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.899757995411460157312345244418337958);
    X[2] = -Q(0.677186279510737753445885427091342451);
    X[3] = -Q(0.363117463826178158710752068708659213);
    X[4] =  Q(0.0);
    X[5] = -X[3];
    X[6] = -X[2];
    X[7] = -X[1];
    X[8] = -X[0];
    W[0] =  Q(0.027777777777777777777777777777777778);
    W[1] =  Q(0.165495361560805525046339720029208306);
    W[2] =  Q(0.274538712500161735280705618579372726);
    W[3] =  Q(0.346428510973046345115131532139718288);
    W[4] =  Q(0.371519274376417233560090702947845805);
    W[5] =  W[3];
    W[6] =  W[2];
    W[7] =  W[1];
    W[8] =  W[0];
    break;
  default:
    return -1;
  }
  return 0;
}
