#include "petiga.h"

const char *const IGARuleTypes[] = {
  "LEGENDRE",
  "LOBATTO",
  "REDUCED",
  "USER",
  /* */
  "IGARuleType","IGA_RULE_",NULL};

PetscErrorCode IGARuleCreate(IGARule *_rule)
{
  IGARule        rule;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_rule,1);
  ierr = PetscCalloc1(1,&rule);CHKERRQ(ierr);
  *_rule = rule; rule->refct = 1;
  PetscFunctionReturn(0);
}

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

PetscErrorCode IGARuleReference(IGARule rule)
{
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  rule->refct++;
  PetscFunctionReturn(0);
}

PetscErrorCode IGARuleCopy(IGARule base,IGARule rule)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(base,1);
  PetscValidPointer(rule,2);
  if (base == rule) PetscFunctionReturn(0);
  ierr = IGARuleReset(rule);CHKERRQ(ierr);
  rule->type = base->type;
  rule->nqp  = base->nqp;
  if (base->nqp > 0 && base->point) {
    ierr = PetscMalloc1((size_t)base->nqp,&rule->point);CHKERRQ(ierr);
    ierr = PetscMemcpy(rule->point,base->point,(size_t)base->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (base->nqp > 0 && base->weight) {
    ierr = PetscMalloc1((size_t)base->nqp,&rule->weight);CHKERRQ(ierr);
    ierr = PetscMemcpy(rule->weight,base->weight,(size_t)base->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGARuleDuplicate(IGARule base,IGARule *rule)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(base,1);
  PetscValidPointer(rule,2);
  ierr = IGARuleCreate(rule);CHKERRQ(ierr);
  ierr = IGARuleCopy(base,*rule);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGARuleSetType(IGARule rule,IGARuleType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  if (rule->type == type) PetscFunctionReturn(0);
  rule->type = type;
  if (rule->nqp > 0) {ierr = IGARuleSetUp(rule);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode IGARuleSetSize(IGARule rule,PetscInt nqp)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  if (nqp < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of quadrature points must be greater than zero, got %D",nqp);
  if (rule->nqp == nqp) PetscFunctionReturn(0);
  ierr = IGARuleReset(rule);CHKERRQ(ierr);
  rule->nqp = nqp;
  ierr = IGARuleSetUp(rule);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IGA_Rule_GaussLegendre (PetscInt q,PetscReal X[],PetscReal W[]);
static PetscErrorCode IGA_Rule_GaussLobatto  (PetscInt q,PetscReal X[],PetscReal W[]);

PetscErrorCode IGARuleSetUp(IGARule rule)
{
  PetscErrorCode (*ComputeRule)(PetscInt,PetscReal[],PetscReal[]) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  if (rule->nqp < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGARuleSetSize() first");
  if (!rule->point)  {ierr = PetscCalloc1((size_t)rule->nqp,&rule->point);CHKERRQ(ierr);}
  if (!rule->weight) {ierr = PetscCalloc1((size_t)rule->nqp,&rule->weight);CHKERRQ(ierr);}
  switch (rule->type) {
  case IGA_RULE_LEGENDRE:
    ComputeRule = IGA_Rule_GaussLegendre; break;
  case IGA_RULE_LOBATTO:
    ComputeRule = IGA_Rule_GaussLobatto; break;
  case IGA_RULE_USER:
    ComputeRule = NULL; break;
  case IGA_RULE_REDUCED:
    ComputeRule = IGA_Rule_GaussLegendre; break;
  }
  if (ComputeRule && ComputeRule(rule->nqp,rule->point,rule->weight) != 0)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of quadrature points %D not implemented",rule->nqp);
  PetscFunctionReturn(0);
}

PetscErrorCode IGARuleSetRule(IGARule rule,PetscInt q,const PetscReal x[],const PetscReal w[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(rule,1);
  PetscValidPointer(x,3);
  PetscValidPointer(w,4);
  rule->type = IGA_RULE_USER;
  ierr = IGARuleSetSize(rule,q);CHKERRQ(ierr);
  ierr = PetscMemcpy(rule->point,x,(size_t)rule->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(rule->weight,w,(size_t)rule->nqp*sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  case (1): /* p <= 1 */
    X[0] =  Q(0.0);
    W[0] =  Q(2.0);
    break;
  case (2): /* p <= 3 */
    X[0] = -Q(0.577350269189625764509148780501957456); /* 1/sqrt(3) */
    X[1] = -X[0];
    W[0] =  Q(1.0);
    W[1] =  W[0];
    break;
  case (3): /* p <= 5 */
    X[0] = -Q(0.774596669241483377035853079956479922); /* sqrt(3/5) */
    X[1] =  Q(0.0);
    X[2] = -X[0];
    W[0] =  Q(0.555555555555555555555555555555555556); /* 5/9 */
    W[1] =  Q(0.888888888888888888888888888888888889); /* 8/9 */
    W[2] =  W[0];
    break;
  case (4): /* p <= 7 */
    X[0] = -Q(0.861136311594052575223946488892809505); /* sqrt((3+2*sqrt(6/5))/7) */
    X[1] = -Q(0.339981043584856264802665759103244687); /* sqrt((3-2*sqrt(6/5))/7) */
    X[2] = -X[1];
    X[3] = -X[0];
    W[0] =  Q(0.347854845137453857373063949221999407); /* (18-sqrt(30))/36 */
    W[1] =  Q(0.652145154862546142626936050778000593); /* (18+sqrt(30))/36 */
    W[2] =  W[1];
    W[3] =  W[0];
    break;
  case (5): /* p <= 9 */
    X[0] = -Q(0.906179845938663992797626878299392965); /* 1/3*sqrt(5+2*sqrt(10/7)) */
    X[1] = -Q(0.538469310105683091036314420700208805); /* 1/3*sqrt(5-2*sqrt(10/7)) */
    X[2] =  Q(0.0);
    X[3] = -X[1];
    X[4] = -X[0];
    W[0] =  Q(0.236926885056189087514264040719917363); /* (322-13*sqrt(70))/900 */
    W[1] =  Q(0.478628670499366468041291514835638193); /* (322+13*sqrt(70))/900 */
    W[2] =  Q(0.568888888888888888888888888888888889); /* 128/225 */
    W[3] =  W[1];
    W[4] =  W[0];
    break;
  case (6): /* p <= 11 */
    X[0] = -Q(0.932469514203152027812301554493994609);
    X[1] = -Q(0.661209386466264513661399595019905347);
    X[2] = -Q(0.238619186083196908630501721680711935);
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
  case (7): /* p <= 13 */
    X[0] = -Q(0.949107912342758524526189684047851262);
    X[1] = -Q(0.741531185599394439863864773280788407);
    X[2] = -Q(0.405845151377397166906606412076961463);
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
  case (8): /* p <= 15 */
    X[0] = -Q(0.960289856497536231683560868569472990);
    X[1] = -Q(0.796666477413626739591553936475830437);
    X[2] = -Q(0.525532409916328985817739049189246349);
    X[3] = -Q(0.183434642495649804939476142360183981);
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
  case (9): /* p <= 17 */
    X[0] = -Q(0.968160239507626089835576202903672870);
    X[1] = -Q(0.836031107326635794299429788069734877);
    X[2] = -Q(0.613371432700590397308702039341474185);
    X[3] = -Q(0.324253423403808929038538014643336609);
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
  case (10): /* p <= 19 */
    X[0] = -Q(0.973906528517171720077964012084452053);
    X[1] = -Q(0.865063366688984510732096688423493049);
    X[2] = -Q(0.679409568299024406234327365114873576);
    X[3] = -Q(0.433395394129247190799265943165784162);
    X[4] = -Q(0.148874338981631210884826001129719985);
    X[5] = -X[4];
    X[6] = -X[3];
    X[7] = -X[2];
    X[8] = -X[1];
    X[9] = -X[0];
    W[0] =  Q(0.066671344308688137593568809893331793);
    W[1] =  Q(0.149451349150580593145776339657697332);
    W[2] =  Q(0.219086362515982043995534934228163192);
    W[3] =  Q(0.269266719309996355091226921569469353);
    W[4] =  Q(0.295524224714752870173892994651338329);
    W[5] =  W[4];
    W[6] =  W[3];
    W[7] =  W[2];
    W[8] =  W[1];
    W[9] =  W[0];
    break;
  default:
    return -1;
  }
  return 0;
}

static PetscErrorCode IGA_Rule_GaussLobatto(PetscInt q,PetscReal X[],PetscReal W[])
{
  switch (q)  {
  case (2): /* p <= 1 */
    X[0] = -Q(1.0);
    X[1] = -X[0];
    W[0] =  Q(1.0);
    W[1] =  W[0];
    break;
  case (3): /* p <= 3 */
    X[0] = -Q(1.0);
    X[1] =  Q(0.0);
    X[2] = -X[0];
    W[0] =  Q(0.333333333333333333333333333333333333); /* 1/3 */
    W[1] =  Q(1.333333333333333333333333333333333333); /* 4/3 */
    W[2] =  W[0];
    break;
  case (4): /* p <= 5 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.447213595499957939281834733746255247); /* 1/sqrt(5) */
    X[2] = -X[1];
    X[3] = -X[0];
    W[0] =  Q(0.166666666666666666666666666666666667); /* 1/6 */
    W[1] =  Q(0.833333333333333333333333333333333333); /* 5/6 */
    W[2] =  W[1];
    W[3] =  W[0];
    break;
  case (5): /* p <= 7 */
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
  case (6): /* p <= 9 */
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
  case (7): /* p <= 11 */
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
  case (8): /* p <= 13 */
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
  case (9): /* p <= 15 */
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
  case (10): /* p <= 17 */
    X[0] = -Q(1.0);
    X[1] = -Q(0.919533908166458813828932660822338134);
    X[2] = -Q(0.738773865105505075003106174859830725);
    X[3] = -Q(0.477924949810444495661175092731257998);
    X[4] = -Q(0.165278957666387024626219765958173533);
    X[5] = -X[4];
    X[6] = -X[3];
    X[7] = -X[2];
    X[8] = -X[1];
    X[9] = -X[0];
    W[0] =  Q(0.022222222222222222222222222222222222);
    W[1] =  Q(0.133305990851070111126227170755392898);
    W[2] =  Q(0.224889342063126452119457821731047843);
    W[3] =  Q(0.292042683679683757875582257374443892);
    W[4] =  Q(0.327539761183897456656510527916893145);
    W[5] =  W[4];
    W[6] =  W[3];
    W[7] =  W[2];
    W[8] =  W[1];
    W[9] =  W[0];
    break;
  default:
    return -1;
  }
  return 0;
}
