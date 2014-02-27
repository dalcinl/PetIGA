#include "petiga.h"

/*
#define hypot(x,y) sqrt((x)*(x)+(y)*(y))
*/

#define AssertEQUAL(a,b)                            \
  do {                                              \
    double _a = (a), _b = (b);                      \
    if (fabs(_a-_b) > 1e-6)                         \
      SETERRQ2(PETSC_COMM_SELF,1,"%f != %f",_a,_b); \
  } while(0)

static PetscReal PX[3][3] = {
  { 1.0, 1.0, 0.0 },
  { 1.5, 1.5, 0.0 },
  { 2.0, 2.0, 0.0 },
};
static PetscReal PY[3][3] = {
  { 0.0, 1.0, 1.0 },
  { 0.0, 1.5, 1.5 },
  { 0.0, 2.0, 2.0 },
};
static PetscReal PW[3][3] = {
#define sqrt2 1.4142135623730951
  { 1., sqrt2/2., 1. },
  { 1., sqrt2/2., 1. },
  { 1., sqrt2/2., 1. },
#undef  sqrt2
};

#undef  __FUNCT__
#define __FUNCT__ "TestGeometryMap"
PetscErrorCode TestGeometryMap(IGAPoint p)
{
  PetscInt dim = p->dim;
  PetscReal u = p->point[0];
  PetscReal v = p->point[1];
  PetscReal w = (dim==3)?p->point[2]:0.;
  PetscReal X[3] = {0.,0.,0.};
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAPointFormGeomMap(p,&X[0]);CHKERRQ(ierr);
  {
    PetscReal xw = (1+u)*(v*v*(-1+sqrt(2)) + v*(-sqrt(2)+2) - 1);
    PetscReal yw = (1+u)*(v*v*(-1+sqrt(2)) - v*sqrt(2));
    PetscReal ww = v*v*(-2+sqrt(2)) + v*(-sqrt(2)+2) - 1;
    PetscReal x = xw / ww;
    PetscReal y = yw / ww;
    PetscReal z = 2*w;
    AssertEQUAL(X[0], x);
    AssertEQUAL(X[1], y);
    AssertEQUAL(X[2], z);
  }
  {
    PetscReal detX = p->detX[0];
    PetscReal Jw = sqrt(2)*(1+u);
    PetscReal ww = (2-sqrt(2))*v*v + (-2+sqrt(2))*v + 1;
    PetscReal J = Jw/ww;
    if (dim==3) J *= 2;
    AssertEQUAL(detX, J);
  }
  if (dim==2) {
    PetscReal F[2][2],ww,F00,F01,F10,F11;
    ierr = PetscMemcpy(&F[0][0],p->gradX[0],sizeof(F));CHKERRQ(ierr);
    ww  = (v*v*(-2+sqrt(2))+v*(-sqrt(2)+2)-1);
    F00 = (v*v*(-1+sqrt(2))+v*(-sqrt(2)+2)-1)/ww;
    F01 = (-v*(u + 1)*(-2*v + sqrt(2)*v + 2))/(ww*ww);
    F10 = (v*v*(-1+sqrt(2))-v*sqrt(2))/ww;
    F11 = ((u+1)*(v-1)*(-2*v+sqrt(2)*v-sqrt(2)))/(ww*ww);
    AssertEQUAL(F[0][0], F00);
    AssertEQUAL(F[0][1], F01);
    AssertEQUAL(F[1][0], F10);
    AssertEQUAL(F[1][1], F11);
  }
  if (dim==3) {
    PetscReal F[3][3],ww,F00,F01,F10,F11;
    ierr = PetscMemcpy(&F[0][0],p->gradX[0],sizeof(F));CHKERRQ(ierr);
    ww  = (v*v*(-2+sqrt(2))+v*(-sqrt(2)+2)-1);
    F00 = (v*v*(-1+sqrt(2))+v*(-sqrt(2)+2)-1)/ww;
    F01 = (-v*(u + 1)*(-2*v + sqrt(2)*v + 2))/(ww*ww);
    F10 = (v*v*(-1+sqrt(2))-v*sqrt(2))/ww;
    F11 = ((u+1)*(v-1)*(-2*v+sqrt(2)*v-sqrt(2)))/(ww*ww);
    AssertEQUAL(F[0][0], F00);
    AssertEQUAL(F[0][1], F01);
    AssertEQUAL(F[0][2], 0.0);
    AssertEQUAL(F[1][0], F10);
    AssertEQUAL(F[1][1], F11);
    AssertEQUAL(F[2][0], 0.0);
    AssertEQUAL(F[2][1], 0.0);
    AssertEQUAL(F[2][2], 2.0);
  }
  if (dim==2) {
    PetscInt  a,i,j,k;
    PetscReal G[2][2];
    PetscReal H[2][2][2];
    PetscReal D[2][2][2][2];
    ierr = PetscMemcpy(&G[0][0],      p->gradX[0],sizeof(G));CHKERRQ(ierr);
    ierr = PetscMemcpy(&H[0][0][0],   p->hessX[0],sizeof(H));CHKERRQ(ierr);
    ierr = PetscMemcpy(&D[0][0][0][0],p->der3X[0],sizeof(D));CHKERRQ(ierr);
    for (a=0;a<dim;a++)
      for (i=0;i<dim;i++)
        for (j=0;j<dim;j++)
          {
            AssertEQUAL(H[a][i][j], H[a][j][i]);
          }
    AssertEQUAL(H[0][0][0], 0.0);
    AssertEQUAL(H[1][0][0], 0.0);
    for (a=0;a<dim;a++)
      for (i=0;i<dim;i++)
        for (j=0;j<dim;j++)
          for (k=0;k<dim;k++)
            {
              AssertEQUAL(D[a][i][j][k], D[a][j][i][k]);
              AssertEQUAL(D[a][i][j][k], D[a][i][k][j]);
              AssertEQUAL(D[a][i][j][k], D[a][k][j][i]);
            }
    {
      PetscReal x  = X[0];
      PetscReal y  = X[1];
      PetscReal x1 = G[0][1];
      PetscReal x2 = H[0][1][1];
      PetscReal y1 = G[1][1];
      PetscReal y2 = H[1][1][1];
      PetscReal radius = hypot(x,y);
      PetscReal kappa  = (x1*y2-y1*x2)/pow(x1*x1+y1*y1,3./2);
      AssertEQUAL(kappa, 1/radius);
    }
  }
  if (dim==3) {
    PetscInt a,i,j,k;
    PetscReal G[3][3];
    PetscReal H[3][3][3];
    PetscReal D[3][3][3][3];
    ierr = PetscMemcpy(&G[0][0],      p->gradX[0],sizeof(G));CHKERRQ(ierr);
    ierr = PetscMemcpy(&H[0][0][0],   p->hessX[0],sizeof(H));CHKERRQ(ierr);
    ierr = PetscMemcpy(&D[0][0][0][0],p->der3X[0],sizeof(D));CHKERRQ(ierr);
    for (a=0;a<dim;a++)
      for (i=0;i<dim;i++)
        for (j=0;j<dim;j++)
          {
            AssertEQUAL(H[a][i][j], H[a][j][i]);
          }
    AssertEQUAL(H[0][0][0], 0.0);
    AssertEQUAL(H[1][0][0], 0.0);
    AssertEQUAL(H[2][0][0], 0.0);
    AssertEQUAL(H[2][1][1], 0.0);
    AssertEQUAL(H[0][2][2], 0.0);
    AssertEQUAL(H[1][2][2], 0.0);
    AssertEQUAL(H[2][2][2], 0.0);
    for (a=0;a<dim;a++)
      for (i=0;i<dim;i++)
        for (j=0;j<dim;j++)
          for (k=0;k<dim;k++)
            {
              AssertEQUAL(D[a][i][j][k], D[a][j][i][k]);
              AssertEQUAL(D[a][i][j][k], D[a][i][k][j]);
              AssertEQUAL(D[a][i][j][k], D[a][k][j][i]);
            }
    {
      PetscReal x  = X[0];
      PetscReal y  = X[1];
      PetscReal x1 = G[0][1];
      PetscReal x2 = H[0][1][1];
      PetscReal y1 = G[1][1];
      PetscReal y2 = H[1][1][1];
      PetscReal radius = hypot(x,y);
      PetscReal kappa  = (x1*y2-y1*x2)/pow(x1*x1+y1*y1,3./2);
      AssertEQUAL(kappa, 1/radius);
    }
  }
  if (dim==2) {
    PetscInt  nen = p->nen;
    PetscInt  a,l,i,j,k;
    PetscReal (*C)[2] = (PetscReal(*)[2]) p->geometry;
    PetscReal (*N1)[2] = (PetscReal(*)[2]) p->shape[1];
    PetscReal (*N2)[2][2] = (PetscReal(*)[2][2]) p->shape[2];
    PetscReal (*N3)[2][2][2] = (PetscReal(*)[2][2][2]) p->shape[3];
    PetscReal G[2][2],H[2][2][2],D[2][2][2][2];
    ierr = PetscMemzero(&G[0][0],sizeof(G));CHKERRQ(ierr);
    ierr = PetscMemzero(&H[0][0][0],sizeof(H));CHKERRQ(ierr);
    ierr = PetscMemzero(&D[0][0][0][0],sizeof(D));CHKERRQ(ierr);
    /* */
    for (a=0;a<nen;a++)
      for (l=0;l<dim;l++)
        for (i=0;i<dim;i++)
          G[l][i] += C[a][l]*N1[a][i];
    for (a=0;a<nen;a++)
      for (l=0;l<dim;l++)
        for (i=0;i<dim;i++)
          for (j=0;j<dim;j++)
            H[l][i][j] += C[a][l]*N2[a][i][j];
    for (a=0;a<nen;a++)
      for (l=0;l<dim;l++)
        for (i=0;i<dim;i++)
          for (j=0;j<dim;j++)
            for (k=0;k<dim;k++)
              D[l][i][j][k] += C[a][l]*N3[a][i][j][k];
    /* */
    for (i=0;i<dim;i++)
      for (j=0;j<dim;j++)
        if (i==j)
          AssertEQUAL(G[i][j], 1.0);
        else
          AssertEQUAL(G[i][j], 0.0);
    for (i=0;i<dim;i++)
      for (j=0;j<dim;j++)
        for (k=0;k<dim;k++)
          AssertEQUAL(H[i][j][k], 0.0);
    for (i=0;i<dim;i++)
      for (j=0;j<dim;j++)
        for (k=0;k<dim;k++)
          for (l=0;l<dim;l++)
            AssertEQUAL(D[i][j][k][l], 0.0);
  }
  if (dim==3) {
    PetscInt  nen = p->nen;
    PetscInt  a,l,i,j,k;
    PetscReal (*C)[3] = (PetscReal(*)[3]) p->geometry;
    PetscReal (*N1)[3] = (PetscReal(*)[3]) p->shape[1];
    PetscReal (*N2)[3][3] = (PetscReal(*)[3][3]) p->shape[2];
    PetscReal (*N3)[3][3][3] = (PetscReal(*)[3][3][3]) p->shape[3];
    PetscReal G[3][3],H[3][3][3],D[3][3][3][3];
    ierr = PetscMemzero(&G[0][0],sizeof(G));CHKERRQ(ierr);
    ierr = PetscMemzero(&H[0][0][0],sizeof(H));CHKERRQ(ierr);
    ierr = PetscMemzero(&D[0][0][0][0],sizeof(D));CHKERRQ(ierr);
    /* */
    for (a=0;a<nen;a++)
      for (l=0;l<dim;l++)
        for (i=0;i<dim;i++)
          G[l][i] += C[a][l]*N1[a][i];
    for (a=0;a<nen;a++)
      for (l=0;l<dim;l++)
        for (i=0;i<dim;i++)
          for (j=0;j<dim;j++)
            H[l][i][j] += C[a][l]*N2[a][i][j];
    for (a=0;a<nen;a++)
      for (l=0;l<dim;l++)
        for (i=0;i<dim;i++)
          for (j=0;j<dim;j++)
            for (k=0;k<dim;k++)
              D[l][i][j][k] += C[a][l]*N3[a][i][j][k];
    /* */
    for (i=0;i<dim;i++)
      for (j=0;j<dim;j++)
        if (i==j) AssertEQUAL(G[i][j], 1.0);
        else      AssertEQUAL(G[i][j], 0.0);
    for (i=0;i<dim;i++)
      for (j=0;j<dim;j++)
        for (k=0;k<dim;k++)
          AssertEQUAL(H[i][j][k], 0.0);
    for (i=0;i<dim;i++)
      for (j=0;j<dim;j++)
        for (k=0;k<dim;k++)
          for (l=0;l<dim;l++)
            AssertEQUAL(D[i][j][k][l], 0.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Domain"
PetscErrorCode Domain(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TestGeometryMap(p);CHKERRQ(ierr);
  {
    PetscInt i,dim = p->dim;
    PetscReal dS = p->detS[0];
    PetscReal *n = p->normal;
    AssertEQUAL(dS, 0.0);
    for (i=0;i<dim;i++) AssertEQUAL(n[i], 0.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Boundary_00"
PetscErrorCode Boundary_00(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  PetscReal      R   = 1.0;
  PetscInt       dim = p->dim;
  PetscReal      X[3];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TestGeometryMap(p);CHKERRQ(ierr);
  ierr = IGAPointFormGeomMap(p,&X[0]);CHKERRQ(ierr);
  {
    PetscReal x = X[0];
    PetscReal y = X[1];
    PetscReal r = hypot(x,y);
    PetscReal *n = p->normal;
    AssertEQUAL(r, R);
    AssertEQUAL(n[0], -x/r);
    AssertEQUAL(n[1], -y/r);
    if (dim==3) AssertEQUAL(n[2], 0.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Boundary_01"
PetscErrorCode Boundary_01(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  PetscReal      R   = 2.0;
  PetscInt       dim = p->dim;
  PetscReal      X[3];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TestGeometryMap(p);CHKERRQ(ierr);
  ierr = IGAPointFormGeomMap(p,&X[0]);CHKERRQ(ierr);
  {
    PetscReal x = X[0];
    PetscReal y = X[1];
    PetscReal r = hypot(x,y);
    PetscReal *n = p->normal;
    AssertEQUAL(r, R);
    AssertEQUAL(n[0], +x/r);
    AssertEQUAL(n[1], +y/r);
    if (dim==3) AssertEQUAL(n[2], 0.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Boundary_10"
PetscErrorCode Boundary_10(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TestGeometryMap(p);CHKERRQ(ierr);
  {
    PetscInt dim = p->dim;
    PetscReal dS = p->detS[0];
    PetscReal *n = p->normal;
    if (dim==2) AssertEQUAL(dS, 1.0);
    if (dim==3) AssertEQUAL(dS, 2.0);
    AssertEQUAL(n[0],  0.0);
    AssertEQUAL(n[1], -1.0);
    if (dim==3) AssertEQUAL(n[2], 0.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Boundary_11"
PetscErrorCode Boundary_11(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TestGeometryMap(p);CHKERRQ(ierr);
  {
    PetscInt dim = p->dim;
    PetscReal dS = p->detS[0];
    PetscReal *n = p->normal;
    if (dim==2) AssertEQUAL(dS, 1.0);
    if (dim==3) AssertEQUAL(dS, 2.0);
    AssertEQUAL(n[0], -1.0);
    AssertEQUAL(n[1],  0.0);
    if (dim==3) AssertEQUAL(n[2], 0.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Boundary_20"
PetscErrorCode Boundary_20(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TestGeometryMap(p);CHKERRQ(ierr);
  {
    PetscInt dim = p->dim;
    PetscReal dV = p->detX[0];
    PetscReal dS = p->detS[0];
    PetscReal *n = p->normal;
    AssertEQUAL(dS, dV/(dim-1));
    AssertEQUAL(n[0],  0.0);
    AssertEQUAL(n[1],  0.0);
    AssertEQUAL(n[2], -1.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Boundary_21"
PetscErrorCode Boundary_21(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TestGeometryMap(p);CHKERRQ(ierr);
  {
    PetscInt dim = p->dim;
    PetscReal dV = p->detX[0];
    PetscReal dS = p->detS[0];
    PetscReal *n = p->normal;
    AssertEQUAL(dS, dV/(dim-1));
    AssertEQUAL(n[0],  0.0);
    AssertEQUAL(n[1],  0.0);
    AssertEQUAL(n[2], +1.0);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Boundary"
PetscErrorCode Boundary(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  static PetscErrorCode (*boundary[3][2])(IGAPoint,PetscScalar*,PetscScalar*,void*)
    = {{Boundary_00, Boundary_01},
       {Boundary_10, Boundary_11},
       {Boundary_20, Boundary_21}};
  PetscInt d,s;
  d = p->boundary_id / 2;
  s = p->boundary_id % 2;
  return boundary[d][s](p,A,b,ctx);
}

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *A,PetscScalar *b,void *ctx)
{
  if (p->atboundary)
    return Boundary(p,A,b,ctx);
  else
    return Domain(p,A,b,ctx);
}


#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  S[0] = 1.0;
  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscInt        dim = 3;
  IGA             iga;
  IGAAxis         axis;
  IGAForm         form;
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","GeometryMap Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Number of space dimensions",__FILE__,dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (dim < 2) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE, "Problem requires dim={2,3}, not %D",dim);

  ierr = IGACreate(PETSC_COMM_SELF,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);

  ierr = IGAGetAxis(iga,0,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,2);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,1,0.0,1.0,0);CHKERRQ(ierr);
  ierr = IGAGetAxis(iga,1,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,2);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,1,0.0,1.0,1);CHKERRQ(ierr);
  ierr = IGAGetAxis(iga,2,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,1);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,1,0.0,1.0,0);CHKERRQ(ierr);

  ierr = IGASetQuadrature(iga,0,7);CHKERRQ(ierr);
  ierr = IGASetQuadrature(iga,1,9);CHKERRQ(ierr);
  ierr = IGASetQuadrature(iga,2,8);CHKERRQ(ierr);

  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetOrder(iga,3);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  iga->rational = PETSC_TRUE;
  ierr = IGASetGeometryDim(iga,dim);CHKERRQ(ierr);
  ierr = PetscMalloc(3*3*(dim-1)*sizeof(PetscReal),&iga->rationalW);CHKERRQ(ierr);
  ierr = PetscMalloc(3*3*(dim-1)*dim*sizeof(PetscReal),&iga->geometryX);CHKERRQ(ierr);
  {
    PetscInt i,j,k,m=dim-1;
    PetscInt posx=0,posw=0;
    PetscReal *W = iga->rationalW;
    PetscReal *X = iga->geometryX;
    for (k=0;k<m;k++) {
      for (j=0;j<3;j++) {
        for (i=0;i<3;i++) {
          W[posw++] = PW[i][j];
          X[posx++] = PX[i][j];
          X[posx++] = PY[i][j];
          if (dim==3)
            X[posx++] = (PetscReal)2*k;
        }
      }
    }
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGAGetForm(iga,&form);CHKERRQ(ierr);
  ierr = IGAFormSetSystem(form,System,NULL);CHKERRQ(ierr);
  ierr = IGAFormSetBoundaryForm(form,0,0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAFormSetBoundaryForm(form,1,0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAFormSetBoundaryForm(form,2,0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAFormSetBoundaryForm(form,0,1,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAFormSetBoundaryForm(form,1,1,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAFormSetBoundaryForm(form,2,1,PETSC_TRUE);CHKERRQ(ierr);

  {
    Mat A;
    Vec b;
    ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
    ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }
  {
    PetscReal   h  = (dim==2)?1.0:2.0;
    PetscReal   pi = M_PI;
    PetscReal   Ri = 1.0;
    PetscReal   Ro = 2.0;
    PetscReal   V  = h*pi*(Ro*Ro-Ri*Ri)/4;
    PetscScalar vol;
    Vec x;
    ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
    ierr = IGAComputeScalar(iga,x,1,&vol,Error,NULL);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    AssertEQUAL(V, PetscRealPart(vol));
  }

  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
