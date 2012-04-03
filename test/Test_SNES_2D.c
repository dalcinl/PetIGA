#include "petiga.h"

PetscScalar Peaks(PetscReal x, PetscReal y)
{
  PetscReal X = x*3;
  PetscReal Y = y*3;
  return   3 * pow(1-X,2) * exp(-pow(X,2) - pow(Y+1,2))
    /**/ - 10 * (X/5 - pow(X,3) - pow(Y,5)) * exp(-pow(X,2) - pow(Y,2))
    /**/ - 1.0/3 * exp(-pow(X+1,2) - pow(Y,2));
}

#undef  __FUNCT__
#define __FUNCT__ "Function"
PetscErrorCode Function(IGAPoint p,const PetscScalar *Ue,PetscScalar *Fe,void *ctx)
{
  PetscInt nen=p->nen;
  PetscInt dof=p->dof;
  PetscScalar (*F)[dof] = (PetscScalar (*)[dof])Fe;

  PetscReal xy[2];
  IGAPointGetPoint(p,xy);
  PetscReal x = xy[0];
  PetscReal y = xy[1];

  PetscScalar U0[dof],U1[dof][2];
  IGAPointGetValue(p,Ue,&U0[0]);
  IGAPointGetGrad (p,Ue,&U1[0][0]);
  PetscScalar PETSC_UNUSED u = U0[0], u_x = U1[0][0], u_y = U1[0][1];
  PetscScalar PETSC_UNUSED v = U0[1], v_x = U1[1][0], v_y = U1[1][1];
  PetscScalar PETSC_UNUSED w = U0[2], w_x = U1[2][0], w_y = U1[2][1];
  PetscScalar PETSC_UNUSED r = U0[3], r_x = U1[3][0], r_y = U1[3][1];

  PetscReal *N0 = p->shape[0];
  PetscReal (*N1)[2] = (PetscReal (*)[2]) p->shape[1];

  PetscInt a;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    F[a][0] = Na*u - Na * Peaks(x,y);
    F[a][1] = Na_x*v_x + Na_y*v_y - Na * 1.0;
    F[a][2] = Na*w + Na_x*w_x + Na_y*w_y - Na * 1.0;
    F[a][3] = Na_x*r_x + Na_y*r_y - Na * 1*exp(r);
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Jacobian"
PetscErrorCode Jacobian(IGAPoint p,const PetscScalar *Ue,PetscScalar *Je,void *ctx)
{
  PetscInt nen=p->nen;
  PetscInt dof=p->dof;
  PetscScalar (*J)[dof][nen][dof] = (PetscScalar (*)[dof][nen][dof])Je;

  PetscScalar U0[dof];
  IGAPointGetValue(p,Ue,&U0[0]);
  PetscScalar PETSC_UNUSED r = U0[3];

  PetscReal *N0 = p->shape[0];
  PetscReal (*N1)[2] = (PetscReal (*)[2]) p->shape[1];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    for (b=0; b<nen; b++) {
      PetscReal Nb   = N0[b];
      PetscReal Nb_x = N1[b][0];
      PetscReal Nb_y = N1[b][1];
      J[a][0][b][0] = Na*Nb;
      J[a][1][b][1] = Na_x*Nb_x + Na_y*Nb_y;
      J[a][2][b][2] = Na*Nb + Na_x*Nb_x + Na_y*Nb_y;
      J[a][3][b][3] = Na_x*Nb_x + Na_y*Nb_y - Na*Nb * 1*exp(r);
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscInt N=16, p=2, C=PETSC_DECIDE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","MultiComp2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","number of elements (along one dimension)",__FILE__,N,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","polynomial order",__FILE__,p,&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C","global continuity order",__FILE__,C,&C,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,4);CHKERRQ(ierr);

  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,N,-1.0,1.0,C);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  ierr = IGAAxisCopy(axis0,axis1);CHKERRQ(ierr);

  IGABoundary bnd;
  PetscInt dir,side;
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      PetscScalar value = 1.0;
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,1,value);CHKERRQ(ierr);
    }
  }
  for (side=0, dir=0; dir<2; dir++) {
    PetscScalar value = 0.0;
    ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetValue(bnd,2,value);CHKERRQ(ierr);
  }
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      PetscScalar value = 0.0;
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,3,value);CHKERRQ(ierr);
    }
  }

  ierr = IGASetUserFunction(iga,Function,0);CHKERRQ(ierr);
  ierr = IGASetUserJacobian(iga,Jacobian,0);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Vec x;
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  SNES snes;
  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,0,x);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  PetscBool flag = PETSC_FALSE;
  PetscReal secs = -1;
  ierr = PetscOptionsHasName(0,"-sleep",&flag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-sleep",&secs,0);CHKERRQ(ierr);
  if (flag) {ierr = PetscSleep(secs);CHKERRQ(ierr);}

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
