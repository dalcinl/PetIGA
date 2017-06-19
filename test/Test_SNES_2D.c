#include "petiga.h"

PetscScalar Peaks(PetscReal x, PetscReal y)
{
  PetscReal X = x*3;
  PetscReal Y = y*3;
  return   3 * pow(1-X,2) * exp(-pow(X,2) - pow(Y+1,2))
    /**/ - 10 * (X/5 - pow(X,3) - pow(Y,5)) * exp(-pow(X,2) - pow(Y,2))
    /**/ - 1.0/3 * exp(-pow(X+1,2) - pow(Y,2));
}

PetscErrorCode Function(IGAPoint p,const PetscScalar *Ue,PetscScalar *Fe,void *ctx)
{
  PetscInt nen=p->nen;

  PetscReal xy[2];
  IGAPointFormGeomMap(p,xy);
  PetscReal x = xy[0];
  PetscReal y = xy[1];

  PetscScalar u0[4],u1[4][2];
  IGAPointFormValue(p,Ue,&u0[0]);
  IGAPointFormGrad (p,Ue,&u1[0][0]);
  PetscScalar PETSC_UNUSED u = u0[0], u_x = u1[0][0], u_y = u1[0][1];
  PetscScalar PETSC_UNUSED v = u0[1], v_x = u1[1][0], v_y = u1[1][1];
  PetscScalar PETSC_UNUSED w = u0[2], w_x = u1[2][0], w_y = u1[2][1];
  PetscScalar PETSC_UNUSED r = u0[3], r_x = u1[3][0], r_y = u1[3][1];

  PetscReal *N0 = p->shape[0];
  PetscReal (*N1)[2] = (PetscReal(*)[2])p->shape[1];

  PetscInt a;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    Fe[a*4+0] = Na*u - Na * Peaks(x,y);
    Fe[a*4+1] = Na_x*v_x + Na_y*v_y - Na * 1.0;
    Fe[a*4+2] = Na*w + Na_x*w_x + Na_y*w_y - Na * 1.0;
    Fe[a*4+3] = Na_x*r_x + Na_y*r_y - Na * 1.0*exp(r);
  }
  return 0;
}

PetscErrorCode Jacobian(IGAPoint p,const PetscScalar *Ue,PetscScalar *Je,void *ctx)
{
  PetscInt nen=p->nen;

  PetscScalar u0[4];
  IGAPointFormValue(p,Ue,&u0[0]);
  PetscScalar PETSC_UNUSED r = u0[3];

  PetscReal *N0 = p->shape[0];
  PetscReal (*N1)[2] = (PetscReal(*)[2])p->shape[1];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    for (b=0; b<nen; b++) {
      PetscReal Nb   = N0[b];
      PetscReal Nb_x = N1[b][0];
      PetscReal Nb_y = N1[b][1];
      Je[a*nen*16+0*nen*4+b*4+0] = Na*Nb;
      Je[a*nen*16+1*nen*4+b*4+1] = Na_x*Nb_x + Na_y*Nb_y;
      Je[a*nen*16+2*nen*4+b*4+2] = Na*Nb + Na_x*Nb_x + Na_y*Nb_y;
      Je[a*nen*16+3*nen*4+b*4+3] = Na_x*Nb_x + Na_y*Nb_y - Na*Nb * 1.0*exp(r);
    }
  }
  return 0;
}

PetscErrorCode FunctionCollocation(IGAPoint p,const PetscScalar *Ue,PetscScalar *Fe,void *ctx)
{
  PetscReal xy[2];
  IGAPointFormGeomMap(p,xy);
  PetscReal x = xy[0];
  PetscReal y = xy[1];

  PetscScalar u0[4],u1[4][2],u2[4][2][2];
  IGAPointFormValue(p,Ue,&u0[0]);
  IGAPointFormGrad (p,Ue,&u1[0][0]);
  IGAPointFormHess (p,Ue,&u2[0][0][0]);

  PetscScalar PETSC_UNUSED u = u0[0];
  PetscScalar PETSC_UNUSED v = u0[1];
  PetscScalar PETSC_UNUSED w = u0[2];
  PetscScalar PETSC_UNUSED r = u0[3];

  PetscScalar PETSC_UNUSED u_x = u1[0][0], u_y = u1[0][1];
  PetscScalar PETSC_UNUSED v_x = u1[1][0], v_y = u1[1][1];
  PetscScalar PETSC_UNUSED w_x = u1[2][0], w_y = u1[2][1];
  PetscScalar PETSC_UNUSED r_x = u1[3][0], r_y = u1[3][1];

  PetscScalar PETSC_UNUSED u_xx = u2[0][0][0], u_yy = u2[0][1][1];
  PetscScalar PETSC_UNUSED v_xx = u2[1][0][0], v_yy = u2[1][1][1];
  PetscScalar PETSC_UNUSED w_xx = u2[2][0][0], w_yy = u2[2][1][1];
  PetscScalar PETSC_UNUSED r_xx = u2[3][0][0], r_yy = u2[3][1][1];

  Fe[0] = u - Peaks(x,y);
  Fe[1] = -(v_xx + v_yy) - 1.0;
  Fe[2] = -(w_xx + w_yy) + w - 1.0;
  Fe[3] = -(r_xx + r_yy) - 1.0*exp(r);
  return 0;
}

PetscErrorCode JacobianCollocation(IGAPoint p,const PetscScalar *Ue,PetscScalar *Je,void *ctx)
{
  PetscInt nen = p->nen;

  PetscScalar u0[4];
  IGAPointFormValue(p,Ue,&u0[0]);
  PetscScalar r = u0[3];

  PetscReal *N0 = p->shape[0];
  PetscReal (*N2)[2][2] = (PetscReal(*)[2][2])p->shape[2];

  PetscInt a;
  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_xx = N2[a][0][0];
    PetscReal Na_yy = N2[a][1][1];
    Je[0*nen*4+a*4+0] = Na;
    Je[1*nen*4+a*4+1] = -(Na_xx + Na_yy);
    Je[2*nen*4+a*4+2] = -(Na_xx + Na_yy) + Na;
    Je[3*nen*4+a*4+3] = -(Na_xx + Na_yy) - Na * 1*exp(r);
  }
  return 0;
}

int main(int argc, char *argv[])
{
  IGA             iga;
  IGAForm         form;
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","MultiComp2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-N","16","-iga_elements");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-p", "2","-iga_degree");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-C","-1","-iga_continuity");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-L","-1,+1","-iga_limits");CHKERRQ(ierr);

  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,4);CHKERRQ(ierr);

  ierr = IGASetFieldName(iga,0,"L2Projection");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"Poisson");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"ReactionDiffusion");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,3,"Bratu");CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGAGetForm(iga,&form);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGAFormSetFunction(form,Function,0);CHKERRQ(ierr);
    ierr = IGAFormSetJacobian(form,Jacobian,0);CHKERRQ(ierr);
  } else {
    ierr = IGAFormSetFunction(form,FunctionCollocation,0);CHKERRQ(ierr);
    ierr = IGAFormSetJacobian(form,JacobianCollocation,0);CHKERRQ(ierr);
  }
  PetscInt dir,side;
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      PetscInt    field = 1;
      PetscScalar value = 1.0;
      ierr = IGAFormSetBoundaryValue(form,dir,side,field,value);CHKERRQ(ierr);
    }
  }
  for (dir=0; dir<2; dir++) {
    PetscInt    field = 2;
    PetscScalar value = 0.0;
    ierr = IGAFormSetBoundaryValue(form,dir,side=0,field,value);CHKERRQ(ierr);
    ierr = IGAFormSetBoundaryValue(form,dir,side=1,field,value);CHKERRQ(ierr);
  }
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      PetscInt    field = 3;
      PetscScalar value = 0.0;
      ierr = IGAFormSetBoundaryValue(form,dir,side,field,value);CHKERRQ(ierr);
    }
  }

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

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
