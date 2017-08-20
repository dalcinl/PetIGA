#include "petiga.h"

PetscErrorCode Scalar(IGAPoint p,const PetscScalar U[],PetscInt n,PetscScalar *S,void *ctx)
{
  PetscInt i;
  for (i=0; i<n; i++) S[i] = (PetscScalar)1.0;
  return 0;
}

PetscErrorCode Vector(IGAPoint p,PetscScalar *F,void *ctx)
{
  PetscInt dof = p->dof;
  PetscInt nen = p->nen;
  PetscReal *N = p->shape[0];
  PetscInt a,i;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (i=0; i<dof; i++)
      F[a*dof+i] = Na * 1;
  }
  return 0;
}

PetscErrorCode Matrix(IGAPoint p,PetscScalar *K,void *ctx)
{
  PetscInt dof = p->dof;
  PetscInt nen = p->nen;
  PetscReal *N = p->shape[0];
  PetscInt a,b,i,j;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb = N[b];
      for (i=0; i<dof; i++)
        for (j=0; j<dof; j++)
          if (i==j)
            K[a*dof*nen*dof+i*nen*dof+b*dof+j] = Na*Nb;
    }
  }
  return 0;
}

PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt dof = p->dof;
  PetscInt nen = p->nen;
  PetscReal *N = p->shape[0];
  PetscInt a,b,i,j;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb = N[b];
      for (i=0; i<dof; i++)
        for (j=0; j<dof; j++)
          if (i==j)
            K[a*dof*nen*dof+i*nen*dof+b*dof+j] = Na*Nb;
    }
    for (i=0; i<dof; i++)
      F[a*dof+i] = Na * 1;
  }
  return 0;
}

int main(int argc, char *argv[])
{
  PetscInt       dim,dof;
  IGA            iga,giga;
  PetscScalar    s;
  PetscReal      xmin,xmax;
  Vec            b,x;
  Mat            A,B,C;
  KSP            ksp;
  SNES           snes;
  TS             ts;
  DM             dm;
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim < 1) {ierr = IGASetDim(iga,dim=3);CHKERRQ(ierr);}
  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
  if (dof < 1) {ierr = IGASetDof(iga,dof=1);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGAClone(iga,dim,&giga);CHKERRQ(ierr);
  ierr = IGADestroy(&giga);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGAComputeScalar(iga,x,1,&s,Scalar,NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatDuplicate(B,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  ierr = IGASetFormMatrix(iga,Matrix,NULL);CHKERRQ(ierr);
  ierr = IGAComputeMatrix(iga,A);CHKERRQ(ierr);
  ierr = IGAComputeMatrix(iga,B);CHKERRQ(ierr);
  ierr = IGAComputeMatrix(iga,C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecMin(x,NULL,&xmin);CHKERRQ(ierr);
  ierr = VecMax(x,NULL,&xmax);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  if ((xmax-xmin) > 1e-2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Unexpected result: x_min=%g x_max=%g\n",
                       (double)xmin,(double)xmax);CHKERRQ(ierr);
  }

  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = IGASetFormVector(iga,Vector,NULL);CHKERRQ(ierr);
  ierr = IGAComputeVector(iga,b);CHKERRQ(ierr);
  ierr = IGASetFormMatrix(iga,Matrix,NULL);CHKERRQ(ierr);
  ierr = IGAComputeMatrix(iga,A);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecMin(x,NULL,&xmin);CHKERRQ(ierr);
  ierr = VecMax(x,NULL,&xmax);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  if ((xmax-xmin) > 1e-2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Unexpected result: x_min=%g x_max=%g\n",
                       (double)xmin,(double)xmax);CHKERRQ(ierr);
  }

  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = IGACreateElemDM(iga,dof,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = IGACreateGeomDM(iga,dim,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = IGACreateNodeDM(iga,1,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = DMIGACreate(iga,&dm);CHKERRQ(ierr);
  ierr = DMIGAGetIGA(dm,&iga);CHKERRQ(ierr);
  ierr = DMIGASetIGA(dm,iga);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&b);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = DMIGACreate(iga,&dm);CHKERRQ(ierr);
  {
    DM newdm;
    ierr = DMClone(dm,&newdm);CHKERRQ(ierr);
    ierr = DMDestroy(&newdm);CHKERRQ(ierr);
  }
  {
    DM cdm;
    ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
  }
  {
    PetscInt i,j,fields[64];IS is;DM subdm;
    for (i=0; i<dof; i++) fields[i] = i;
    for (i=0; i<dof; i++) {
      ierr = DMCreateSubDM(dm,1,&i,&is,&subdm);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = DMDestroy(&subdm);CHKERRQ(ierr);
      ierr = DMCreateSubDM(dm,i+1,fields,&is,&subdm);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = DMDestroy(&subdm);CHKERRQ(ierr);
      ierr = DMCreateSubDM(dm,dof-i,fields+i,&is,&subdm);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = DMDestroy(&subdm);CHKERRQ(ierr);
      for (j=i+1; j<dof; j++) {
        ierr = DMCreateSubDM(dm,dof-j,fields+i,&is,&subdm);CHKERRQ(ierr);
        ierr = ISDestroy(&is);CHKERRQ(ierr);
        ierr = DMDestroy(&subdm);CHKERRQ(ierr);
      }
    }
  }
  {
    PetscInt i,len;char **namelist;IS *islist;
    ierr = DMCreateFieldIS(dm,&len,&namelist,&islist);CHKERRQ(ierr);
    for (i=0; i<len; i++) {
      ierr = PetscFree(namelist[i]);CHKERRQ(ierr);
      ierr = ISDestroy(&islist[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(namelist);CHKERRQ(ierr);
    ierr = PetscFree(islist);CHKERRQ(ierr);
  }
  {
    PetscInt i,len;char **namelist;IS *islist;DM *dmlist;
    ierr = DMCreateFieldDecomposition(dm,&len,&namelist,&islist,&dmlist);CHKERRQ(ierr);
    for (i=0; i<len; i++) {
      ierr = PetscFree(namelist[i]);CHKERRQ(ierr);
      ierr = ISDestroy(&islist[i]);CHKERRQ(ierr);
      ierr = DMDestroy(&dmlist[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(namelist);CHKERRQ(ierr);
    ierr = PetscFree(islist);CHKERRQ(ierr);
    ierr = PetscFree(dmlist);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
