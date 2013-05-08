/*
  keywords: steady, vector, linear
 */
#include "petiga.h"

typedef struct {
  PetscReal nu,E,t,k;
} AppCtx;

PetscErrorCode BetaRhoKappa(PetscReal N,PetscReal N_xi,PetscReal N_eta,
			    PetscReal A1,PetscReal A1_xi,PetscReal A1_eta,
			    PetscReal A2,PetscReal A2_xi,PetscReal A2_eta,
			    PetscReal b1, PetscReal b2,
			    PetscReal (*Beta)[3],PetscReal (*Rho)[2],PetscReal (*Kappa)[3])
{
  PetscFunctionBegin;

  Beta[0][0] = 1./A1*N_xi;
  Beta[0][1] = 1./A1/A2*A2_xi*N;
  Beta[0][2] = 0.5*(N_eta-A1_eta*N/A1)/A2;
  Beta[1][0] = 1./A1/A2*A1_eta*N;
  Beta[1][1] = 1./A2*N_eta;
  Beta[1][2] = 0.5*(1./A1*N_xi-1./A1/A2*A2_xi*N);
  Beta[2][0] = b1*N;
  Beta[2][1] = b2*N;
  Beta[2][2] = 0.;

  memset(Rho,0,10*sizeof(PetscReal));
  Rho[0][0] = b1*N;
  Rho[1][1] = b2*N;
  Rho[2][0] = -N_xi/A1;
  Rho[2][1] = -N_eta/A2;
  Rho[3][0] = N;
  Rho[4][1] = N;

  memset(Kappa,0,15*sizeof(PetscReal));
  Kappa[0][2] = 0.5*(-b1/A2*N_eta+b2/A1/A2*A1_eta*N);
  Kappa[1][2] = 0.5*(-b2/A1*N_xi+b1/A1/A2*A2_xi*N);
  Kappa[3][0] = 1./A1*N_xi;
  Kappa[3][1] = 1./A1/A2*A2_xi*N;
  Kappa[3][2] = 0.5*(1./A2*N_eta-1./A1/A2*A1_eta*N);
  Kappa[4][0] = 1./A1/A2*A1_eta*N;
  Kappa[4][1] = 1./A2*N_eta;
  Kappa[4][2] = 0.5*(1./A1*N_xi-1./A1/A2*A2_xi*N);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_Interpolate(PetscInt nen,PetscInt dof,PetscInt dim,PetscInt der,
                            const PetscReal N[],const PetscScalar U[],PetscScalar u[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGAShellInterpolate"
PetscErrorCode IGAShellInterpolate(IGAPoint point,PetscInt ider,PetscInt dof,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(U,2);
  PetscValidPointer(u,3);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  {
    PetscInt nen = point->nen;
    PetscInt dim = point->dim;
    PetscReal *N = point->basis[ider];
    IGA_Interpolate(nen,dof,dim,ider,N,U,u);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal nu = user->nu;
  PetscReal E  = user->E;
  PetscReal t  = user->t;

  // get geometry
  PetscScalar *geom = p->geometry;
  PetscScalar grad_g[3][2],hess_g[3][2][2];
  IGAShellInterpolate(p,1,3,geom,&grad_g[0][0]);
  IGAShellInterpolate(p,2,3,geom,&hess_g[0][0][0]);

  // compute unit normal
  PetscScalar n[3],rmagn;
  n[0] =  (grad_g[1][0]*grad_g[2][1]-grad_g[2][0]*grad_g[1][1]);
  n[1] = -(grad_g[0][0]*grad_g[2][1]-grad_g[2][0]*grad_g[0][1]);
  n[2] =  (grad_g[0][0]*grad_g[1][1]-grad_g[1][0]*grad_g[0][1]);
  rmagn = 1./sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
  n[0] *= rmagn; n[1] *= rmagn; n[2] *= rmagn;

  // compute curvatures
  PetscScalar A1,A2,b1,b2;
  A1 = sqrt(grad_g[0][0]*grad_g[0][0]+grad_g[1][0]*grad_g[1][0]+grad_g[2][0]*grad_g[2][0]);
  A2 = sqrt(grad_g[0][1]*grad_g[0][1]+grad_g[1][1]*grad_g[1][1]+grad_g[2][1]*grad_g[2][1]);
  b1 = -1./A1/A1 * (n[0]*hess_g[0][0][0]+n[1]*hess_g[1][0][0]+n[2]*hess_g[2][0][0]);
  b2 = -1./A2/A2 * (n[0]*hess_g[0][1][1]+n[1]*hess_g[1][1][1]+n[2]*hess_g[2][1][1]);

  PetscReal dA1[2]; dA1[0] = dA1[1] = 0.;
  PetscReal dA2[2]; dA2[0] = dA2[1] = 0.;

  // get basis functions 
  const PetscReal *N0         = p->basis[0];
  const PetscReal (*N1)[2]    = (const PetscReal (*)[2])(p->basis[1]);
  //const PetscReal (*N2)[2][2] = (const PetscReal (*)[2][2])(p->basis[2]);

  PetscInt    i,j;
  PetscInt a,b,nen=p->nen;
  PetscScalar (*KK)[5][nen][5] = (PetscScalar (*)[5][nen][5])K;
  for (a=0; a<nen; a++) {
    PetscReal Na     = N0[a];
    PetscReal Na_x   = N1[a][0];
    PetscReal Na_y   = N1[a][1];
    //PetscReal Na_xx  = N2[a][0][0];
    //PetscReal Na_xy  = N2[a][0][1];
    //PetscReal Na_yy  = N2[a][1][1];

    PetscReal dBeta[3][3];
    PetscReal dRho[5][2];
    PetscReal dKappa[5][3];
    BetaRhoKappa(Na,Na_x,Na_y,A1,dA1[0],dA1[1],A2,dA2[0],dA2[1],b1,b2,
		 dBeta,dRho,dKappa);

    for (b=0; b<nen; b++) {
      PetscReal Nb     = N0[b];
      PetscReal Nb_x   = N1[b][0];
      PetscReal Nb_y   = N1[b][1];
      //PetscReal Nb_xx  = N2[b][0][0];
      //PetscReal Nb_xy  = N2[b][0][1];
      //PetscReal Nb_yy  = N2[b][1][1];

      PetscReal Beta[3][3];
      PetscReal Rho[5][2];
      PetscReal Kappa[5][3];
      BetaRhoKappa(Nb,Nb_x,Nb_y,A1,dA1[0],dA1[1],A2,dA2[0],dA2[1],b1,b2,
		   Beta,Rho,Kappa);

      PetscScalar T[5][5];
      memset(T,0,sizeof(T));

      for(i=0;i<3;i++)
	for(j=0;j<3;j++)
	  T[i][j] += E*t/(1-nu*nu)*(nu*(dBeta[i][0]+dBeta[i][1])*(Beta[j][0]+Beta[j][1])
				    +(1-nu)*(dBeta[i][0]*Beta[j][0]+2.*dBeta[i][2]*Beta[j][2]+dBeta[i][1]*Beta[j][1]));       

      for(i=0;i<5;i++)
	for(j=0;j<5;j++)
	  T[i][j] += user->k*0.5*E*t/(1.0+nu)*(dRho[i][0]*Rho[j][0]+dRho[i][1]*Rho[j][1]);		

      for(i=0;i<5;i++)
	for(j=0;j<5;j++)
	  T[i][j] += E*t*t*t/12./(1-nu*nu)*(nu*(dKappa[i][0]+dKappa[i][1])*(Kappa[j][0]+Kappa[j][1])
					    +(1-nu)*(dKappa[i][0]*Kappa[j][0]+2.*dKappa[i][2]*Kappa[j][2]+dKappa[i][1]*Kappa[j][1])); 

      for (i=0;i<5;i++)
        for (j=0;j<5;j++) 
          KK[a][i][b][j] += T[i][j]*A1*A2;
	
    }
  }

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.nu = 0.3;
  user.E  = 3.e7;
  user.t  = 1.;
  user.k  = 5./6.;

  char filename[PETSC_MAX_PATH_LEN] = "ClassicalShell.dat";
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetGeometryDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetDof(iga,5);CHKERRQ(ierr); // dofs = {ux,uy,uz,psix,psiy}
  ierr = IGARead(iga,filename);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  IGABoundary bnd;
  // Boundary conditions on u = 0, v = [0:1]
  ierr = IGAGetBoundary(iga,0,0,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,2,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,4,0.0);CHKERRQ(ierr);
  // Boundary conditions on u = 1, v = [0:1]
  ierr = IGAGetBoundary(iga,0,1,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,3,0.0);CHKERRQ(ierr);
  // Boundary conditions on u = [0:1], v = 0
  ierr = IGAGetBoundary(iga,1,0,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,4,0.0);CHKERRQ(ierr);
  // Boundary conditions on u = [0:1], v = 1
  ierr = IGAGetBoundary(iga,1,1,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,4,0.0);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,&user);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  //MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);
  //PetscBool symm=1;
  //ierr = MatIsSymmetric(A,1e-2,&symm);CHKERRQ(ierr);
  //printf("symm=%d\n",(int)symm);
  
  PetscInt fdof;
  VecGetSize(b,&fdof);
  fdof -= 3;
  VecSetValue(b,fdof,-0.25,INSERT_VALUES);
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  int rank,size;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if(rank == size-1){ 
    PetscScalar sol;
    ierr = VecGetValues(x,1,&fdof,&sol);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"x[%d]=%g\n",fdof,sol);
  }
  ierr = IGAWriteVec(iga,x,"ClassicalShell.out");CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  PetscBool flag = PETSC_FALSE;
  PetscReal secs = -1;
  ierr = PetscOptionsHasName(0,"-draw",&flag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-draw",&secs,0);CHKERRQ(ierr);
  if (flag) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = PetscSleep(secs);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
