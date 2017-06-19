/*
  keywords: steady, vector, linear
 */
#include "petiga.h"

typedef struct {
  PetscReal nu,E,t,k;
} AppCtx;

void BetaRhoKappa(PetscReal N, PetscReal N_xi, PetscReal N_eta,
                  PetscReal A1,PetscReal A1_xi,PetscReal A1_eta,
                  PetscReal A2,PetscReal A2_xi,PetscReal A2_eta,
                  PetscReal b1,PetscReal b2,
                  PetscReal Beta[3][3],PetscReal Rho[5][2],PetscReal Kappa[5][3])
{
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
}

PetscErrorCode System(IGAPoint p,PetscScalar K[],PetscScalar F[],void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal nu = user->nu;
  PetscReal E  = user->E;
  PetscReal t  = user->t;
  PetscReal k  = user->k;

  // get geometry
  PetscReal grad_g[3][2],hess_g[3][2][2];
  memcpy(grad_g,p->mapX[1],sizeof(grad_g));
  memcpy(hess_g,p->mapX[2],sizeof(hess_g));

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

  PetscReal dA1[2] = {0,0};
  PetscReal dA2[2] = {0,0};

  // get basis functions
  const PetscReal (*N0)    = p->basis[0];
  const PetscReal (*N1)[2] = (const PetscReal (*)[2])p->basis[1];

  PetscInt i,j;
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
          T[i][j] += k*0.5*E*t/(1.0+nu)*(dRho[i][0]*Rho[j][0]+dRho[i][1]*Rho[j][1]);

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

int main(int argc, char *argv[]) {

  char           filename[PETSC_MAX_PATH_LEN] = "ClassicalShell.dat";
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.nu = 0.3;
  user.E  = 3.e7;
  user.t  = 1.;
  user.k  = 5./6.;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,5);CHKERRQ(ierr); // dofs = {ux,uy,uz,psix,psiy}
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetGeometryDim(iga,3);CHKERRQ(ierr);
  ierr = IGARead(iga,filename);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Boundary conditions on u = 0, v = [0:1]
  ierr = IGASetBoundaryValue(iga,0,0,1,0.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,0,0,2,0.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,0,0,4,0.0);CHKERRQ(ierr);
  // Boundary conditions on u = 1, v = [0:1]
  ierr = IGASetBoundaryValue(iga,0,1,0,0.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,0,1,3,0.0);CHKERRQ(ierr);
  // Boundary conditions on u = [0:1], v = 0
  ierr = IGASetBoundaryValue(iga,1,0,1,0.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,1,0,4,0.0);CHKERRQ(ierr);
  // Boundary conditions on u = [0:1], v = 1
  ierr = IGASetBoundaryValue(iga,1,1,1,0.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,1,1,4,0.0);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,&user);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  PetscInt index;
  ierr = VecGetSize(b,&index);CHKERRQ(ierr);
  index -= 3;
  ierr = VecSetValue(b,index,-0.25,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscMPIInt rank,size;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if(rank == size-1){
    PetscScalar value;
    ierr = VecGetValues(x,1,&index,&value);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"x[%d]=%g\n",index,(double)value);CHKERRQ(ierr);
  }

  ierr = IGAWriteVec(iga,x,"ClassicalShell.out");CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = IGADrawVecVTK(iga,x,"ClassicalShell.vts");CHKERRQ(ierr);
#endif

  PetscBool draw = IGAGetOptBool(NULL,"-draw",PETSC_FALSE);
  if (draw) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
