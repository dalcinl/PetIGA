#include "petiga.h"

typedef struct {
  PetscReal   nu;
  PetscScalar fx,fy,fz;
  TS ts; /* XXX */
} AppCtx;

void Tau(PetscReal J[3][3],PetscReal dt,
         PetscScalar u[],PetscReal nu,
         PetscScalar *tauM,PetscScalar *tauC)
{
  PetscReal C_I = 1.0/12.0;

  PetscInt i,j,k;

  PetscReal G[3][3] = {{0}};
  for (i=0;i<3;i++)
  for (j=0;j<3;j++)
  for (k=0;k<3;k++)
    G[i][j] += J[i][k]*J[j][k];
  PetscReal g[3] = {0};
  for (i=0;i<3;i++)
  for (j=0;j<3;j++)
    g[i] += J[i][j];

  PetscReal G_G = 0;
  for (i=0;i<3;i++)
  for (j=0;j<3;j++)
    G_G += G[i][j]*G[i][j];
  PetscReal g_g = 0;
  for (i=0;i<3;i++)
    g_g += g[i]*g[i];

  PetscScalar u_G_u = 0;
  for (i=0;i<3;i++)
  for (j=0;j<3;j++)
    u_G_u += u[i]*G[i][j]*u[j];

  // Eqn 63
  *tauM = 4/(dt*dt) + u_G_u + C_I * nu*nu * G_G;
  *tauM = 1/sqrt(*tauM);
  // Eqn 64
  *tauC = (*tauM) * g_g;
  *tauC = 1/(*tauC);
}

void FineScale(const AppCtx *user,PetscScalar tauM,PetscScalar tauC,
               PetscScalar ux,
               PetscScalar ux_t,
               PetscScalar ux_x ,PetscScalar ux_y, PetscScalar ux_z,
               PetscScalar ux_xx,PetscScalar ux_yy,PetscScalar ux_zz,
               PetscScalar uy,
               PetscScalar uy_t,
               PetscScalar uy_x ,PetscScalar uy_y, PetscScalar uy_z,
               PetscScalar uy_xx,PetscScalar uy_yy,PetscScalar uy_zz,
               PetscScalar uz,
               PetscScalar uz_t,
               PetscScalar uz_x ,PetscScalar uz_y, PetscScalar uz_z,
               PetscScalar uz_xx,PetscScalar uz_yy,PetscScalar uz_zz,
               PetscScalar p,PetscScalar p_x,PetscScalar p_y,PetscScalar p_z,
               PetscScalar *ux_s,PetscScalar *uy_s,PetscScalar *uz_s,PetscScalar *p_s)
{
  // Eqn 61
  (*ux_s) = ux_t + (ux*ux_x + uy*ux_y + uz*ux_z) + p_x - user->nu*(ux_xx + ux_yy + ux_zz) - user->fx;
  (*uy_s) = uy_t + (ux*uy_x + uy*uy_y + uz*uy_z) + p_y - user->nu*(uy_xx + uy_yy + uy_zz) - user->fy;
  (*uz_s) = uz_t + (ux*uz_x + uy*uz_y + uz*uz_z) + p_z - user->nu*(uz_xx + uz_yy + uz_zz) - user->fz;
  // Eqn 62
  (*p_s)  = ux_x + uy_y + uz_z;
  // Eqn 58
  (*ux_s) *= -tauM;
  (*uy_s) *= -tauM;
  (*uz_s) *= -tauM;
  (*p_s)  *= -tauC;
}


PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal nu = user->nu;
  PetscReal dt; TSGetTimeStep(user->ts,&dt);

  PetscScalar u_t[4],u[4];
  PetscScalar grad_u[4][3];
  PetscScalar der2_u[4][3][3];
  IGAPointFormValue(pnt,V,&u_t[0]);
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);
  IGAPointFormHess (pnt,U,&der2_u[0][0][0]);

  PetscScalar ux=u[0],ux_t=u_t[0];
  PetscScalar uy=u[1],uy_t=u_t[1];
  PetscScalar uz=u[2],uz_t=u_t[2];
  PetscScalar p =u[3];

  PetscScalar ux_x=grad_u[0][0],ux_y=grad_u[0][1],ux_z=grad_u[0][2];
  PetscScalar uy_x=grad_u[1][0],uy_y=grad_u[1][1],uy_z=grad_u[1][2];
  PetscScalar uz_x=grad_u[2][0],uz_y=grad_u[2][1],uz_z=grad_u[2][2];
  PetscScalar p_x =grad_u[3][0],p_y =grad_u[3][1],p_z =grad_u[3][2];

  PetscScalar ux_xx=der2_u[0][0][0],ux_yy=der2_u[0][1][1],ux_zz=der2_u[0][2][2];
  PetscScalar uy_xx=der2_u[1][0][0],uy_yy=der2_u[1][1][1],uy_zz=der2_u[1][2][2];
  PetscScalar uz_xx=der2_u[2][0][0],uz_yy=der2_u[2][1][1],uz_zz=der2_u[2][2][2];

  PetscReal InvGradMap[3][3];
  IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);
  PetscScalar tauM,tauC;
  Tau(InvGradMap,dt,u,nu,&tauM,&tauC);
  PetscScalar ux_s,uy_s,uz_s,p_s;
  FineScale(user,tauM,tauC,
            ux,ux_t,ux_x,ux_y,ux_z,ux_xx,ux_yy,ux_zz,
            uy,uy_t,uy_x,uy_y,uy_z,uy_xx,uy_yy,uy_zz,
            uz,uz_t,uz_x,uz_y,uz_z,uz_xx,uz_yy,uz_zz,
            p,p_x,p_y,p_z,
            &ux_s,&uy_s,&uz_s,&p_s);

  PetscReal  *N0 = pnt->shape[0];
  PetscReal (*N1)[3] = (PetscReal (*)[3]) pnt->shape[1];

  PetscScalar (*R)[4] = (PetscScalar (*)[4])Re;
  PetscInt a,nen=pnt->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];
    /* ----- */
    PetscScalar Rux,Ruy,Ruz,Rp;
    // -L(W)
    Rux = -Na*user->fx;
    Ruy = -Na*user->fy;
    Ruz = -Na*user->fz;
    Rp  = 0.0;
    // += B_1(W,U)
    Rux += Na*ux_t - Na_x*p + nu*( Na_x*( ux_x + ux_x ) + Na_y*( ux_y + uy_x ) + Na_z*( ux_z + uz_x ) );
    Ruy += Na*uy_t - Na_y*p + nu*( Na_x*( uy_x + ux_y ) + Na_y*( uy_y + uy_y ) + Na_z*( uy_z + uz_y ) );
    Ruz += Na*uz_t - Na_z*p + nu*( Na_x*( uz_x + ux_z ) + Na_y*( uz_y + uy_z ) + Na_z*( uz_z + uz_z ) );
    Rp  += Na*( ux_x + uy_y + uz_z );
    // += Btilde_1(W,U')
    Rux += - ( Na_x*p_s );
    Ruy += - ( Na_y*p_s );
    Ruz += - ( Na_z*p_s );
    Rp  += - ( Na_x*ux_s + Na_y*uy_s + Na_z*uz_s );
    // += B_2(W,U,U+U') [after integration by parts, div(u+u')=0]
    Rux += + Na * ( (ux+ux_s)*ux_x + (uy+uy_s)*ux_y + (uz+uz_s)*ux_z );
    Ruy += + Na * ( (ux+ux_s)*uy_x + (uy+uy_s)*uy_y + (uz+uz_s)*uy_z );
    Ruz += + Na * ( (ux+ux_s)*uz_x + (uy+uy_s)*uz_y + (uz+uz_s)*uz_z );
    // += B_2(W,U',U+U')
    Rux += - ( Na_x*ux_s*(ux+ux_s) + Na_y*ux_s*(uy+uy_s) + Na_z*ux_s*(uz+uz_s) );
    Ruy += - ( Na_x*uy_s*(ux+ux_s) + Na_y*uy_s*(uy+uy_s) + Na_z*uy_s*(uz+uz_s) );
    Ruz += - ( Na_x*uz_s*(ux+ux_s) + Na_y*uz_s*(uy+uy_s) + Na_z*uz_s*(uz+uz_s) );
    /* ----- */
    R[a][0] = Rux;
    R[a][1] = Ruy;
    R[a][2] = Ruz;
    R[a][3] = Rp;
  }

  return 0;
}

PetscErrorCode Tangent(IGAPoint pnt,
                       PetscReal shift,const PetscScalar *V,
                       PetscReal t,const PetscScalar *U,
                       PetscScalar *Ke,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal nu = user->nu;
  PetscReal dt; TSGetTimeStep(user->ts,&dt);

  PetscScalar u[4];
  IGAPointFormValue(pnt,U,&u[0]);
  PetscScalar ux=u[0];
  PetscScalar uy=u[1];
  PetscScalar uz=u[2];

  PetscReal InvGradMap[3][3];
  IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);
  PetscScalar tauM,tauC;
  Tau(InvGradMap,dt,u,nu,&tauM,&tauC);

  PetscReal *N0 = pnt->shape[0];
  PetscReal (*N1)[3] = (PetscReal (*)[3]) pnt->shape[1];

  PetscInt a,b,nen=pnt->nen;
  PetscScalar (*K)[4][nen][4] = (PetscScalar (*)[4][nen][4])Ke;
  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];
    for (b=0; b<nen; b++) {
      PetscReal Nb    = N0[b];
      PetscReal Nb_x  = N1[b][0];
      PetscReal Nb_y  = N1[b][1];
      PetscReal Nb_z  = N1[b][2];
      /* ----- */
      PetscInt    i,j;
      PetscScalar T[4][4];
      PetscScalar Tii =
        (+ shift * Na * Nb
         + Na * (ux * Nb_x + uy * Nb_y + uz * Nb_z)
         + nu * (Na_x * Nb_x + Na_y * Nb_y + Na_z * Nb_z)

         + tauM * (ux * Na_x + uy * Na_y + uz * Na_z) *
         /**/     (shift * Nb + (ux * Nb_x + uy * Nb_y + uz * Nb_z))
         );
      T[0][0] = /*+ Na * Nb * ux_x*/  +  nu * Na_x * Nb_x  +  tauC * Na_x * Nb_x;
      T[0][1] = /*+ Na * Nb * ux_y*/  +  nu * Na_y * Nb_x  +  tauC * Na_x * Nb_y;
      T[0][2] = /*+ Na * Nb * ux_z*/  +  nu * Na_z * Nb_x  +  tauC * Na_x * Nb_z;
      //
      T[1][0] = /*+ Na * Nb * uy_x*/  +  nu * Na_x * Nb_y  +  tauC * Na_y * Nb_x;
      T[1][1] = /*+ Na * Nb * uy_y*/  +  nu * Na_y * Nb_y  +  tauC * Na_y * Nb_y;
      T[1][2] = /*+ Na * Nb * uy_z*/  +  nu * Na_z * Nb_y  +  tauC * Na_y * Nb_z;
      //
      T[2][0] = /*+ Na * Nb * uz_x*/  +  nu * Na_x * Nb_z  +  tauC * Na_z * Nb_x;
      T[2][1] = /*+ Na * Nb * uz_y*/  +  nu * Na_y * Nb_z  +  tauC * Na_z * Nb_y;
      T[2][2] = /*+ Na * Nb * uz_z*/  +  nu * Na_z * Nb_z  +  tauC * Na_z * Nb_z;
      T[0][0] += Tii;
      T[1][1] += Tii;
      T[2][2] += Tii;
      // G as in Eq. (104)
      T[0][3] = - Na_x * Nb  +  tauM * (ux * Na_x + uy * Na_y + uz * Na_z) * Nb_x;
      T[1][3] = - Na_y * Nb  +  tauM * (ux * Na_x + uy * Na_y + uz * Na_z) * Nb_y;
      T[2][3] = - Na_z * Nb  +  tauM * (ux * Na_x + uy * Na_y + uz * Na_z) * Nb_z;
      // D as in Eq. (106)
      T[3][0] = + Na * Nb_x  +  tauM * Na_x * (shift * Nb + (ux * Nb_x + uy * Nb_y + uz * Nb_z));
      T[3][1] = + Na * Nb_y  +  tauM * Na_y * (shift * Nb + (ux * Nb_x + uy * Nb_y + uz * Nb_z));
      T[3][2] = + Na * Nb_z  +  tauM * Na_z * (shift * Nb + (ux * Nb_x + uy * Nb_y + uz * Nb_z));
      // L as in Eq. (108)
      T[3][3] = + tauM * (Na_x * Nb_x + Na_y * Nb_y + Na_z * Nb_z);
      /* ----- */
      for (i=0;i<4;i++)
        for (j=0;j<4;j++)
          K[a][i][b][j] += T[i][j];
    }
  }

  return 0;
}

PetscErrorCode FormInitialCondition(AppCtx *user,IGA iga,const char datafile[],PetscReal t,Vec U)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(U);CHKERRQ(ierr);

  if (datafile[0] != 0) { /* initial condition from datafile */
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,datafile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  } else {
    DM da;
    ierr = IGACreateNodeDM(iga,4,&da);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    PetscScalar ****u;
    ierr = DMDAVecGetArrayDOF(da,U,&u);CHKERRQ(ierr);

    PetscScalar H    = 2;//user->Ly;
    PetscScalar visc = user->nu;
    PetscScalar dpdx = user->fx;
    PetscInt  i,j,k;
    PetscReal jmax = (info.my-1);
    for(k=info.zs;k<info.zs+info.zm;k++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        for(i=info.xs;i<info.xs+info.xm;i++){
          PetscReal   y = (j/jmax)*H;
          PetscScalar ux = 1/(2*visc) * dpdx * y*(H-y);
          u[k][j][i][0] = ux;
        }
      }
    }

    ierr = DMDAVecRestoreArrayDOF(da,U,&u);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);

    PetscReal v; Vec pert;
    ierr = VecDuplicate(U,&pert);CHKERRQ(ierr);
    ierr = VecSetRandom(pert,NULL);CHKERRQ(ierr);
    ierr = VecMax(U,NULL,&v);CHKERRQ(ierr);
    ierr = VecAXPY(U,0.05*v,pert);CHKERRQ(ierr);
    ierr = VecStrideSet(U,3,0.0);CHKERRQ(ierr);
    ierr = VecDestroy(&pert);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode OutputMonitor(TS ts,PetscInt it_number,PetscReal c_time,Vec U,void *ctx)
{
  IGA            iga;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof(filename),"./nsvms%d.dat",(int)it_number);CHKERRQ(ierr);
  ierr = IGAWriteVec(iga,U,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.nu = 1.47200e-4;
  user.fx = 3.37204e-3;
  user.fy = 0.0;
  user.fz = 0.0;

  PetscReal PI = 4.0*atan(1.0);
  PetscReal Lx = 2.0*PI;
  PetscReal Ly = 2.0;
  PetscReal Lz = 2.0/3.0*PI;

  PetscInt N[3] = {16,16,16}, nN = 3;
  PetscInt p[3] = { 2, 2, 2}, np = 3;
  PetscInt C[3] = {-1,-1,-1}, nC = 3;
  PetscBool output = PETSC_FALSE;
  PetscBool monitor = PETSC_FALSE;
  char initial[PETSC_MAX_PATH_LEN] = {0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","NavierStokesVMS Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-N","number of elements",     __FILE__,N,&nN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-p","polynomial order",       __FILE__,p,&np,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-C","global continuity order",__FILE__,C,&nC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-ns_initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool  ("-ns_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool  ("-ns_monitor","Compute and show statistics of solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (nN == 1) N[2] = N[1] = N[0]; if (nN == 2) N[2] = N[0];
  if (np == 1) p[2] = p[1] = p[0]; if (np == 2) p[2] = p[0];
  if (nC == 1) C[2] = C[1] = C[0]; if (nC == 2) C[2] = C[0];
  if (C[0] == -1) C[0] = p[0]-1;
  if (C[1] == -1) C[1] = p[1]-1;
  if (C[2] == -1) C[2] = p[2]-1;
  PetscInt i;
  for (i=0; i<3; i++) {
    if (p[i] < 2 || C[i] < 1) /* Problem requires a p>=2 C1 basis */
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
              "Problem requires minimum of p = 2 and C = 1");
    if (p[i] <= C[i])         /* Check C < p */
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
              "Discretization inconsistent: polynomial order must be greater than degree of continuity");
  }

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetDof(iga,4);CHKERRQ(ierr);

  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis0,p[0]);CHKERRQ(ierr);
  ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);
  //ierr = IGAAxisInitUniform(axis0,N,-0.5*Lx,0.5*Lx);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,N[0],0.0,Lx,C[0]);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis1,p[1]);CHKERRQ(ierr);
  //ierr = IGAAxisInitUniform(axis1,N,-0.5*Ly,0.5*Ly);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1,N[1],0.0,Ly,C[1]);CHKERRQ(ierr);
  IGAAxis axis2;
  ierr = IGAGetAxis(iga,2,&axis2);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis2,p[2]);CHKERRQ(ierr);
  ierr = IGAAxisSetPeriodic(axis2,PETSC_TRUE);CHKERRQ(ierr);
  //ierr = IGAAxisInitUniform(axis2,N,-0.5*Lz,0.5*Lz);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis2,N[2],0.0,Lz,C[2]);CHKERRQ(ierr);

  PetscInt dir=1,side,field;
  for (side=0;side<2;side++) {
    for (field=0;field<3;field++) {
      ierr = IGASetBoundaryValue(iga,dir,side,field,0.0);CHKERRQ(ierr);
    }
  }

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Tangent ,&user);CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  user.ts = ts;

  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1.0e-2);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1000.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (output) {ierr = TSMonitorSet(ts,OutputMonitor,&user,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(&user,iga,initial,t,U);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
