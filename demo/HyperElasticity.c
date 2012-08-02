#include "petiga.h"

typedef struct {
  PetscReal lambda,mu,M;  //If M = 1 then S corresponds to NeoHook HyperElasticity Model
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "Model"
void Model(IGAPoint pnt, const PetscScalar *U, PetscScalar (*F)[3], PetscScalar (*S)[3], PetscScalar (*D)[6], void *ctx)
{    
  // This is the function that calculates F,S,D for using in the Residual and the Jacobian
  
  AppCtx *user = (AppCtx *)ctx;

  PetscReal lambda = user->lambda;
  PetscReal mu = user->mu;
  PetscReal M = user->M;

  // Interpolate the solution and gradient given U
  PetscScalar u[3];
  PetscScalar grad_u[3][3];
  IGAPointInterpolate(pnt,0,U,&u[0]);
  IGAPointInterpolate(pnt,1,U,&grad_u[0][0]);

  // Compute the deformation gradient,its determinant, and inverse
  PetscScalar Finv[3][3];
  PetscScalar J;

  F[0][0] = 1.0 + grad_u[0][0]; 
  F[0][1] = grad_u[0][1]; 
  F[0][2] = grad_u[0][2]; 
  F[1][0] = grad_u[1][0]; 
  F[1][1] = 1.0 + grad_u[1][1]; 
  F[1][2] = grad_u[1][2]; 
  F[2][0] = grad_u[2][0]; 
  F[2][1] = grad_u[2][1]; 
  F[2][2] = 1.0 + grad_u[2][2]; 

  J  = F[0][0]*(F[1][1]*F[2][2] - F[1][2]*F[2][1]);
  J -= F[0][1]*(F[1][0]*F[2][2] - F[1][2]*F[2][0]);
  J += F[0][2]*(F[1][0]*F[2][1] - F[1][1]*F[2][0]);

  //Finv GAEB Comment:revised with matlab symbolic tool 
  Finv[0][0] =  (F[1][1]*F[2][2]-F[2][1]*F[1][2])/J; //same as the Transpose
  Finv[1][0] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0])/J;
  Finv[2][0] =  (F[1][0]*F[2][1]-F[2][0]*F[1][1])/J;
  Finv[0][1] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1])/J;
  Finv[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0])/J; //same as the Transpose
  Finv[2][1] = -(F[0][0]*F[2][1]-F[2][0]*F[0][1])/J;
  Finv[0][2] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1])/J;
  Finv[1][2] = -(F[0][0]*F[1][2]-F[1][0]*F[0][2])/J;
  Finv[2][2] =  (F[0][0]*F[1][1]-F[1][0]*F[0][1])/J; //same as the Transpose


  // C^-1 = (F^T F)^-1 = F^-1 F^-T
  PetscScalar Cinv[3][3];
  Cinv[0][0] = Finv[0][0]*Finv[0][0] + Finv[0][1]*Finv[0][1] + Finv[0][2]*Finv[0][2];
  Cinv[0][1] = Finv[0][0]*Finv[1][0] + Finv[0][1]*Finv[1][1] + Finv[0][2]*Finv[1][2];
  Cinv[0][2] = Finv[0][0]*Finv[2][0] + Finv[0][1]*Finv[2][1] + Finv[0][2]*Finv[2][2];
  Cinv[1][0] = Finv[1][0]*Finv[0][0] + Finv[1][1]*Finv[0][1] + Finv[1][2]*Finv[0][2];
  Cinv[1][1] = Finv[1][0]*Finv[1][0] + Finv[1][1]*Finv[1][1] + Finv[1][2]*Finv[1][2];
  Cinv[1][2] = Finv[1][0]*Finv[2][0] + Finv[1][1]*Finv[2][1] + Finv[1][2]*Finv[2][2];
  Cinv[2][0] = Finv[2][0]*Finv[0][0] + Finv[2][1]*Finv[0][1] + Finv[2][2]*Finv[0][2];
  Cinv[2][1] = Finv[2][0]*Finv[1][0] + Finv[2][1]*Finv[1][1] + Finv[2][2]*Finv[1][2];
  Cinv[2][2] = Finv[2][0]*Finv[2][0] + Finv[2][1]*Finv[2][1] + Finv[2][2]*Finv[2][2];
	  
  if (M == 1){
    //Corresponds to Neohook compressible elasticity. Model by Ciarlet (1988).
    // Stress tensor
    PetscScalar temp=(0.5*lambda)*(J*J-1.0);
    S[0][0] = temp*Cinv[0][0] + mu*(1.0-Cinv[0][0]);
    S[0][1] = temp*Cinv[0][1] + mu*(-Cinv[0][1]);
    S[0][2] = temp*Cinv[0][2] + mu*(-Cinv[0][2]);
    S[1][0] = temp*Cinv[1][0] + mu*(-Cinv[1][0]);
    S[1][1] = temp*Cinv[1][1] + mu*(1.0-Cinv[1][1]);
    S[1][2] = temp*Cinv[1][2] + mu*(-Cinv[1][2]);
    S[2][0] = temp*Cinv[2][0] + mu*(-Cinv[2][0]);
    S[2][1] = temp*Cinv[2][1] + mu*(-Cinv[2][1]);
    S[2][2] = temp*Cinv[2][2] + mu*(1.0-Cinv[2][2]);
    
    //C_abcd=lambda*J^2*Cinv_ab*Cinv_cd+[2*miu-lambda(J^2-1)]*0.5(Cinv_ac*Cinv_bd+Cinv_ad*Cinv_bc)
    PetscScalar temp1=lambda*J*J;
    PetscScalar temp2=2*mu-lambda*(J*J-1);
    
    D[0][0]=temp1*Cinv[0][0]*Cinv[0][0]+temp2*0.5*(Cinv[0][0]*Cinv[0][0]+Cinv[0][0]*Cinv[0][0]);
    D[0][1]=temp1*Cinv[0][0]*Cinv[1][1]+temp2*0.5*(Cinv[0][1]*Cinv[0][1]+Cinv[0][1]*Cinv[0][1]);
    D[0][2]=temp1*Cinv[0][0]*Cinv[2][2]+temp2*0.5*(Cinv[0][2]*Cinv[0][2]+Cinv[0][2]*Cinv[0][2]);
    D[0][3]=temp1*Cinv[0][0]*Cinv[0][1]+temp2*0.5*(Cinv[0][0]*Cinv[0][1]+Cinv[0][1]*Cinv[0][0]);
    D[0][4]=temp1*Cinv[0][0]*Cinv[1][2]+temp2*0.5*(Cinv[0][1]*Cinv[0][2]+Cinv[0][2]*Cinv[0][1]);
    D[0][5]=temp1*Cinv[0][0]*Cinv[0][2]+temp2*0.5*(Cinv[0][0]*Cinv[0][2]+Cinv[0][2]*Cinv[0][0]);
    D[1][1]=temp1*Cinv[1][1]*Cinv[1][1]+temp2*0.5*(Cinv[1][1]*Cinv[1][1]+Cinv[1][1]*Cinv[1][1]);
    D[1][2]=temp1*Cinv[1][1]*Cinv[2][2]+temp2*0.5*(Cinv[1][2]*Cinv[1][2]+Cinv[1][2]*Cinv[1][2]);
    D[1][3]=temp1*Cinv[1][1]*Cinv[0][1]+temp2*0.5*(Cinv[1][0]*Cinv[1][1]+Cinv[1][1]*Cinv[1][0]);
    D[1][4]=temp1*Cinv[1][1]*Cinv[1][2]+temp2*0.5*(Cinv[1][1]*Cinv[1][2]+Cinv[1][2]*Cinv[1][1]);
    D[1][5]=temp1*Cinv[1][1]*Cinv[0][2]+temp2*0.5*(Cinv[1][0]*Cinv[1][2]+Cinv[1][2]*Cinv[1][0]);
    D[2][2]=temp1*Cinv[2][2]*Cinv[2][2]+temp2*0.5*(Cinv[2][2]*Cinv[2][2]+Cinv[2][2]*Cinv[2][2]);
    D[2][3]=temp1*Cinv[2][2]*Cinv[0][1]+temp2*0.5*(Cinv[2][0]*Cinv[2][1]+Cinv[2][1]*Cinv[2][0]);
    D[2][4]=temp1*Cinv[2][2]*Cinv[1][2]+temp2*0.5*(Cinv[2][1]*Cinv[2][2]+Cinv[2][2]*Cinv[2][1]);
    D[2][5]=temp1*Cinv[2][2]*Cinv[0][2]+temp2*0.5*(Cinv[2][0]*Cinv[2][2]+Cinv[2][2]*Cinv[2][0]);
    D[3][3]=temp1*Cinv[0][1]*Cinv[0][1]+temp2*0.5*(Cinv[0][0]*Cinv[1][1]+Cinv[0][1]*Cinv[1][0]);
    D[3][4]=temp1*Cinv[0][1]*Cinv[1][2]+temp2*0.5*(Cinv[0][1]*Cinv[1][2]+Cinv[0][2]*Cinv[1][1]);
    D[3][5]=temp1*Cinv[0][1]*Cinv[0][2]+temp2*0.5*(Cinv[0][0]*Cinv[1][2]+Cinv[0][2]*Cinv[1][0]);
    D[4][4]=temp1*Cinv[1][2]*Cinv[1][2]+temp2*0.5*(Cinv[1][1]*Cinv[2][2]+Cinv[1][2]*Cinv[2][1]);
    D[4][5]=temp1*Cinv[1][2]*Cinv[0][2]+temp2*0.5*(Cinv[1][0]*Cinv[2][2]+Cinv[1][2]*Cinv[2][0]);
    D[5][5]=temp1*Cinv[0][2]*Cinv[0][2]+temp2*0.5*(Cinv[0][0]*Cinv[2][2]+Cinv[0][2]*Cinv[2][0]);
    D[1][0]=D[0][1];
    D[2][0]=D[0][2];
    D[3][0]=D[0][3];
    D[4][0]=D[0][4];
    D[5][0]=D[0][5];
    D[2][1]=D[1][2];
    D[3][1]=D[1][3];
    D[4][1]=D[1][4];
    D[5][1]=D[1][5];
    D[3][2]=D[2][3];
    D[4][2]=D[2][4];
    D[5][2]=D[2][5];
    D[4][3]=D[3][4];
    D[5][3]=D[3][5];
    D[5][4]=D[4][5];

  }
  
}
#undef __FUNCT__
#define __FUNCT__ "DeltaE"
void DeltaE(const AppCtx *user, PetscScalar Nx, PetscScalar Ny, PetscScalar Nz, PetscScalar (*F)[3], PetscScalar (*B)[3], void *ctx)
{    
  // This is the function that calculates B  
  B[0][0] = F[0][0]*Nx; 
  B[0][1] = F[1][0]*Nx; 
  B[0][2] = F[2][0]*Nx;
  B[1][0] = F[0][1]*Ny; 
  B[1][1] = F[1][1]*Ny; 
  B[1][2] = F[2][1]*Ny;
  B[2][0] = F[0][2]*Nz; 
  B[2][1] = F[1][2]*Nz; 
  B[2][2] = F[2][2]*Nz;
  B[3][0] = F[0][0]*Ny+F[0][1]*Nx; 
  B[3][1] = F[1][0]*Ny+F[1][1]*Nx; 
  B[3][2] = F[2][0]*Ny+F[2][1]*Nx;
  B[4][0] = F[0][1]*Nz+F[0][2]*Ny; 
  B[4][1] = F[1][1]*Nz+F[1][2]*Ny; 
  B[4][2] = F[2][1]*Nz+F[2][2]*Ny;
  B[5][0] = F[0][0]*Nz+F[0][2]*Nx; 
  B[5][1] = F[1][0]*Nz+F[1][2]*Nx; 
  B[5][2] = F[2][0]*Nz+F[2][2]*Nx;

}


#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(IGAPoint pnt,const PetscScalar *U,PetscScalar *Re,void *ctx)
{    
  // This is the Residual for HyperElasticity without Neumann conditions and body forces. 
  //   R(w,u;u)=a(w,u;u)-L(w)~0

  AppCtx *user = (AppCtx *)ctx;

  PetscScalar F[3][3],S[3][3],D[6][6],B[6][3];
  Model(pnt, U, F, S, D, ctx);

  // Get basis functions and gradients
  PetscReal (*N1)[3] = (PetscReal (*)[3]) pnt->shape[1];

  // Put together the residual
  PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
  PetscInt a,nen=pnt->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];
    DeltaE(user, Na_x, Na_y, Na_z, F, B,ctx);
    R[a][0] = B[0][0]*S[0][0]+B[1][0]*S[1][1]+B[2][0]*S[2][2]+B[3][0]*S[0][1]+B[4][0]*S[1][2]+B[5][0]*S[0][2]; 
    R[a][1] = B[0][1]*S[0][0]+B[1][1]*S[1][1]+B[2][1]*S[2][2]+B[3][1]*S[0][1]+B[4][1]*S[1][2]+B[5][1]*S[0][2];
    R[a][2] = B[0][2]*S[0][0]+B[1][2]*S[1][1]+B[2][2]*S[2][2]+B[3][2]*S[0][1]+B[4][2]*S[1][2]+B[5][2]*S[0][2];

  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Jacobian"
PetscErrorCode Jacobian(IGAPoint pnt,const PetscScalar *U,PetscScalar *Je,void *ctx)
{    

  AppCtx *user = (AppCtx *)ctx;

  PetscScalar F[3][3],S[3][3],D[6][6];
  Model(pnt, U, F , S, D, ctx);

  // Get basis functions and gradients
  PetscReal (*N1)[3] = (PetscReal (*)[3]) pnt->shape[1];


  // Put together the jacobian
  PetscInt a,b,nen=pnt->nen;
  PetscScalar (*K)[3][nen][3] = (PetscScalar (*)[3][nen][3])Je;
  PetscScalar G;
  PetscScalar Chi[3][3];
  PetscScalar Ba[6][3];
  PetscScalar Bb[6][3];

  for (a=0; a<nen; a++) {
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];  
	//lamemos a DeltaB
    for (b=0; b<nen; b++) {
      PetscReal Nb_x  = N1[b][0];
      PetscReal Nb_y  = N1[b][1];
      PetscReal Nb_z  = N1[b][2];

	//G Matrix for the first integral 	
	G=Na_x*(S[0][0]*Nb_x + S[0][1]*Nb_y + S[0][2]*Nb_z) +
		Na_y*(S[1][0]*Nb_x + S[1][1]*Nb_y + S[1][2]*Nb_z) +
		Na_z*(S[2][0]*Nb_x + S[2][1]*Nb_y + S[2][2]*Nb_z);
		

	DeltaE(user, Na_x, Na_y, Na_z, F, Ba, ctx);
	DeltaE(user, Nb_x, Nb_y, Nb_z, F, Bb, ctx);
	
	Chi[0][0]=Bb[0][0]*(Ba[0][0]*D[0][0] + Ba[1][0]*D[1][0] + Ba[2][0]*D[2][0] + Ba[3][0]*D[3][0] + Ba[4][0]*D[4][0] + Ba[5][0]*D[5][0]) + 
		Bb[1][0]*(Ba[0][0]*D[0][1] + Ba[1][0]*D[1][1] + Ba[2][0]*D[2][1] + Ba[3][0]*D[3][1] + Ba[4][0]*D[4][1] + Ba[5][0]*D[5][1]) + 
		Bb[2][0]*(Ba[0][0]*D[0][2] + Ba[1][0]*D[1][2] + Ba[2][0]*D[2][2] + Ba[3][0]*D[3][2] + Ba[4][0]*D[4][2] + Ba[5][0]*D[5][2]) + 
		Bb[3][0]*(Ba[0][0]*D[0][3] + Ba[1][0]*D[1][3] + Ba[2][0]*D[2][3] + Ba[3][0]*D[3][3] + Ba[4][0]*D[4][3] + Ba[5][0]*D[5][3]) + 
		Bb[4][0]*(Ba[0][0]*D[0][4] + Ba[1][0]*D[1][4] + Ba[2][0]*D[2][4] + Ba[3][0]*D[3][4] + Ba[4][0]*D[4][4] + Ba[5][0]*D[5][4]) + 
		Bb[5][0]*(Ba[0][0]*D[0][5] + Ba[1][0]*D[1][5] + Ba[2][0]*D[2][5] + Ba[3][0]*D[3][5] + Ba[4][0]*D[4][5] + Ba[5][0]*D[5][5]);
	
	Chi[0][1]=Bb[0][1]*(Ba[0][0]*D[0][0] + Ba[1][0]*D[1][0] + Ba[2][0]*D[2][0] + Ba[3][0]*D[3][0] + Ba[4][0]*D[4][0] + Ba[5][0]*D[5][0]) + 
		Bb[1][1]*(Ba[0][0]*D[0][1] + Ba[1][0]*D[1][1] + Ba[2][0]*D[2][1] + Ba[3][0]*D[3][1] + Ba[4][0]*D[4][1] + Ba[5][0]*D[5][1]) + 
		Bb[2][1]*(Ba[0][0]*D[0][2] + Ba[1][0]*D[1][2] + Ba[2][0]*D[2][2] + Ba[3][0]*D[3][2] + Ba[4][0]*D[4][2] + Ba[5][0]*D[5][2]) + 
		Bb[3][1]*(Ba[0][0]*D[0][3] + Ba[1][0]*D[1][3] + Ba[2][0]*D[2][3] + Ba[3][0]*D[3][3] + Ba[4][0]*D[4][3] + Ba[5][0]*D[5][3]) + 
		Bb[4][1]*(Ba[0][0]*D[0][4] + Ba[1][0]*D[1][4] + Ba[2][0]*D[2][4] + Ba[3][0]*D[3][4] + Ba[4][0]*D[4][4] + Ba[5][0]*D[5][4]) + 
		Bb[5][1]*(Ba[0][0]*D[0][5] + Ba[1][0]*D[1][5] + Ba[2][0]*D[2][5] + Ba[3][0]*D[3][5] + Ba[4][0]*D[4][5] + Ba[5][0]*D[5][5]);
	
	Chi[0][2]=Bb[0][2]*(Ba[0][0]*D[0][0] + Ba[1][0]*D[1][0] + Ba[2][0]*D[2][0] + Ba[3][0]*D[3][0] + Ba[4][0]*D[4][0] + Ba[5][0]*D[5][0]) + 
		Bb[1][2]*(Ba[0][0]*D[0][1] + Ba[1][0]*D[1][1] + Ba[2][0]*D[2][1] + Ba[3][0]*D[3][1] + Ba[4][0]*D[4][1] + Ba[5][0]*D[5][1]) + 
		Bb[2][2]*(Ba[0][0]*D[0][2] + Ba[1][0]*D[1][2] + Ba[2][0]*D[2][2] + Ba[3][0]*D[3][2] + Ba[4][0]*D[4][2] + Ba[5][0]*D[5][2]) + 
		Bb[3][2]*(Ba[0][0]*D[0][3] + Ba[1][0]*D[1][3] + Ba[2][0]*D[2][3] + Ba[3][0]*D[3][3] + Ba[4][0]*D[4][3] + Ba[5][0]*D[5][3]) + 
		Bb[4][2]*(Ba[0][0]*D[0][4] + Ba[1][0]*D[1][4] + Ba[2][0]*D[2][4] + Ba[3][0]*D[3][4] + Ba[4][0]*D[4][4] + Ba[5][0]*D[5][4]) + 
		Bb[5][2]*(Ba[0][0]*D[0][5] + Ba[1][0]*D[1][5] + Ba[2][0]*D[2][5] + Ba[3][0]*D[3][5] + Ba[4][0]*D[4][5] + Ba[5][0]*D[5][5]);
	
	Chi[1][0]=Bb[0][0]*(Ba[0][1]*D[0][0] + Ba[1][1]*D[1][0] + Ba[2][1]*D[2][0] + Ba[3][1]*D[3][0] + Ba[4][1]*D[4][0] + Ba[5][1]*D[5][0]) + 
		Bb[1][0]*(Ba[0][1]*D[0][1] + Ba[1][1]*D[1][1] + Ba[2][1]*D[2][1] + Ba[3][1]*D[3][1] + Ba[4][1]*D[4][1] + Ba[5][1]*D[5][1]) + 
		Bb[2][0]*(Ba[0][1]*D[0][2] + Ba[1][1]*D[1][2] + Ba[2][1]*D[2][2] + Ba[3][1]*D[3][2] + Ba[4][1]*D[4][2] + Ba[5][1]*D[5][2]) + 
		Bb[3][0]*(Ba[0][1]*D[0][3] + Ba[1][1]*D[1][3] + Ba[2][1]*D[2][3] + Ba[3][1]*D[3][3] + Ba[4][1]*D[4][3] + Ba[5][1]*D[5][3]) + 
		Bb[4][0]*(Ba[0][1]*D[0][4] + Ba[1][1]*D[1][4] + Ba[2][1]*D[2][4] + Ba[3][1]*D[3][4] + Ba[4][1]*D[4][4] + Ba[5][1]*D[5][4]) + 
		Bb[5][0]*(Ba[0][1]*D[0][5] + Ba[1][1]*D[1][5] + Ba[2][1]*D[2][5] + Ba[3][1]*D[3][5] + Ba[4][1]*D[4][5] + Ba[5][1]*D[5][5]);
	
	Chi[1][1]=Bb[0][1]*(Ba[0][1]*D[0][0] + Ba[1][1]*D[1][0] + Ba[2][1]*D[2][0] + Ba[3][1]*D[3][0] + Ba[4][1]*D[4][0] + Ba[5][1]*D[5][0]) + 
		Bb[1][1]*(Ba[0][1]*D[0][1] + Ba[1][1]*D[1][1] + Ba[2][1]*D[2][1] + Ba[3][1]*D[3][1] + Ba[4][1]*D[4][1] + Ba[5][1]*D[5][1]) + 
		Bb[2][1]*(Ba[0][1]*D[0][2] + Ba[1][1]*D[1][2] + Ba[2][1]*D[2][2] + Ba[3][1]*D[3][2] + Ba[4][1]*D[4][2] + Ba[5][1]*D[5][2]) + 
		Bb[3][1]*(Ba[0][1]*D[0][3] + Ba[1][1]*D[1][3] + Ba[2][1]*D[2][3] + Ba[3][1]*D[3][3] + Ba[4][1]*D[4][3] + Ba[5][1]*D[5][3]) + 
		Bb[4][1]*(Ba[0][1]*D[0][4] + Ba[1][1]*D[1][4] + Ba[2][1]*D[2][4] + Ba[3][1]*D[3][4] + Ba[4][1]*D[4][4] + Ba[5][1]*D[5][4]) + 
		Bb[5][1]*(Ba[0][1]*D[0][5] + Ba[1][1]*D[1][5] + Ba[2][1]*D[2][5] + Ba[3][1]*D[3][5] + Ba[4][1]*D[4][5] + Ba[5][1]*D[5][5]);
	
	Chi[1][2]=Bb[0][2]*(Ba[0][1]*D[0][0] + Ba[1][1]*D[1][0] + Ba[2][1]*D[2][0] + Ba[3][1]*D[3][0] + Ba[4][1]*D[4][0] + Ba[5][1]*D[5][0]) + 
		Bb[1][2]*(Ba[0][1]*D[0][1] + Ba[1][1]*D[1][1] + Ba[2][1]*D[2][1] + Ba[3][1]*D[3][1] + Ba[4][1]*D[4][1] + Ba[5][1]*D[5][1]) + 
		Bb[2][2]*(Ba[0][1]*D[0][2] + Ba[1][1]*D[1][2] + Ba[2][1]*D[2][2] + Ba[3][1]*D[3][2] + Ba[4][1]*D[4][2] + Ba[5][1]*D[5][2]) + 
		Bb[3][2]*(Ba[0][1]*D[0][3] + Ba[1][1]*D[1][3] + Ba[2][1]*D[2][3] + Ba[3][1]*D[3][3] + Ba[4][1]*D[4][3] + Ba[5][1]*D[5][3]) + 
		Bb[4][2]*(Ba[0][1]*D[0][4] + Ba[1][1]*D[1][4] + Ba[2][1]*D[2][4] + Ba[3][1]*D[3][4] + Ba[4][1]*D[4][4] + Ba[5][1]*D[5][4]) + 
		Bb[5][2]*(Ba[0][1]*D[0][5] + Ba[1][1]*D[1][5] + Ba[2][1]*D[2][5] + Ba[3][1]*D[3][5] + Ba[4][1]*D[4][5] + Ba[5][1]*D[5][5]);
	
	Chi[2][0]=Bb[0][0]*(Ba[0][2]*D[0][0] + Ba[1][2]*D[1][0] + Ba[2][2]*D[2][0] + Ba[3][2]*D[3][0] + Ba[4][2]*D[4][0] + Ba[5][2]*D[5][0]) + 
		Bb[1][0]*(Ba[0][2]*D[0][1] + Ba[1][2]*D[1][1] + Ba[2][2]*D[2][1] + Ba[3][2]*D[3][1] + Ba[4][2]*D[4][1] + Ba[5][2]*D[5][1]) + 
		Bb[2][0]*(Ba[0][2]*D[0][2] + Ba[1][2]*D[1][2] + Ba[2][2]*D[2][2] + Ba[3][2]*D[3][2] + Ba[4][2]*D[4][2] + Ba[5][2]*D[5][2]) + 
		Bb[3][0]*(Ba[0][2]*D[0][3] + Ba[1][2]*D[1][3] + Ba[2][2]*D[2][3] + Ba[3][2]*D[3][3] + Ba[4][2]*D[4][3] + Ba[5][2]*D[5][3]) + 
		Bb[4][0]*(Ba[0][2]*D[0][4] + Ba[1][2]*D[1][4] + Ba[2][2]*D[2][4] + Ba[3][2]*D[3][4] + Ba[4][2]*D[4][4] + Ba[5][2]*D[5][4]) + 
		Bb[5][0]*(Ba[0][2]*D[0][5] + Ba[1][2]*D[1][5] + Ba[2][2]*D[2][5] + Ba[3][2]*D[3][5] + Ba[4][2]*D[4][5] + Ba[5][2]*D[5][5]); 
	
	Chi[2][1]=Bb[0][1]*(Ba[0][2]*D[0][0] + Ba[1][2]*D[1][0] + Ba[2][2]*D[2][0] + Ba[3][2]*D[3][0] + Ba[4][2]*D[4][0] + Ba[5][2]*D[5][0]) + 
		Bb[1][1]*(Ba[0][2]*D[0][1] + Ba[1][2]*D[1][1] + Ba[2][2]*D[2][1] + Ba[3][2]*D[3][1] + Ba[4][2]*D[4][1] + Ba[5][2]*D[5][1]) + 
		Bb[2][1]*(Ba[0][2]*D[0][2] + Ba[1][2]*D[1][2] + Ba[2][2]*D[2][2] + Ba[3][2]*D[3][2] + Ba[4][2]*D[4][2] + Ba[5][2]*D[5][2]) + 
		Bb[3][1]*(Ba[0][2]*D[0][3] + Ba[1][2]*D[1][3] + Ba[2][2]*D[2][3] + Ba[3][2]*D[3][3] + Ba[4][2]*D[4][3] + Ba[5][2]*D[5][3]) + 
		Bb[4][1]*(Ba[0][2]*D[0][4] + Ba[1][2]*D[1][4] + Ba[2][2]*D[2][4] + Ba[3][2]*D[3][4] + Ba[4][2]*D[4][4] + Ba[5][2]*D[5][4]) + 
		Bb[5][1]*(Ba[0][2]*D[0][5] + Ba[1][2]*D[1][5] + Ba[2][2]*D[2][5] + Ba[3][2]*D[3][5] + Ba[4][2]*D[4][5] + Ba[5][2]*D[5][5]); 
	
	Chi[2][2]=Bb[0][2]*(Ba[0][2]*D[0][0] + Ba[1][2]*D[1][0] + Ba[2][2]*D[2][0] + Ba[3][2]*D[3][0] + Ba[4][2]*D[4][0] + Ba[5][2]*D[5][0]) + 
		Bb[1][2]*(Ba[0][2]*D[0][1] + Ba[1][2]*D[1][1] + Ba[2][2]*D[2][1] + Ba[3][2]*D[3][1] + Ba[4][2]*D[4][1] + Ba[5][2]*D[5][1]) + 
		Bb[2][2]*(Ba[0][2]*D[0][2] + Ba[1][2]*D[1][2] + Ba[2][2]*D[2][2] + Ba[3][2]*D[3][2] + Ba[4][2]*D[4][2] + Ba[5][2]*D[5][2]) + 
		Bb[3][2]*(Ba[0][2]*D[0][3] + Ba[1][2]*D[1][3] + Ba[2][2]*D[2][3] + Ba[3][2]*D[3][3] + Ba[4][2]*D[4][3] + Ba[5][2]*D[5][3]) + 
		Bb[4][2]*(Ba[0][2]*D[0][4] + Ba[1][2]*D[1][4] + Ba[2][2]*D[2][4] + Ba[3][2]*D[3][4] + Ba[4][2]*D[4][4] + Ba[5][2]*D[5][4]) + 
		Bb[5][2]*(Ba[0][2]*D[0][5] + Ba[1][2]*D[1][5] + Ba[2][2]*D[2][5] + Ba[3][2]*D[3][5] + Ba[4][2]*D[4][5] + Ba[5][2]*D[5][5]);

	K[a][0][b][0] = G+Chi[0][0];
	K[a][1][b][0] = Chi[1][0];
	K[a][2][b][0] = Chi[2][0];
	
	K[a][0][b][1] = Chi[0][1];
	K[a][1][b][1] = G+Chi[1][1];
	K[a][2][b][1] = Chi[2][1];
	
	K[a][0][b][2] = Chi[0][2];
	K[a][1][b][2] = Chi[1][2];
	K[a][2][b][2] = G+Chi[2][2];
    }
  }
  
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) 
{
  // Initialization of PETSc
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  
  // Application specific data
  AppCtx user;
  user.lambda = 1.0;
  user.mu     = 1.0;
  user.M	  =   1;

  // Initialize the discretization
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Set boundary conditions
  IGABoundary bnd;
  ierr = IGAGetBoundary(iga,0,0,&bnd);CHKERRQ(ierr); // u = [0,0,0] @ x = [0,:,:]
  ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,2,0.0);CHKERRQ(ierr);
  ierr = IGAGetBoundary(iga,0,1,&bnd);CHKERRQ(ierr); // ux = 1 @ x = [1,:,:]
  ierr = IGABoundarySetValue(bnd,0,0.1);CHKERRQ(ierr);

  // Setup the nonlinear solver
  SNES snes;
  ierr = IGASetUserFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetUserJacobian(iga,Jacobian,&user);CHKERRQ(ierr);
  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // Solve
  Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,U);CHKERRQ(ierr);

  // Dump solution vector
  MPI_Comm        comm;
  PetscViewer     viewer;
  ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,"solution.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(U,viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
