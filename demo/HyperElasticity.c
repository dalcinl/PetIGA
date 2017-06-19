#include "petiga.h"
#include "petscmat.h"
#include "petscblaslapack.h"

/*
  This code implements a HyperElastic material model in the context of
  large deformation elasticity. Implementation credit goes to students
  of the 2012 summer course `Nonlinear Finite Element Analysis' given
  in Universidad de los Andes, Bogota, Colombia:

  Lina María Bernal Martinez
  Gabriel Andres Espinosa Barrios
  Federico Fuentes Caycedo
  Juan Camilo Mahecha Zambrano

 */

typedef struct {
  PetscReal lambda,mu,a,b,c,d,c1,c2,kappa;
  void (*model) (IGAPoint pnt, const PetscScalar *U, PetscScalar (*F)[3], PetscScalar (*S)[3], PetscScalar (*D)[6], void *ctx);
} AppCtx;

void NeoHookeanModel(IGAPoint pnt, const PetscScalar *U, PetscScalar (*F)[3], PetscScalar (*S)[3], PetscScalar (*D)[6], void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscReal lambda = user->lambda;
  PetscReal mu = user->mu;

  // Interpolate the solution and gradient given U
  PetscScalar u[3];
  PetscScalar grad_u[3][3];
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);

  // F = I + u_{i,j}
  F[0][0] = 1+grad_u[0][0]; F[0][1] =   grad_u[0][1];  F[0][2] =   grad_u[0][2];
  F[1][0] =   grad_u[1][0]; F[1][1] = 1+grad_u[1][1];  F[1][2] =   grad_u[1][2];
  F[2][0] =   grad_u[2][0]; F[2][1] =   grad_u[2][1];  F[2][2] = 1+grad_u[2][2];

  // Finv
  PetscScalar Finv[3][3],J,Jinv;
  J  = F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]);
  J -= F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0]);
  J += F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
  Jinv = 1./J;
  Finv[0][0] =  (F[1][1]*F[2][2]-F[2][1]*F[1][2])*Jinv;
  Finv[1][0] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0])*Jinv;
  Finv[2][0] =  (F[1][0]*F[2][1]-F[2][0]*F[1][1])*Jinv;
  Finv[0][1] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1])*Jinv;
  Finv[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0])*Jinv;
  Finv[2][1] = -(F[0][0]*F[2][1]-F[2][0]*F[0][1])*Jinv;
  Finv[0][2] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1])*Jinv;
  Finv[1][2] = -(F[0][0]*F[1][2]-F[1][0]*F[0][2])*Jinv;
  Finv[2][2] =  (F[0][0]*F[1][1]-F[1][0]*F[0][1])*Jinv;

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

  // 2nd Piola-Kirchoff stress tensor
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

  // C_abcd=lambda*J^2*Cinv_ab*Cinv_cd+[2*miu-lambda(J^2-1)]*0.5(Cinv_ac*Cinv_bd+Cinv_ad*Cinv_bc)
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

void StVenantModel(IGAPoint pnt, const PetscScalar *U, PetscScalar (*F)[3], PetscScalar (*S)[3], PetscScalar (*D)[6], void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscReal lambda = user->lambda;
  PetscReal mu = user->mu;

  // Interpolate the solution and gradient given U
  PetscScalar u[3];
  PetscScalar grad_u[3][3];
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);

  // F = I + u_{i,j}
  F[0][0] = 1+grad_u[0][0]; F[0][1] =   grad_u[0][1];  F[0][2] =   grad_u[0][2];
  F[1][0] =   grad_u[1][0]; F[1][1] = 1+grad_u[1][1];  F[1][2] =   grad_u[1][2];
  F[2][0] =   grad_u[2][0]; F[2][1] =   grad_u[2][1];  F[2][2] = 1+grad_u[2][2];

  // C^-1 = (F^T F)^-1 = F^-1 F^-T
  PetscScalar C[3][3];
  C[0][0] = F[0][0]*F[0][0] + F[1][0]*F[1][0] + F[2][0]*F[2][0];
  C[0][1] = F[0][0]*F[0][1] + F[1][0]*F[1][1] + F[2][0]*F[2][1];
  C[0][2] = F[0][0]*F[0][2] + F[1][0]*F[1][2] + F[2][0]*F[2][2];
  C[1][0] = F[0][1]*F[0][0] + F[1][1]*F[1][0] + F[2][1]*F[2][0];
  C[1][1] = F[0][1]*F[0][1] + F[1][1]*F[1][1] + F[2][1]*F[2][1];
  C[1][2] = F[0][1]*F[0][2] + F[1][1]*F[1][2] + F[2][1]*F[2][2];
  C[2][0] = F[0][2]*F[0][0] + F[1][2]*F[1][0] + F[2][2]*F[2][0];
  C[2][1] = F[0][2]*F[0][1] + F[1][2]*F[1][1] + F[2][2]*F[2][1];
  C[2][2] = F[0][2]*F[0][2] + F[1][2]*F[1][2] + F[2][2]*F[2][2];

  //Corresponds to Saint-Venant - Kirchhoff
  // Second Piola Kirchhoff Stress tensor
  PetscScalar temp=0.5*lambda*(C[0][0]+C[1][1]+C[2][2]-3);
  S[0][0] = temp + mu*(C[0][0]-1);
  S[0][1] = mu*C[0][1];
  S[0][2] = mu*C[0][2];
  S[1][0] = mu*C[1][0];
  S[1][1] = temp + mu*(C[1][1]-1);
  S[1][2] = mu*C[1][2];
  S[2][0] = mu*C[2][0];
  S[2][1] = mu*C[2][1];
  S[2][2] = temp + mu*(C[2][2]-1);

  //C_abcd = lambda*delta_ab*delta_cd+2*mu*0.5*(delta_ac*delta_bd+delta_ad*delta_bc)
  //Elasticity material tensor
  D[0][0]=lambda+2*mu;
  D[0][1]=lambda;
  D[0][2]=lambda;
  D[0][3]=0;
  D[0][4]=0;
  D[0][5]=0;
  D[1][1]=lambda+2*mu;
  D[1][2]=lambda;
  D[1][3]=0;
  D[1][4]=0;
  D[1][5]=0;
  D[2][2]=lambda+2*mu;
  D[2][3]=0;
  D[2][4]=0;
  D[2][5]=0;
  D[3][3]=mu;
  D[3][4]=0;
  D[3][5]=0;
  D[4][4]=mu;
  D[4][5]=0;
  D[5][5]=mu;
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

void MooneyRivlinModel1(IGAPoint pnt, const PetscScalar *U, PetscScalar (*F)[3], PetscScalar (*S)[3], PetscScalar (*D)[6], void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscReal a = user->a;
  PetscReal b = user->b;
  PetscReal c = user->c;
  PetscReal d = user->d;

  // Interpolate the solution and gradient given U
  PetscScalar u[3];
  PetscScalar grad_u[3][3];
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);

  // F = I + u_{i,j}
  F[0][0] = 1+grad_u[0][0]; F[0][1] =   grad_u[0][1];  F[0][2] =   grad_u[0][2];
  F[1][0] =   grad_u[1][0]; F[1][1] = 1+grad_u[1][1];  F[1][2] =   grad_u[1][2];
  F[2][0] =   grad_u[2][0]; F[2][1] =   grad_u[2][1];  F[2][2] = 1+grad_u[2][2];

  // Finv
  PetscScalar Finv[3][3],J,Jinv;
  J  = F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]);
  J -= F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0]);
  J += F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
  Jinv = 1./J;
  Finv[0][0] =  (F[1][1]*F[2][2]-F[2][1]*F[1][2])*Jinv;
  Finv[1][0] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0])*Jinv;
  Finv[2][0] =  (F[1][0]*F[2][1]-F[2][0]*F[1][1])*Jinv;
  Finv[0][1] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1])*Jinv;
  Finv[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0])*Jinv;
  Finv[2][1] = -(F[0][0]*F[2][1]-F[2][0]*F[0][1])*Jinv;
  Finv[0][2] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1])*Jinv;
  Finv[1][2] = -(F[0][0]*F[1][2]-F[1][0]*F[0][2])*Jinv;
  Finv[2][2] =  (F[0][0]*F[1][1]-F[1][0]*F[0][1])*Jinv;

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

  //Cs=Cinv^2
  PetscScalar Cs[3][3];
  Cs[0][0]= Cinv[0][0]*Cinv[0][0] + Cinv[0][1]*Cinv[1][0] + Cinv[0][2]*Cinv[2][0];
  Cs[0][1]= Cinv[0][0]*Cinv[0][1] + Cinv[0][1]*Cinv[1][1] + Cinv[0][2]*Cinv[2][1];
  Cs[0][2]= Cinv[0][0]*Cinv[0][2] + Cinv[0][1]*Cinv[1][2] + Cinv[0][2]*Cinv[2][2];
  Cs[1][0]= Cinv[1][0]*Cinv[0][0] + Cinv[1][1]*Cinv[1][0] + Cinv[1][2]*Cinv[2][0];
  Cs[1][1]= Cinv[1][0]*Cinv[0][1] + Cinv[1][1]*Cinv[1][1] + Cinv[1][2]*Cinv[2][1];
  Cs[1][2]= Cinv[1][0]*Cinv[0][2] + Cinv[1][1]*Cinv[1][2] + Cinv[1][2]*Cinv[2][2];
  Cs[2][0]= Cinv[2][0]*Cinv[0][0] + Cinv[2][1]*Cinv[1][0] + Cinv[2][2]*Cinv[2][0];
  Cs[2][1]= Cinv[2][0]*Cinv[0][1] + Cinv[2][1]*Cinv[1][1] + Cinv[2][2]*Cinv[2][1];
  Cs[2][2]= Cinv[2][0]*Cinv[0][2] + Cinv[2][1]*Cinv[1][2] + Cinv[2][2]*Cinv[2][2];

  //Sij=2*a*deltaij+(J^2*(2*c+b*tr(Cinv))-d)Cinvij-b*J^2*Csij
  PetscScalar temp=Cinv[0][0]+Cinv[1][1]+Cinv[2][2]; //tr(Cinv)
  S[0][0]=2*a+(J*J*(2*c+b*temp)-d)*Cinv[0][0]-b*J*J*Cs[0][0];
  S[0][1]=(J*J*(2*c+b*temp)-d)*Cinv[0][1]-b*J*J*Cs[0][1];
  S[0][2]=(J*J*(2*c+b*temp)-d)*Cinv[0][2]-b*J*J*Cs[0][2];
  S[1][0]=(J*J*(2*c+b*temp)-d)*Cinv[1][0]-b*J*J*Cs[1][0];
  S[1][1]=2*a+(J*J*(2*c+b*temp)-d)*Cinv[1][1]-b*J*J*Cs[1][1];
  S[1][2]=(J*J*(2*c+b*temp)-d)*Cinv[1][2]-b*J*J*Cs[1][2];
  S[2][0]=(J*J*(2*c+b*temp)-d)*Cinv[2][0]-b*J*J*Cs[2][0];
  S[2][1]=(J*J*(2*c+b*temp)-d)*Cinv[2][1]-b*J*J*Cs[2][1];
  S[2][2]=2*a+(J*J*(2*c+b*temp)-d)*Cinv[2][2]-b*J*J*Cs[2][2];

  //Dijkl = J^2(4c+2b*tr(Cinv))*Cinvij*Cinvkl-2bJ^2(Cinv^2ij*Cinvkl+Cinvij*Cinv^2kl-0.5*(Cinv^2li*Cinvjk+Cinvli*Cinv^2jk+Cinv^2lj*Cinvik+Cinvlj*Cinv^2ik))-(J^2*(4c+2btr(Cinv))-2d)*0.5*(Cinvik*Cinvlj+Cinvli*Cinvjk)
  PetscScalar temp1= J*J*(4*c+2*b*temp);
  PetscScalar temp2=2*b*J*J;
  PetscScalar temp3=temp1-2*d;
  D[0][0]=temp1*Cinv[0][0]*Cinv[0][0]-temp2*(Cs[0][0]*Cinv[0][0]+Cinv[0][0]*Cs[0][0]-0.5*(Cs[0][0]*Cinv[0][0]+Cinv[0][0]*Cs[0][0]+Cs[0][0]*Cinv[0][0]+Cinv[0][0]*Cs[0][0]))-temp3*0.5*(Cinv[0][0]*Cinv[0][0]+Cinv[0][0]*Cinv[0][0]);
  D[0][1]=temp1*Cinv[0][0]*Cinv[1][1]-temp2*(Cs[0][0]*Cinv[1][1]+Cinv[0][0]*Cs[1][1]-0.5*(Cs[1][0]*Cinv[0][1]+Cinv[1][0]*Cs[0][1]+Cs[1][0]*Cinv[0][1]+Cinv[1][0]*Cs[0][1]))-temp3*0.5*(Cinv[0][1]*Cinv[1][0]+Cinv[1][0]*Cinv[0][1]);
  D[0][2]=temp1*Cinv[0][0]*Cinv[2][2]-temp2*(Cs[0][0]*Cinv[2][2]+Cinv[0][0]*Cs[2][2]-0.5*(Cs[2][0]*Cinv[0][2]+Cinv[2][0]*Cs[0][2]+Cs[2][0]*Cinv[0][2]+Cinv[2][0]*Cs[0][2]))-temp3*0.5*(Cinv[0][2]*Cinv[2][0]+Cinv[2][0]*Cinv[0][2]);
  D[0][3]=temp1*Cinv[0][0]*Cinv[0][1]-temp2*(Cs[0][0]*Cinv[0][1]+Cinv[0][0]*Cs[0][1]-0.5*(Cs[1][0]*Cinv[0][0]+Cinv[1][0]*Cs[0][0]+Cs[1][0]*Cinv[0][0]+Cinv[1][0]*Cs[0][0]))-temp3*0.5*(Cinv[0][0]*Cinv[1][0]+Cinv[1][0]*Cinv[0][0]);
  D[0][4]=temp1*Cinv[0][0]*Cinv[1][2]-temp2*(Cs[0][0]*Cinv[1][2]+Cinv[0][0]*Cs[1][2]-0.5*(Cs[2][0]*Cinv[0][1]+Cinv[2][0]*Cs[0][1]+Cs[2][0]*Cinv[0][1]+Cinv[2][0]*Cs[0][1]))-temp3*0.5*(Cinv[0][1]*Cinv[2][0]+Cinv[2][0]*Cinv[0][1]);
  D[0][5]=temp1*Cinv[0][0]*Cinv[0][2]-temp2*(Cs[0][0]*Cinv[0][2]+Cinv[0][0]*Cs[0][2]-0.5*(Cs[2][0]*Cinv[0][0]+Cinv[2][0]*Cs[0][0]+Cs[2][0]*Cinv[0][0]+Cinv[2][0]*Cs[0][0]))-temp3*0.5*(Cinv[0][0]*Cinv[2][0]+Cinv[2][0]*Cinv[0][0]);
  D[1][1]=temp1*Cinv[1][1]*Cinv[1][1]-temp2*(Cs[1][1]*Cinv[1][1]+Cinv[1][1]*Cs[1][1]-0.5*(Cs[1][1]*Cinv[1][1]+Cinv[1][1]*Cs[1][1]+Cs[1][1]*Cinv[1][1]+Cinv[1][1]*Cs[1][1]))-temp3*0.5*(Cinv[1][1]*Cinv[1][1]+Cinv[1][1]*Cinv[1][1]);
  D[1][2]=temp1*Cinv[1][1]*Cinv[2][2]-temp2*(Cs[1][1]*Cinv[2][2]+Cinv[1][1]*Cs[2][2]-0.5*(Cs[2][1]*Cinv[1][2]+Cinv[2][1]*Cs[1][2]+Cs[2][1]*Cinv[1][2]+Cinv[2][1]*Cs[1][2]))-temp3*0.5*(Cinv[1][2]*Cinv[2][1]+Cinv[2][1]*Cinv[1][2]);
  D[1][3]=temp1*Cinv[1][1]*Cinv[0][1]-temp2*(Cs[1][1]*Cinv[0][1]+Cinv[1][1]*Cs[0][1]-0.5*(Cs[1][1]*Cinv[1][0]+Cinv[1][1]*Cs[1][0]+Cs[1][1]*Cinv[1][0]+Cinv[1][1]*Cs[1][0]))-temp3*0.5*(Cinv[1][0]*Cinv[1][1]+Cinv[1][1]*Cinv[1][0]);
  D[1][4]=temp1*Cinv[1][1]*Cinv[1][2]-temp2*(Cs[1][1]*Cinv[1][2]+Cinv[1][1]*Cs[1][2]-0.5*(Cs[2][1]*Cinv[1][1]+Cinv[2][1]*Cs[1][1]+Cs[2][1]*Cinv[1][1]+Cinv[2][1]*Cs[1][1]))-temp3*0.5*(Cinv[1][1]*Cinv[2][1]+Cinv[2][1]*Cinv[1][1]);
  D[1][5]=temp1*Cinv[1][1]*Cinv[0][2]-temp2*(Cs[1][1]*Cinv[0][2]+Cinv[1][1]*Cs[0][2]-0.5*(Cs[2][1]*Cinv[1][0]+Cinv[2][1]*Cs[1][0]+Cs[2][1]*Cinv[1][0]+Cinv[2][1]*Cs[1][0]))-temp3*0.5*(Cinv[1][0]*Cinv[2][1]+Cinv[2][1]*Cinv[1][0]);
  D[2][2]=temp1*Cinv[2][2]*Cinv[2][2]-temp2*(Cs[2][2]*Cinv[2][2]+Cinv[2][2]*Cs[2][2]-0.5*(Cs[2][2]*Cinv[2][2]+Cinv[2][2]*Cs[2][2]+Cs[2][2]*Cinv[2][2]+Cinv[2][2]*Cs[2][2]))-temp3*0.5*(Cinv[2][2]*Cinv[2][2]+Cinv[2][2]*Cinv[2][2]);
  D[2][3]=temp1*Cinv[2][2]*Cinv[0][1]-temp2*(Cs[2][2]*Cinv[0][1]+Cinv[2][2]*Cs[0][1]-0.5*(Cs[1][2]*Cinv[2][0]+Cinv[1][2]*Cs[2][0]+Cs[1][2]*Cinv[2][0]+Cinv[1][2]*Cs[2][0]))-temp3*0.5*(Cinv[2][0]*Cinv[1][2]+Cinv[1][2]*Cinv[2][0]);
  D[2][4]=temp1*Cinv[2][2]*Cinv[1][2]-temp2*(Cs[2][2]*Cinv[1][2]+Cinv[2][2]*Cs[1][2]-0.5*(Cs[2][2]*Cinv[2][1]+Cinv[2][2]*Cs[2][1]+Cs[2][2]*Cinv[2][1]+Cinv[2][2]*Cs[2][1]))-temp3*0.5*(Cinv[2][1]*Cinv[2][2]+Cinv[2][2]*Cinv[2][1]);
  D[2][5]=temp1*Cinv[2][2]*Cinv[0][2]-temp2*(Cs[2][2]*Cinv[0][2]+Cinv[2][2]*Cs[0][2]-0.5*(Cs[2][2]*Cinv[2][0]+Cinv[2][2]*Cs[2][0]+Cs[2][2]*Cinv[2][0]+Cinv[2][2]*Cs[2][0]))-temp3*0.5*(Cinv[2][0]*Cinv[2][2]+Cinv[2][2]*Cinv[2][0]);
  D[3][3]=temp1*Cinv[0][1]*Cinv[0][1]-temp2*(Cs[0][1]*Cinv[0][1]+Cinv[0][1]*Cs[0][1]-0.5*(Cs[1][0]*Cinv[1][0]+Cinv[1][0]*Cs[1][0]+Cs[1][1]*Cinv[0][0]+Cinv[1][1]*Cs[0][0]))-temp3*0.5*(Cinv[0][0]*Cinv[1][1]+Cinv[1][0]*Cinv[1][0]);
  D[3][4]=temp1*Cinv[0][1]*Cinv[1][2]-temp2*(Cs[0][1]*Cinv[1][2]+Cinv[0][1]*Cs[1][2]-0.5*(Cs[2][0]*Cinv[1][1]+Cinv[2][0]*Cs[1][1]+Cs[2][1]*Cinv[0][1]+Cinv[2][1]*Cs[0][1]))-temp3*0.5*(Cinv[0][1]*Cinv[2][1]+Cinv[2][0]*Cinv[1][1]);
  D[3][5]=temp1*Cinv[0][1]*Cinv[0][2]-temp2*(Cs[0][1]*Cinv[0][2]+Cinv[0][1]*Cs[0][2]-0.5*(Cs[2][0]*Cinv[1][0]+Cinv[2][0]*Cs[1][0]+Cs[2][1]*Cinv[0][0]+Cinv[2][1]*Cs[0][0]))-temp3*0.5*(Cinv[0][0]*Cinv[2][1]+Cinv[2][0]*Cinv[1][0]);
  D[4][4]=temp1*Cinv[1][2]*Cinv[1][2]-temp2*(Cs[1][2]*Cinv[1][2]+Cinv[1][2]*Cs[1][2]-0.5*(Cs[2][1]*Cinv[2][1]+Cinv[2][1]*Cs[2][1]+Cs[2][2]*Cinv[1][1]+Cinv[2][2]*Cs[1][1]))-temp3*0.5*(Cinv[1][1]*Cinv[2][2]+Cinv[2][1]*Cinv[2][1]);
  D[4][5]=temp1*Cinv[1][2]*Cinv[0][2]-temp2*(Cs[1][2]*Cinv[0][2]+Cinv[1][2]*Cs[0][2]-0.5*(Cs[2][1]*Cinv[2][0]+Cinv[2][1]*Cs[2][0]+Cs[2][2]*Cinv[1][0]+Cinv[2][2]*Cs[1][0]))-temp3*0.5*(Cinv[1][0]*Cinv[2][2]+Cinv[2][1]*Cinv[2][0]);
  D[5][5]=temp1*Cinv[0][2]*Cinv[0][2]-temp2*(Cs[0][2]*Cinv[0][2]+Cinv[0][2]*Cs[0][2]-0.5*(Cs[2][0]*Cinv[2][0]+Cinv[2][0]*Cs[2][0]+Cs[2][2]*Cinv[0][0]+Cinv[2][2]*Cs[0][0]))-temp3*0.5*(Cinv[0][0]*Cinv[2][2]+Cinv[2][0]*Cinv[2][0]);
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

void MooneyRivlinModel2(IGAPoint pnt, const PetscScalar *U, PetscScalar (*F)[3], PetscScalar (*S)[3], PetscScalar (*D)[6], void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscReal c1 = user->c1;
  PetscReal c2 = user->c2;
  PetscReal kappa = user->kappa;

  // Interpolate the solution and gradient given U
  PetscScalar u[3];
  PetscScalar grad_u[3][3];
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);

  // F = I + u_{i,j}
  F[0][0] = 1+grad_u[0][0]; F[0][1] =   grad_u[0][1];  F[0][2] =   grad_u[0][2];
  F[1][0] =   grad_u[1][0]; F[1][1] = 1+grad_u[1][1];  F[1][2] =   grad_u[1][2];
  F[2][0] =   grad_u[2][0]; F[2][1] =   grad_u[2][1];  F[2][2] = 1+grad_u[2][2];

  // Finv
  PetscScalar Finv[3][3],J,Jinv;
  J  = F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]);
  J -= F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0]);
  J += F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
  Jinv = 1./J;
  Finv[0][0] =  (F[1][1]*F[2][2]-F[2][1]*F[1][2])*Jinv;
  Finv[1][0] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0])*Jinv;
  Finv[2][0] =  (F[1][0]*F[2][1]-F[2][0]*F[1][1])*Jinv;
  Finv[0][1] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1])*Jinv;
  Finv[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0])*Jinv;
  Finv[2][1] = -(F[0][0]*F[2][1]-F[2][0]*F[0][1])*Jinv;
  Finv[0][2] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1])*Jinv;
  Finv[1][2] = -(F[0][0]*F[1][2]-F[1][0]*F[0][2])*Jinv;
  Finv[2][2] =  (F[0][0]*F[1][1]-F[1][0]*F[0][1])*Jinv;

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

  PetscScalar C[3][3];
  C[0][0] = F[0][0]*F[0][0] + F[1][0]*F[1][0] + F[2][0]*F[2][0];
  C[0][1] = F[0][0]*F[0][1] + F[1][0]*F[1][1] + F[2][0]*F[2][1];
  C[0][2] = F[0][0]*F[0][2] + F[1][0]*F[1][2] + F[2][0]*F[2][2];
  C[1][0] = F[0][1]*F[0][0] + F[1][1]*F[1][0] + F[2][1]*F[2][0];
  C[1][1] = F[0][1]*F[0][1] + F[1][1]*F[1][1] + F[2][1]*F[2][1];
  C[1][2] = F[0][1]*F[0][2] + F[1][1]*F[1][2] + F[2][1]*F[2][2];
  C[2][0] = F[0][2]*F[0][0] + F[1][2]*F[1][0] + F[2][2]*F[2][0];
  C[2][1] = F[0][2]*F[0][1] + F[1][2]*F[1][1] + F[2][2]*F[2][1];
  C[2][2] = F[0][2]*F[0][2] + F[1][2]*F[1][2] + F[2][2]*F[2][2];

  PetscScalar trC = C[0][0]+C[1][1]+C[2][2];

  //Cs=C^2
  PetscScalar trCs = 0;
  trCs += C[0][0]*C[0][0] + C[0][1]*C[1][0] + C[0][2]*C[2][0];
  trCs += C[1][0]*C[0][1] + C[1][1]*C[1][1] + C[1][2]*C[2][1];
  trCs += C[2][0]*C[0][2] + C[2][1]*C[1][2] + C[2][2]*C[2][2];

  PetscScalar sinva = 0.5*(trC*trC - trCs);

  PetscScalar temp = pow(J,-2/3);
  PetscScalar tempD = (c1*trC+2*c2*temp*sinva);

  //Sij=2*a*deltaij+(J^2*(2*c+b*tr(Cinv))-d)Cinvij-b*J^2*Csij
  PetscScalar Siso[3][3];
  Siso[0][0]=2*temp*(c1+c2*temp*(trC-C[0][0])-(1/3)*tempD*Cinv[0][0]);
  Siso[0][1]=2*temp*(-c2*temp*C[0][1]-(1/3)*tempD*Cinv[0][1]);
  Siso[0][2]=2*temp*(-c2*temp*C[0][2]-(1/3)*tempD*Cinv[0][2]);
  Siso[1][0]=2*temp*(-c2*temp*C[1][0]-(1/3)*tempD*Cinv[1][0]);
  Siso[1][1]=2*temp*(c1+c2*temp*(trC-C[1][1])-(1/3)*tempD*Cinv[1][1]);
  Siso[1][2]=2*temp*(-c2*temp*C[1][2]-(1/3)*tempD*Cinv[1][2]);
  Siso[2][0]=2*temp*(-c2*temp*C[2][0]-(1/3)*tempD*Cinv[2][0]);
  Siso[2][1]=2*temp*(-c2*temp*C[2][1]-(1/3)*tempD*Cinv[2][1]);
  Siso[2][2]=2*temp*(c1+c2*temp*(trC-C[2][2])-(1/3)*tempD*Cinv[2][2]);

  S[0][0]=Siso[0][0]+kappa*J*(J-1)*Cinv[0][0];
  S[0][1]=Siso[0][1]+kappa*J*(J-1)*Cinv[0][1];
  S[0][2]=Siso[0][2]+kappa*J*(J-1)*Cinv[0][2];
  S[1][0]=Siso[1][0]+kappa*J*(J-1)*Cinv[1][0];
  S[1][1]=Siso[1][1]+kappa*J*(J-1)*Cinv[1][1];
  S[1][2]=Siso[1][2]+kappa*J*(J-1)*Cinv[1][2];
  S[2][0]=Siso[2][0]+kappa*J*(J-1)*Cinv[2][0];
  S[2][1]=Siso[2][1]+kappa*J*(J-1)*Cinv[2][1];
  S[2][2]=Siso[2][2]+kappa*J*(J-1)*Cinv[2][2];

  PetscScalar Diso[6][6];
  Diso[0][0]=(-2/3)*(Siso[0][0]*Cinv[0][0]+Cinv[0][0]*Siso[0][0])+(4/3)*temp*tempD*(0.5*(Cinv[0][0]*Cinv[0][0]+Cinv[0][0]*Cinv[0][0])-(1/3)*Cinv[0][0]*Cinv[0][0])+4*temp*temp*c2*(1-(1/3)*trC*(1*Cinv[0][0]+Cinv[0][0]*1)+(1/3)*(C[0][0]*Cinv[0][0]+Cinv[0][0]*C[0][0])+(2/9)*sinva*Cinv[0][0]*Cinv[0][0]-1.0);
  Diso[0][1]=(-2/3)*(Siso[0][0]*Cinv[1][1]+Cinv[0][0]*Siso[1][1])+(4/3)*temp*tempD*(0.5*(Cinv[0][1]*Cinv[0][1]+Cinv[0][1]*Cinv[0][1])-(1/3)*Cinv[0][0]*Cinv[1][1])+4*temp*temp*c2*(1-(1/3)*trC*(1*Cinv[1][1]+Cinv[0][0]*1)+(1/3)*(C[0][0]*Cinv[1][1]+Cinv[0][0]*C[1][1])+(2/9)*sinva*Cinv[0][0]*Cinv[1][1]-0.0);
  Diso[0][2]=(-2/3)*(Siso[0][0]*Cinv[2][2]+Cinv[0][0]*Siso[2][2])+(4/3)*temp*tempD*(0.5*(Cinv[0][2]*Cinv[0][2]+Cinv[0][2]*Cinv[0][2])-(1/3)*Cinv[0][0]*Cinv[2][2])+4*temp*temp*c2*(1-(1/3)*trC*(1*Cinv[2][2]+Cinv[0][0]*1)+(1/3)*(C[0][0]*Cinv[2][2]+Cinv[0][0]*C[2][2])+(2/9)*sinva*Cinv[0][0]*Cinv[2][2]-0.0);
  Diso[0][3]=(-2/3)*(Siso[0][0]*Cinv[0][1]+Cinv[0][0]*Siso[0][1])+(4/3)*temp*tempD*(0.5*(Cinv[0][0]*Cinv[0][1]+Cinv[0][1]*Cinv[0][0])-(1/3)*Cinv[0][0]*Cinv[0][1])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[0][1]+Cinv[0][0]*0)+(1/3)*(C[0][0]*Cinv[0][1]+Cinv[0][0]*C[0][1])+(2/9)*sinva*Cinv[0][0]*Cinv[0][1]-0.0);
  Diso[0][4]=(-2/3)*(Siso[0][0]*Cinv[1][2]+Cinv[0][0]*Siso[1][2])+(4/3)*temp*tempD*(0.5*(Cinv[0][1]*Cinv[0][2]+Cinv[0][2]*Cinv[0][1])-(1/3)*Cinv[0][0]*Cinv[1][2])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[1][2]+Cinv[0][0]*0)+(1/3)*(C[0][0]*Cinv[1][2]+Cinv[0][0]*C[1][2])+(2/9)*sinva*Cinv[0][0]*Cinv[1][2]-0.0);
  Diso[0][5]=(-2/3)*(Siso[0][0]*Cinv[0][2]+Cinv[0][0]*Siso[0][2])+(4/3)*temp*tempD*(0.5*(Cinv[0][0]*Cinv[0][2]+Cinv[0][2]*Cinv[0][0])-(1/3)*Cinv[0][0]*Cinv[0][2])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[0][2]+Cinv[0][0]*0)+(1/3)*(C[0][0]*Cinv[0][2]+Cinv[0][0]*C[0][2])+(2/9)*sinva*Cinv[0][0]*Cinv[0][2]-0.0);
  Diso[1][1]=(-2/3)*(Siso[1][1]*Cinv[1][1]+Cinv[1][1]*Siso[1][1])+(4/3)*temp*tempD*(0.5*(Cinv[1][1]*Cinv[1][1]+Cinv[1][1]*Cinv[1][1])-(1/3)*Cinv[1][1]*Cinv[1][1])+4*temp*temp*c2*(1-(1/3)*trC*(1*Cinv[1][1]+Cinv[1][1]*1)+(1/3)*(C[1][1]*Cinv[1][1]+Cinv[1][1]*C[1][1])+(2/9)*sinva*Cinv[1][1]*Cinv[1][1]-1.0);
  Diso[1][2]=(-2/3)*(Siso[1][1]*Cinv[2][2]+Cinv[1][1]*Siso[2][2])+(4/3)*temp*tempD*(0.5*(Cinv[1][2]*Cinv[1][2]+Cinv[1][2]*Cinv[1][2])-(1/3)*Cinv[1][1]*Cinv[2][2])+4*temp*temp*c2*(1-(1/3)*trC*(1*Cinv[2][2]+Cinv[1][1]*1)+(1/3)*(C[1][1]*Cinv[2][2]+Cinv[1][1]*C[2][2])+(2/9)*sinva*Cinv[1][1]*Cinv[2][2]-0.0);
  Diso[1][3]=(-2/3)*(Siso[1][1]*Cinv[0][1]+Cinv[1][1]*Siso[0][1])+(4/3)*temp*tempD*(0.5*(Cinv[1][0]*Cinv[1][1]+Cinv[1][1]*Cinv[1][0])-(1/3)*Cinv[1][1]*Cinv[0][1])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[0][1]+Cinv[1][1]*0)+(1/3)*(C[1][1]*Cinv[0][1]+Cinv[1][1]*C[0][1])+(2/9)*sinva*Cinv[1][1]*Cinv[0][1]-0.0);
  Diso[1][4]=(-2/3)*(Siso[1][1]*Cinv[1][2]+Cinv[1][1]*Siso[1][2])+(4/3)*temp*tempD*(0.5*(Cinv[1][1]*Cinv[1][2]+Cinv[1][2]*Cinv[1][1])-(1/3)*Cinv[1][1]*Cinv[1][2])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[1][2]+Cinv[1][1]*0)+(1/3)*(C[1][1]*Cinv[1][2]+Cinv[1][1]*C[1][2])+(2/9)*sinva*Cinv[1][1]*Cinv[1][2]-0.0);
  Diso[1][5]=(-2/3)*(Siso[1][1]*Cinv[0][2]+Cinv[1][1]*Siso[0][2])+(4/3)*temp*tempD*(0.5*(Cinv[1][0]*Cinv[1][2]+Cinv[1][2]*Cinv[1][0])-(1/3)*Cinv[1][1]*Cinv[0][2])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[0][2]+Cinv[1][1]*0)+(1/3)*(C[1][1]*Cinv[0][2]+Cinv[1][1]*C[0][2])+(2/9)*sinva*Cinv[1][1]*Cinv[0][2]-0.0);
  Diso[2][2]=(-2/3)*(Siso[2][2]*Cinv[2][2]+Cinv[2][2]*Siso[2][2])+(4/3)*temp*tempD*(0.5*(Cinv[2][2]*Cinv[2][2]+Cinv[2][2]*Cinv[2][2])-(1/3)*Cinv[2][2]*Cinv[2][2])+4*temp*temp*c2*(1-(1/3)*trC*(1*Cinv[2][2]+Cinv[2][2]*1)+(1/3)*(C[2][2]*Cinv[2][2]+Cinv[2][2]*C[2][2])+(2/9)*sinva*Cinv[2][2]*Cinv[2][2]-1.0);
  Diso[2][3]=(-2/3)*(Siso[2][2]*Cinv[0][1]+Cinv[2][2]*Siso[0][1])+(4/3)*temp*tempD*(0.5*(Cinv[2][0]*Cinv[2][1]+Cinv[2][1]*Cinv[2][0])-(1/3)*Cinv[2][2]*Cinv[0][1])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[0][1]+Cinv[2][2]*0)+(1/3)*(C[2][2]*Cinv[0][1]+Cinv[2][2]*C[0][1])+(2/9)*sinva*Cinv[2][2]*Cinv[0][1]-0.0);
  Diso[2][4]=(-2/3)*(Siso[2][2]*Cinv[1][2]+Cinv[2][2]*Siso[1][2])+(4/3)*temp*tempD*(0.5*(Cinv[2][1]*Cinv[2][2]+Cinv[2][2]*Cinv[2][1])-(1/3)*Cinv[2][2]*Cinv[1][2])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[1][2]+Cinv[2][2]*0)+(1/3)*(C[2][2]*Cinv[1][2]+Cinv[2][2]*C[1][2])+(2/9)*sinva*Cinv[2][2]*Cinv[1][2]-0.0);
  Diso[2][5]=(-2/3)*(Siso[2][2]*Cinv[0][2]+Cinv[2][2]*Siso[0][2])+(4/3)*temp*tempD*(0.5*(Cinv[2][0]*Cinv[2][2]+Cinv[2][2]*Cinv[2][0])-(1/3)*Cinv[2][2]*Cinv[0][2])+4*temp*temp*c2*(0-(1/3)*trC*(1*Cinv[0][2]+Cinv[2][2]*0)+(1/3)*(C[2][2]*Cinv[0][2]+Cinv[2][2]*C[0][2])+(2/9)*sinva*Cinv[2][2]*Cinv[0][2]-0.0);
  Diso[3][3]=(-2/3)*(Siso[0][1]*Cinv[0][1]+Cinv[0][1]*Siso[0][1])+(4/3)*temp*tempD*(0.5*(Cinv[0][0]*Cinv[1][1]+Cinv[0][1]*Cinv[1][0])-(1/3)*Cinv[0][1]*Cinv[0][1])+4*temp*temp*c2*(0-(1/3)*trC*(0*Cinv[0][1]+Cinv[0][1]*0)+(1/3)*(C[0][1]*Cinv[0][1]+Cinv[0][1]*C[0][1])+(2/9)*sinva*Cinv[0][1]*Cinv[0][1]-0.5);
  Diso[3][4]=(-2/3)*(Siso[0][1]*Cinv[1][2]+Cinv[0][1]*Siso[1][2])+(4/3)*temp*tempD*(0.5*(Cinv[0][1]*Cinv[1][2]+Cinv[0][2]*Cinv[1][1])-(1/3)*Cinv[0][1]*Cinv[1][2])+4*temp*temp*c2*(0-(1/3)*trC*(0*Cinv[1][2]+Cinv[0][1]*0)+(1/3)*(C[0][1]*Cinv[1][2]+Cinv[0][1]*C[1][2])+(2/9)*sinva*Cinv[0][1]*Cinv[1][2]-0.0);
  Diso[3][5]=(-2/3)*(Siso[0][1]*Cinv[0][2]+Cinv[0][1]*Siso[0][2])+(4/3)*temp*tempD*(0.5*(Cinv[0][0]*Cinv[1][2]+Cinv[0][2]*Cinv[1][0])-(1/3)*Cinv[0][1]*Cinv[0][2])+4*temp*temp*c2*(0-(1/3)*trC*(0*Cinv[0][2]+Cinv[0][1]*0)+(1/3)*(C[0][1]*Cinv[0][2]+Cinv[0][1]*C[0][2])+(2/9)*sinva*Cinv[0][1]*Cinv[0][2]-0.0);
  Diso[4][4]=(-2/3)*(Siso[1][2]*Cinv[1][2]+Cinv[1][2]*Siso[1][2])+(4/3)*temp*tempD*(0.5*(Cinv[1][1]*Cinv[2][2]+Cinv[1][2]*Cinv[2][1])-(1/3)*Cinv[1][2]*Cinv[1][2])+4*temp*temp*c2*(0-(1/3)*trC*(0*Cinv[1][2]+Cinv[1][2]*0)+(1/3)*(C[1][2]*Cinv[1][2]+Cinv[1][2]*C[1][2])+(2/9)*sinva*Cinv[1][2]*Cinv[1][2]-0.5);
  Diso[4][5]=(-2/3)*(Siso[1][2]*Cinv[0][2]+Cinv[1][2]*Siso[0][2])+(4/3)*temp*tempD*(0.5*(Cinv[1][0]*Cinv[2][2]+Cinv[1][2]*Cinv[2][0])-(1/3)*Cinv[1][2]*Cinv[0][2])+4*temp*temp*c2*(0-(1/3)*trC*(0*Cinv[0][2]+Cinv[1][2]*0)+(1/3)*(C[1][2]*Cinv[0][2]+Cinv[1][2]*C[0][2])+(2/9)*sinva*Cinv[1][2]*Cinv[0][2]-0.0);
  Diso[5][5]=(-2/3)*(Siso[0][2]*Cinv[0][2]+Cinv[0][2]*Siso[0][2])+(4/3)*temp*tempD*(0.5*(Cinv[0][0]*Cinv[2][2]+Cinv[0][2]*Cinv[2][0])-(1/3)*Cinv[0][2]*Cinv[0][2])+4*temp*temp*c2*(0-(1/3)*trC*(0*Cinv[0][2]+Cinv[0][2]*0)+(1/3)*(C[0][2]*Cinv[0][2]+Cinv[0][2]*C[0][2])+(2/9)*sinva*Cinv[0][2]*Cinv[0][2]-0.5);

  PetscScalar Dvol[6][6];
  PetscScalar temp1 = kappa*J*(2*J-1);
  PetscScalar temp2 = 2*kappa*J*(J-1);
  Dvol[0][0]=temp1*Cinv[0][0]*Cinv[0][0]-temp2*0.5*(Cinv[0][0]*Cinv[0][0]+Cinv[0][0]*Cinv[0][0]);
  Dvol[0][1]=temp1*Cinv[0][0]*Cinv[1][1]-temp2*0.5*(Cinv[0][1]*Cinv[1][0]+Cinv[1][0]*Cinv[0][1]);
  Dvol[0][2]=temp1*Cinv[0][0]*Cinv[2][2]-temp2*0.5*(Cinv[0][2]*Cinv[2][0]+Cinv[2][0]*Cinv[0][2]);
  Dvol[0][3]=temp1*Cinv[0][0]*Cinv[0][1]-temp2*0.5*(Cinv[0][0]*Cinv[1][0]+Cinv[1][0]*Cinv[0][0]);
  Dvol[0][4]=temp1*Cinv[0][0]*Cinv[1][2]-temp2*0.5*(Cinv[0][1]*Cinv[2][0]+Cinv[2][0]*Cinv[0][1]);
  Dvol[0][5]=temp1*Cinv[0][0]*Cinv[0][2]-temp2*0.5*(Cinv[0][0]*Cinv[2][0]+Cinv[2][0]*Cinv[0][0]);
  Dvol[1][1]=temp1*Cinv[1][1]*Cinv[1][1]-temp2*0.5*(Cinv[1][1]*Cinv[1][1]+Cinv[1][1]*Cinv[1][1]);
  Dvol[1][2]=temp1*Cinv[1][1]*Cinv[2][2]-temp2*0.5*(Cinv[1][2]*Cinv[2][1]+Cinv[2][1]*Cinv[1][2]);
  Dvol[1][3]=temp1*Cinv[1][1]*Cinv[0][1]-temp2*0.5*(Cinv[1][0]*Cinv[1][1]+Cinv[1][1]*Cinv[1][0]);
  Dvol[1][4]=temp1*Cinv[1][1]*Cinv[1][2]-temp2*0.5*(Cinv[1][1]*Cinv[2][1]+Cinv[2][1]*Cinv[1][1]);
  Dvol[1][5]=temp1*Cinv[1][1]*Cinv[0][2]-temp2*0.5*(Cinv[1][0]*Cinv[2][1]+Cinv[2][1]*Cinv[1][0]);
  Dvol[2][2]=temp1*Cinv[2][2]*Cinv[2][2]-temp2*0.5*(Cinv[2][2]*Cinv[2][2]+Cinv[2][2]*Cinv[2][2]);
  Dvol[2][3]=temp1*Cinv[2][2]*Cinv[0][1]-temp2*0.5*(Cinv[2][0]*Cinv[1][2]+Cinv[1][2]*Cinv[2][0]);
  Dvol[2][4]=temp1*Cinv[2][2]*Cinv[1][2]-temp2*0.5*(Cinv[2][1]*Cinv[2][2]+Cinv[2][2]*Cinv[2][1]);
  Dvol[2][5]=temp1*Cinv[2][2]*Cinv[0][2]-temp2*0.5*(Cinv[2][0]*Cinv[2][2]+Cinv[2][2]*Cinv[2][0]);
  Dvol[3][3]=temp1*Cinv[0][1]*Cinv[0][1]-temp2*0.5*(Cinv[0][0]*Cinv[1][1]+Cinv[1][0]*Cinv[1][0]);
  Dvol[3][4]=temp1*Cinv[0][1]*Cinv[1][2]-temp2*0.5*(Cinv[0][1]*Cinv[2][1]+Cinv[2][0]*Cinv[1][1]);
  Dvol[3][5]=temp1*Cinv[0][1]*Cinv[0][2]-temp2*0.5*(Cinv[0][0]*Cinv[2][1]+Cinv[2][0]*Cinv[1][0]);
  Dvol[4][4]=temp1*Cinv[1][2]*Cinv[1][2]-temp2*0.5*(Cinv[1][1]*Cinv[2][2]+Cinv[2][1]*Cinv[2][1]);
  Dvol[4][5]=temp1*Cinv[1][2]*Cinv[0][2]-temp2*0.5*(Cinv[1][0]*Cinv[2][2]+Cinv[2][1]*Cinv[2][0]);
  Dvol[5][5]=temp1*Cinv[0][2]*Cinv[0][2]-temp2*0.5*(Cinv[0][0]*Cinv[2][2]+Cinv[2][0]*Cinv[2][0]);


  //Dijkl = J^2(4c+2b*tr(Cinv))*Cinvij*Cinvkl-2bJ^2(Cinv^2ij*Cinvkl+Cinvij*Cinv^2kl-0.5*(Cinv^2li*Cinvjk+Cinvli*Cinv^2jk+Cinv^2lj*Cinvik+Cinvlj*Cinv^2ik))-(J^2*(4c+2btr(Cinv))-2d)*0.5*(Cinvik*Cinvlj+Cinvli*Cinvjk)
  D[0][0]=Diso[0][0]+Dvol[0][0];
  D[0][1]=Diso[0][1]+Dvol[0][1];
  D[0][2]=Diso[0][2]+Dvol[0][2];
  D[0][3]=Diso[0][3]+Dvol[0][3];
  D[0][4]=Diso[0][4]+Dvol[0][4];
  D[0][5]=Diso[0][5]+Dvol[0][5];
  D[1][1]=Diso[1][1]+Dvol[1][1];
  D[1][2]=Diso[1][2]+Dvol[1][2];
  D[1][3]=Diso[1][3]+Dvol[1][3];
  D[1][4]=Diso[1][4]+Dvol[1][4];
  D[1][5]=Diso[1][5]+Dvol[1][5];
  D[2][2]=Diso[2][2]+Dvol[2][2];
  D[2][3]=Diso[2][3]+Dvol[2][3];
  D[2][4]=Diso[2][4]+Dvol[2][4];
  D[2][5]=Diso[2][5]+Dvol[2][5];
  D[3][3]=Diso[3][3]+Dvol[3][3];
  D[3][4]=Diso[3][4]+Dvol[3][4];
  D[3][5]=Diso[3][5]+Dvol[3][5];
  D[4][4]=Diso[4][4]+Dvol[4][4];
  D[4][5]=Diso[4][5]+Dvol[4][5];
  D[5][5]=Diso[5][5]+Dvol[5][5];
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

void GeneralModel(IGAPoint pnt, const PetscScalar *U, PetscScalar (*F)[3], PetscScalar (*S)[3], PetscScalar (*D)[6], void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscReal c1 = user->c1;
  PetscReal c2 = user->c2;
  PetscReal kappa = user->kappa;

  // Interpolate the solution and gradient given U
  PetscScalar u[3];
  PetscScalar grad_u[3][3];
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);

  // F = I + u_{i,j}
  F[0][0] = 1+grad_u[0][0]; F[0][1] =   grad_u[0][1];  F[0][2] =   grad_u[0][2];
  F[1][0] =   grad_u[1][0]; F[1][1] = 1+grad_u[1][1];  F[1][2] =   grad_u[1][2];
  F[2][0] =   grad_u[2][0]; F[2][1] =   grad_u[2][1];  F[2][2] = 1+grad_u[2][2];

  // Finv
  PetscScalar Finv[3][3],J,Jinv;
  J  = F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]);
  J -= F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0]);
  J += F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]);
  Jinv = 1./J;
  Finv[0][0] =  (F[1][1]*F[2][2]-F[2][1]*F[1][2])*Jinv;
  Finv[1][0] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0])*Jinv;
  Finv[2][0] =  (F[1][0]*F[2][1]-F[2][0]*F[1][1])*Jinv;
  Finv[0][1] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1])*Jinv;
  Finv[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0])*Jinv;
  Finv[2][1] = -(F[0][0]*F[2][1]-F[2][0]*F[0][1])*Jinv;
  Finv[0][2] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1])*Jinv;
  Finv[1][2] = -(F[0][0]*F[1][2]-F[1][0]*F[0][2])*Jinv;
  Finv[2][2] =  (F[0][0]*F[1][1]-F[1][0]*F[0][1])*Jinv;

  PetscScalar C[3][3];
  C[0][0] = F[0][0]*F[0][0] + F[1][0]*F[1][0] + F[2][0]*F[2][0];
  C[0][1] = F[0][0]*F[0][1] + F[1][0]*F[1][1] + F[2][0]*F[2][1];
  C[0][2] = F[0][0]*F[0][2] + F[1][0]*F[1][2] + F[2][0]*F[2][2];
  C[1][0] = F[0][1]*F[0][0] + F[1][1]*F[1][0] + F[2][1]*F[2][0];
  C[1][1] = F[0][1]*F[0][1] + F[1][1]*F[1][1] + F[2][1]*F[2][1];
  C[1][2] = F[0][1]*F[0][2] + F[1][1]*F[1][2] + F[2][1]*F[2][2];
  C[2][0] = F[0][2]*F[0][0] + F[1][2]*F[1][0] + F[2][2]*F[2][0];
  C[2][1] = F[0][2]*F[0][1] + F[1][2]*F[1][1] + F[2][2]*F[2][1];
  C[2][2] = F[0][2]*F[0][2] + F[1][2]*F[1][2] + F[2][2]*F[2][2];

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

  PetscScalar P[6][6];
  PetscScalar tempJ = pow(J,-2/3);
  P[0][0] = tempJ*(1.0-(1/3)*Cinv[0][0]*C[0][0]);
  P[0][1] = tempJ*(0.0-(1/3)*Cinv[0][0]*C[1][1]);
  P[0][2] = tempJ*(0.0-(1/3)*Cinv[0][0]*C[2][2]);
  P[0][3] = 2*tempJ*(0.0-(1/3)*Cinv[0][0]*C[0][1]);
  P[0][4] = 2*tempJ*(0.0-(1/3)*Cinv[0][0]*C[1][2]);
  P[0][5] = 2*tempJ*(0.0-(1/3)*Cinv[0][0]*C[0][2]);
  P[1][0] = tempJ*(0.0-(1/3)*Cinv[1][1]*C[0][0]);
  P[1][1] = tempJ*(1.0-(1/3)*Cinv[1][1]*C[1][1]);
  P[1][2] = tempJ*(0.0-(1/3)*Cinv[1][1]*C[2][2]);
  P[1][3] = 2*tempJ*(0.0-(1/3)*Cinv[1][1]*C[0][1]);
  P[1][4] = 2*tempJ*(0.0-(1/3)*Cinv[1][1]*C[1][2]);
  P[1][5] = 2*tempJ*(0.0-(1/3)*Cinv[1][1]*C[0][2]);
  P[2][0] = tempJ*(0.0-(1/3)*Cinv[2][2]*C[0][0]);
  P[2][1] = tempJ*(0.0-(1/3)*Cinv[2][2]*C[1][1]);
  P[2][2] = tempJ*(1.0-(1/3)*Cinv[2][2]*C[2][2]);
  P[2][3] = 2*tempJ*(0.0-(1/3)*Cinv[2][2]*C[0][1]);
  P[2][4] = 2*tempJ*(0.0-(1/3)*Cinv[2][2]*C[1][2]);
  P[2][5] = 2*tempJ*(0.0-(1/3)*Cinv[2][2]*C[0][2]);
  P[3][0] = tempJ*(0.0-(1/3)*Cinv[0][1]*C[0][0]);
  P[3][1] = tempJ*(0.0-(1/3)*Cinv[0][1]*C[1][1]);
  P[3][2] = tempJ*(0.0-(1/3)*Cinv[0][1]*C[2][2]);
  P[3][3] = 2*tempJ*(0.5-(1/3)*Cinv[0][1]*C[0][1]);
  P[3][4] = 2*tempJ*(0.0-(1/3)*Cinv[0][1]*C[1][2]);
  P[3][5] = 2*tempJ*(0.0-(1/3)*Cinv[0][1]*C[0][2]);
  P[4][0] = tempJ*(0.0-(1/3)*Cinv[1][2]*C[0][0]);
  P[4][1] = tempJ*(0.0-(1/3)*Cinv[1][2]*C[1][1]);
  P[4][2] = tempJ*(0.0-(1/3)*Cinv[1][2]*C[2][2]);
  P[4][3] = 2*tempJ*(0.0-(1/3)*Cinv[1][2]*C[0][1]);
  P[4][4] = 2*tempJ*(0.5-(1/3)*Cinv[1][2]*C[1][2]);
  P[4][5] = 2*tempJ*(0.0-(1/3)*Cinv[1][2]*C[0][2]);
  P[5][0] = tempJ*(0.0-(1/3)*Cinv[0][2]*C[0][0]);
  P[5][1] = tempJ*(0.0-(1/3)*Cinv[0][2]*C[1][1]);
  P[5][2] = tempJ*(0.0-(1/3)*Cinv[0][2]*C[2][2]);
  P[5][3] = 2*tempJ*(0.0-(1/3)*Cinv[0][2]*C[0][1]);
  P[5][4] = 2*tempJ*(0.0-(1/3)*Cinv[0][2]*C[1][2]);
  P[5][5] = 2*tempJ*(0.5-(1/3)*Cinv[0][2]*C[0][2]);

  PetscScalar trC = C[0][0]+C[1][1]+C[2][2];

  PetscScalar Shat[6];
  Shat[0] = 2*(c1+c2*tempJ*(trC-C[0][0])); //Shat[0][0]
  Shat[1] = 2*(c1+c2*tempJ*(trC-C[1][1])); //Shat[1][1]
  Shat[2] = 2*(c1+c2*tempJ*(trC-C[2][2])); //Shat[2][2]
  Shat[3] = -2*c2*tempJ*C[0][1];  //Shat[0][1]
  Shat[4] = -2*c2*tempJ*C[1][2];  //Shat[1][2]
  Shat[5] = -2*c2*tempJ*C[0][2];  //Shat[0][2]

  PetscScalar Siso[6];
  Siso[0]=P[0][0]*Shat[0] + P[0][1]*Shat[1] + P[0][2]*Shat[2] + P[0][3]*Shat[3] + P[0][4]*Shat[4] + P[0][5]*Shat[5];
  Siso[1]=P[1][0]*Shat[0] + P[1][1]*Shat[1] + P[1][2]*Shat[2] + P[1][3]*Shat[3] + P[1][4]*Shat[4] + P[1][5]*Shat[5];
  Siso[2]=P[2][0]*Shat[0] + P[2][1]*Shat[1] + P[2][2]*Shat[2] + P[2][3]*Shat[3] + P[2][4]*Shat[4] + P[2][5]*Shat[5];
  Siso[3]=P[3][0]*Shat[0] + P[3][1]*Shat[1] + P[3][2]*Shat[2] + P[3][3]*Shat[3] + P[3][4]*Shat[4] + P[3][5]*Shat[5];
  Siso[4]=P[4][0]*Shat[0] + P[4][1]*Shat[1] + P[4][2]*Shat[2] + P[4][3]*Shat[3] + P[4][4]*Shat[4] + P[4][5]*Shat[5];
  Siso[5]=P[5][0]*Shat[0] + P[5][1]*Shat[1] + P[5][2]*Shat[2] + P[5][3]*Shat[3] + P[5][4]*Shat[4] + P[5][5]*Shat[5];

  PetscScalar tempdJ = kappa*(J-1);
  PetscScalar tempd2J2 = kappa;
  PetscScalar Svol[6];
  Svol[0] = tempdJ*J*Cinv[0][0];
  Svol[1] = tempdJ*J*Cinv[1][1];
  Svol[2] = tempdJ*J*Cinv[2][2];
  Svol[3] = tempdJ*J*Cinv[0][1];
  Svol[4] = tempdJ*J*Cinv[1][2];
  Svol[5] = tempdJ*J*Cinv[0][2];

  S[0][0]= Siso[0]+Svol[0];
  S[0][1]= Siso[3]+Svol[3];
  S[0][2]= Siso[5]+Svol[5];
  S[1][0]= S[0][1];
  S[1][1]= Siso[1]+Svol[1];
  S[1][2]= Siso[4]+Svol[4];
  S[2][0]= S[0][2];
  S[2][1]= S[1][2];;
  S[2][2]= Siso[2]+Svol[2];

  PetscScalar tempX = Shat[0]*C[0][0]+Shat[1]*C[1][1]+Shat[2]*C[2][2]+2*Shat[3]*C[0][1]+2*Shat[4]*C[1][2]+2*Shat[5]*C[0][2];

  PetscScalar Pbar[6][6];
  Pbar[0][0]=tempJ*(0.5*(Cinv[0][0]*Cinv[0][0]+Cinv[0][0]*Cinv[0][0])-(1/3)*Cinv[0][0]*Cinv[0][0]);
  Pbar[0][1]=tempJ*(0.5*(Cinv[0][1]*Cinv[0][1]+Cinv[0][1]*Cinv[0][1])-(1/3)*Cinv[0][0]*Cinv[1][1]);
  Pbar[0][2]=tempJ*(0.5*(Cinv[0][2]*Cinv[0][2]+Cinv[0][2]*Cinv[0][2])-(1/3)*Cinv[0][0]*Cinv[2][2]);
  Pbar[0][3]=tempJ*(0.5*(Cinv[0][0]*Cinv[0][1]+Cinv[0][1]*Cinv[0][0])-(1/3)*Cinv[0][0]*Cinv[0][1]);
  Pbar[0][4]=tempJ*(0.5*(Cinv[0][1]*Cinv[0][2]+Cinv[0][2]*Cinv[0][1])-(1/3)*Cinv[0][0]*Cinv[1][2]);
  Pbar[0][5]=tempJ*(0.5*(Cinv[0][0]*Cinv[0][2]+Cinv[0][2]*Cinv[0][0])-(1/3)*Cinv[0][0]*Cinv[0][2]);
  Pbar[1][1]=tempJ*(0.5*(Cinv[1][1]*Cinv[1][1]+Cinv[1][1]*Cinv[1][1])-(1/3)*Cinv[1][1]*Cinv[1][1]);
  Pbar[1][2]=tempJ*(0.5*(Cinv[1][2]*Cinv[1][2]+Cinv[1][2]*Cinv[1][2])-(1/3)*Cinv[1][1]*Cinv[2][2]);
  Pbar[1][3]=tempJ*(0.5*(Cinv[1][0]*Cinv[1][1]+Cinv[1][1]*Cinv[1][0])-(1/3)*Cinv[1][1]*Cinv[0][1]);
  Pbar[1][4]=tempJ*(0.5*(Cinv[1][1]*Cinv[1][2]+Cinv[1][2]*Cinv[1][1])-(1/3)*Cinv[1][1]*Cinv[1][2]);
  Pbar[1][5]=tempJ*(0.5*(Cinv[1][0]*Cinv[1][2]+Cinv[1][2]*Cinv[1][0])-(1/3)*Cinv[1][1]*Cinv[0][2]);
  Pbar[2][2]=tempJ*(0.5*(Cinv[2][2]*Cinv[2][2]+Cinv[2][2]*Cinv[2][2])-(1/3)*Cinv[2][2]*Cinv[2][2]);
  Pbar[2][3]=tempJ*(0.5*(Cinv[2][0]*Cinv[2][1]+Cinv[2][1]*Cinv[2][0])-(1/3)*Cinv[2][2]*Cinv[0][1]);
  Pbar[2][4]=tempJ*(0.5*(Cinv[2][1]*Cinv[2][2]+Cinv[2][2]*Cinv[2][1])-(1/3)*Cinv[2][2]*Cinv[1][2]);
  Pbar[2][5]=tempJ*(0.5*(Cinv[2][0]*Cinv[2][2]+Cinv[2][2]*Cinv[2][0])-(1/3)*Cinv[2][2]*Cinv[0][2]);
  Pbar[3][3]=tempJ*(0.5*(Cinv[0][0]*Cinv[1][1]+Cinv[0][1]*Cinv[1][0])-(1/3)*Cinv[0][1]*Cinv[0][1]);
  Pbar[3][4]=tempJ*(0.5*(Cinv[0][1]*Cinv[1][2]+Cinv[0][2]*Cinv[1][1])-(1/3)*Cinv[0][1]*Cinv[1][2]);
  Pbar[3][5]=tempJ*(0.5*(Cinv[0][0]*Cinv[1][2]+Cinv[0][2]*Cinv[1][0])-(1/3)*Cinv[0][1]*Cinv[0][2]);
  Pbar[4][4]=tempJ*(0.5*(Cinv[1][1]*Cinv[2][2]+Cinv[1][2]*Cinv[2][1])-(1/3)*Cinv[1][2]*Cinv[1][2]);
  Pbar[4][5]=tempJ*(0.5*(Cinv[1][0]*Cinv[2][2]+Cinv[1][2]*Cinv[2][0])-(1/3)*Cinv[1][2]*Cinv[0][2]);
  Pbar[5][5]=tempJ*(0.5*(Cinv[0][0]*Cinv[2][2]+Cinv[0][2]*Cinv[2][0])-(1/3)*Cinv[0][2]*Cinv[0][2]);

  PetscScalar Dhat[6][6];
  Dhat[0][0] = 0; //D[0][0][0][0]
  Dhat[0][1] = 4*c2; //D[0][0][1][1]
  Dhat[0][2] = 4*c2; //D[0][0][2][2]
  Dhat[0][3] = 0; //D[0][0][0][1]
  Dhat[0][4] = 0; //D[0][0][1][2]
  Dhat[0][5] = 0; //D[0][0][0][2]
  Dhat[1][1] = 0; //D[1][1][1][1]
  Dhat[1][2] = 4*c2; //D[1][1][2][2]
  Dhat[1][3] = 0; //D[1][1][0][1]
  Dhat[1][4] = 0; //D[1][1][1][2]
  Dhat[1][5] = 0; //D[1][1][0][2]
  Dhat[2][2] = 0; //D[2][2][2][2]
  Dhat[2][3] = 0; //D[2][2][0][1]
  Dhat[2][4] = 0; //D[2][2][1][2]
  Dhat[2][5] = 0; //D[2][2][0][2]
  Dhat[3][3] = -2*c2; //D[0][1][0][1]
  Dhat[3][4] = 0; //D[0][1][1][2]
  Dhat[3][5] = 0; //D[0][1][0][2]
  Dhat[4][4] = -2*c2; //D[1][2][1][2]
  Dhat[4][5] = 0; //D[1][2][0][2]
  Dhat[5][5] = -2*c2; //D[0][2][0][2]
  Dhat[1][0]=Dhat[0][1];
  Dhat[2][0]=Dhat[0][2];
  Dhat[3][0]=Dhat[0][3];
  Dhat[4][0]=Dhat[0][4];
  Dhat[5][0]=Dhat[0][5];
  Dhat[2][1]=Dhat[1][2];
  Dhat[3][1]=Dhat[1][3];
  Dhat[4][1]=Dhat[1][4];
  Dhat[5][1]=Dhat[1][5];
  Dhat[3][2]=Dhat[2][3];
  Dhat[4][2]=Dhat[2][4];
  Dhat[5][2]=Dhat[2][5];
  Dhat[4][3]=Dhat[3][4];
  Dhat[5][3]=Dhat[3][5];
  Dhat[5][4]=Dhat[4][5];


  PetscScalar Diso[6][6];
  Diso[0][0] = P[0][0]*(Dhat[0][0]*P[0][0] + Dhat[1][0]*P[0][1] + Dhat[2][0]*P[0][2] + Dhat[3][0]*P[0][3] + Dhat[4][0]*P[0][4] + Dhat[5][0]*P[0][5]) + P[0][1]*(Dhat[0][1]*P[0][0] + Dhat[1][1]*P[0][1] + Dhat[2][1]*P[0][2] + Dhat[3][1]*P[0][3] + Dhat[4][1]*P[0][4] + Dhat[5][1]*P[0][5]) + P[0][2]*(Dhat[0][2]*P[0][0] + Dhat[1][2]*P[0][1] + Dhat[2][2]*P[0][2] + Dhat[3][2]*P[0][3] + Dhat[4][2]*P[0][4] + Dhat[5][2]*P[0][5]) + P[0][3]*(Dhat[0][3]*P[0][0] + Dhat[1][3]*P[0][1] + Dhat[2][3]*P[0][2] + Dhat[3][3]*P[0][3] + Dhat[4][3]*P[0][4] + Dhat[5][3]*P[0][5]) + P[0][4]*(Dhat[0][4]*P[0][0] + Dhat[1][4]*P[0][1] + Dhat[2][4]*P[0][2] + Dhat[3][4]*P[0][3] + Dhat[4][4]*P[0][4] + Dhat[5][4]*P[0][5]) + P[0][5]*(Dhat[0][5]*P[0][0] + Dhat[1][5]*P[0][1] + Dhat[2][5]*P[0][2] + Dhat[3][5]*P[0][3] + Dhat[4][5]*P[0][4] + Dhat[5][5]*P[0][5]);
  Diso[0][1] = P[1][0]*(Dhat[0][0]*P[0][0] + Dhat[1][0]*P[0][1] + Dhat[2][0]*P[0][2] + Dhat[3][0]*P[0][3] + Dhat[4][0]*P[0][4] + Dhat[5][0]*P[0][5]) + P[1][1]*(Dhat[0][1]*P[0][0] + Dhat[1][1]*P[0][1] + Dhat[2][1]*P[0][2] + Dhat[3][1]*P[0][3] + Dhat[4][1]*P[0][4] + Dhat[5][1]*P[0][5]) + P[1][2]*(Dhat[0][2]*P[0][0] + Dhat[1][2]*P[0][1] + Dhat[2][2]*P[0][2] + Dhat[3][2]*P[0][3] + Dhat[4][2]*P[0][4] + Dhat[5][2]*P[0][5]) + P[1][3]*(Dhat[0][3]*P[0][0] + Dhat[1][3]*P[0][1] + Dhat[2][3]*P[0][2] + Dhat[3][3]*P[0][3] + Dhat[4][3]*P[0][4] + Dhat[5][3]*P[0][5]) + P[1][4]*(Dhat[0][4]*P[0][0] + Dhat[1][4]*P[0][1] + Dhat[2][4]*P[0][2] + Dhat[3][4]*P[0][3] + Dhat[4][4]*P[0][4] + Dhat[5][4]*P[0][5]) + P[1][5]*(Dhat[0][5]*P[0][0] + Dhat[1][5]*P[0][1] + Dhat[2][5]*P[0][2] + Dhat[3][5]*P[0][3] + Dhat[4][5]*P[0][4] + Dhat[5][5]*P[0][5]);
  Diso[0][2] = P[2][0]*(Dhat[0][0]*P[0][0] + Dhat[1][0]*P[0][1] + Dhat[2][0]*P[0][2] + Dhat[3][0]*P[0][3] + Dhat[4][0]*P[0][4] + Dhat[5][0]*P[0][5]) + P[2][1]*(Dhat[0][1]*P[0][0] + Dhat[1][1]*P[0][1] + Dhat[2][1]*P[0][2] + Dhat[3][1]*P[0][3] + Dhat[4][1]*P[0][4] + Dhat[5][1]*P[0][5]) + P[2][2]*(Dhat[0][2]*P[0][0] + Dhat[1][2]*P[0][1] + Dhat[2][2]*P[0][2] + Dhat[3][2]*P[0][3] + Dhat[4][2]*P[0][4] + Dhat[5][2]*P[0][5]) + P[2][3]*(Dhat[0][3]*P[0][0] + Dhat[1][3]*P[0][1] + Dhat[2][3]*P[0][2] + Dhat[3][3]*P[0][3] + Dhat[4][3]*P[0][4] + Dhat[5][3]*P[0][5]) + P[2][4]*(Dhat[0][4]*P[0][0] + Dhat[1][4]*P[0][1] + Dhat[2][4]*P[0][2] + Dhat[3][4]*P[0][3] + Dhat[4][4]*P[0][4] + Dhat[5][4]*P[0][5]) + P[2][5]*(Dhat[0][5]*P[0][0] + Dhat[1][5]*P[0][1] + Dhat[2][5]*P[0][2] + Dhat[3][5]*P[0][3] + Dhat[4][5]*P[0][4] + Dhat[5][5]*P[0][5]);
  Diso[0][3] = P[3][0]*(Dhat[0][0]*P[0][0] + Dhat[1][0]*P[0][1] + Dhat[2][0]*P[0][2] + Dhat[3][0]*P[0][3] + Dhat[4][0]*P[0][4] + Dhat[5][0]*P[0][5]) + P[3][1]*(Dhat[0][1]*P[0][0] + Dhat[1][1]*P[0][1] + Dhat[2][1]*P[0][2] + Dhat[3][1]*P[0][3] + Dhat[4][1]*P[0][4] + Dhat[5][1]*P[0][5]) + P[3][2]*(Dhat[0][2]*P[0][0] + Dhat[1][2]*P[0][1] + Dhat[2][2]*P[0][2] + Dhat[3][2]*P[0][3] + Dhat[4][2]*P[0][4] + Dhat[5][2]*P[0][5]) + P[3][3]*(Dhat[0][3]*P[0][0] + Dhat[1][3]*P[0][1] + Dhat[2][3]*P[0][2] + Dhat[3][3]*P[0][3] + Dhat[4][3]*P[0][4] + Dhat[5][3]*P[0][5]) + P[3][4]*(Dhat[0][4]*P[0][0] + Dhat[1][4]*P[0][1] + Dhat[2][4]*P[0][2] + Dhat[3][4]*P[0][3] + Dhat[4][4]*P[0][4] + Dhat[5][4]*P[0][5]) + P[3][5]*(Dhat[0][5]*P[0][0] + Dhat[1][5]*P[0][1] + Dhat[2][5]*P[0][2] + Dhat[3][5]*P[0][3] + Dhat[4][5]*P[0][4] + Dhat[5][5]*P[0][5]);
  Diso[0][4] = P[4][0]*(Dhat[0][0]*P[0][0] + Dhat[1][0]*P[0][1] + Dhat[2][0]*P[0][2] + Dhat[3][0]*P[0][3] + Dhat[4][0]*P[0][4] + Dhat[5][0]*P[0][5]) + P[4][1]*(Dhat[0][1]*P[0][0] + Dhat[1][1]*P[0][1] + Dhat[2][1]*P[0][2] + Dhat[3][1]*P[0][3] + Dhat[4][1]*P[0][4] + Dhat[5][1]*P[0][5]) + P[4][2]*(Dhat[0][2]*P[0][0] + Dhat[1][2]*P[0][1] + Dhat[2][2]*P[0][2] + Dhat[3][2]*P[0][3] + Dhat[4][2]*P[0][4] + Dhat[5][2]*P[0][5]) + P[4][3]*(Dhat[0][3]*P[0][0] + Dhat[1][3]*P[0][1] + Dhat[2][3]*P[0][2] + Dhat[3][3]*P[0][3] + Dhat[4][3]*P[0][4] + Dhat[5][3]*P[0][5]) + P[4][4]*(Dhat[0][4]*P[0][0] + Dhat[1][4]*P[0][1] + Dhat[2][4]*P[0][2] + Dhat[3][4]*P[0][3] + Dhat[4][4]*P[0][4] + Dhat[5][4]*P[0][5]) + P[4][5]*(Dhat[0][5]*P[0][0] + Dhat[1][5]*P[0][1] + Dhat[2][5]*P[0][2] + Dhat[3][5]*P[0][3] + Dhat[4][5]*P[0][4] + Dhat[5][5]*P[0][5]);
  Diso[0][5] = P[5][0]*(Dhat[0][0]*P[0][0] + Dhat[1][0]*P[0][1] + Dhat[2][0]*P[0][2] + Dhat[3][0]*P[0][3] + Dhat[4][0]*P[0][4] + Dhat[5][0]*P[0][5]) + P[5][1]*(Dhat[0][1]*P[0][0] + Dhat[1][1]*P[0][1] + Dhat[2][1]*P[0][2] + Dhat[3][1]*P[0][3] + Dhat[4][1]*P[0][4] + Dhat[5][1]*P[0][5]) + P[5][2]*(Dhat[0][2]*P[0][0] + Dhat[1][2]*P[0][1] + Dhat[2][2]*P[0][2] + Dhat[3][2]*P[0][3] + Dhat[4][2]*P[0][4] + Dhat[5][2]*P[0][5]) + P[5][3]*(Dhat[0][3]*P[0][0] + Dhat[1][3]*P[0][1] + Dhat[2][3]*P[0][2] + Dhat[3][3]*P[0][3] + Dhat[4][3]*P[0][4] + Dhat[5][3]*P[0][5]) + P[5][4]*(Dhat[0][4]*P[0][0] + Dhat[1][4]*P[0][1] + Dhat[2][4]*P[0][2] + Dhat[3][4]*P[0][3] + Dhat[4][4]*P[0][4] + Dhat[5][4]*P[0][5]) + P[5][5]*(Dhat[0][5]*P[0][0] + Dhat[1][5]*P[0][1] + Dhat[2][5]*P[0][2] + Dhat[3][5]*P[0][3] + Dhat[4][5]*P[0][4] + Dhat[5][5]*P[0][5]);
  Diso[1][1] = P[1][0]*(Dhat[0][0]*P[1][0] + Dhat[1][0]*P[1][1] + Dhat[2][0]*P[1][2] + Dhat[3][0]*P[1][3] + Dhat[4][0]*P[1][4] + Dhat[5][0]*P[1][5]) + P[1][1]*(Dhat[0][1]*P[1][0] + Dhat[1][1]*P[1][1] + Dhat[2][1]*P[1][2] + Dhat[3][1]*P[1][3] + Dhat[4][1]*P[1][4] + Dhat[5][1]*P[1][5]) + P[1][2]*(Dhat[0][2]*P[1][0] + Dhat[1][2]*P[1][1] + Dhat[2][2]*P[1][2] + Dhat[3][2]*P[1][3] + Dhat[4][2]*P[1][4] + Dhat[5][2]*P[1][5]) + P[1][3]*(Dhat[0][3]*P[1][0] + Dhat[1][3]*P[1][1] + Dhat[2][3]*P[1][2] + Dhat[3][3]*P[1][3] + Dhat[4][3]*P[1][4] + Dhat[5][3]*P[1][5]) + P[1][4]*(Dhat[0][4]*P[1][0] + Dhat[1][4]*P[1][1] + Dhat[2][4]*P[1][2] + Dhat[3][4]*P[1][3] + Dhat[4][4]*P[1][4] + Dhat[5][4]*P[1][5]) + P[1][5]*(Dhat[0][5]*P[1][0] + Dhat[1][5]*P[1][1] + Dhat[2][5]*P[1][2] + Dhat[3][5]*P[1][3] + Dhat[4][5]*P[1][4] + Dhat[5][5]*P[1][5]);
  Diso[1][2] = P[2][0]*(Dhat[0][0]*P[1][0] + Dhat[1][0]*P[1][1] + Dhat[2][0]*P[1][2] + Dhat[3][0]*P[1][3] + Dhat[4][0]*P[1][4] + Dhat[5][0]*P[1][5]) + P[2][1]*(Dhat[0][1]*P[1][0] + Dhat[1][1]*P[1][1] + Dhat[2][1]*P[1][2] + Dhat[3][1]*P[1][3] + Dhat[4][1]*P[1][4] + Dhat[5][1]*P[1][5]) + P[2][2]*(Dhat[0][2]*P[1][0] + Dhat[1][2]*P[1][1] + Dhat[2][2]*P[1][2] + Dhat[3][2]*P[1][3] + Dhat[4][2]*P[1][4] + Dhat[5][2]*P[1][5]) + P[2][3]*(Dhat[0][3]*P[1][0] + Dhat[1][3]*P[1][1] + Dhat[2][3]*P[1][2] + Dhat[3][3]*P[1][3] + Dhat[4][3]*P[1][4] + Dhat[5][3]*P[1][5]) + P[2][4]*(Dhat[0][4]*P[1][0] + Dhat[1][4]*P[1][1] + Dhat[2][4]*P[1][2] + Dhat[3][4]*P[1][3] + Dhat[4][4]*P[1][4] + Dhat[5][4]*P[1][5]) + P[2][5]*(Dhat[0][5]*P[1][0] + Dhat[1][5]*P[1][1] + Dhat[2][5]*P[1][2] + Dhat[3][5]*P[1][3] + Dhat[4][5]*P[1][4] + Dhat[5][5]*P[1][5]);
  Diso[1][3] = P[3][0]*(Dhat[0][0]*P[1][0] + Dhat[1][0]*P[1][1] + Dhat[2][0]*P[1][2] + Dhat[3][0]*P[1][3] + Dhat[4][0]*P[1][4] + Dhat[5][0]*P[1][5]) + P[3][1]*(Dhat[0][1]*P[1][0] + Dhat[1][1]*P[1][1] + Dhat[2][1]*P[1][2] + Dhat[3][1]*P[1][3] + Dhat[4][1]*P[1][4] + Dhat[5][1]*P[1][5]) + P[3][2]*(Dhat[0][2]*P[1][0] + Dhat[1][2]*P[1][1] + Dhat[2][2]*P[1][2] + Dhat[3][2]*P[1][3] + Dhat[4][2]*P[1][4] + Dhat[5][2]*P[1][5]) + P[3][3]*(Dhat[0][3]*P[1][0] + Dhat[1][3]*P[1][1] + Dhat[2][3]*P[1][2] + Dhat[3][3]*P[1][3] + Dhat[4][3]*P[1][4] + Dhat[5][3]*P[1][5]) + P[3][4]*(Dhat[0][4]*P[1][0] + Dhat[1][4]*P[1][1] + Dhat[2][4]*P[1][2] + Dhat[3][4]*P[1][3] + Dhat[4][4]*P[1][4] + Dhat[5][4]*P[1][5]) + P[3][5]*(Dhat[0][5]*P[1][0] + Dhat[1][5]*P[1][1] + Dhat[2][5]*P[1][2] + Dhat[3][5]*P[1][3] + Dhat[4][5]*P[1][4] + Dhat[5][5]*P[1][5]);
  Diso[1][4] = P[4][0]*(Dhat[0][0]*P[1][0] + Dhat[1][0]*P[1][1] + Dhat[2][0]*P[1][2] + Dhat[3][0]*P[1][3] + Dhat[4][0]*P[1][4] + Dhat[5][0]*P[1][5]) + P[4][1]*(Dhat[0][1]*P[1][0] + Dhat[1][1]*P[1][1] + Dhat[2][1]*P[1][2] + Dhat[3][1]*P[1][3] + Dhat[4][1]*P[1][4] + Dhat[5][1]*P[1][5]) + P[4][2]*(Dhat[0][2]*P[1][0] + Dhat[1][2]*P[1][1] + Dhat[2][2]*P[1][2] + Dhat[3][2]*P[1][3] + Dhat[4][2]*P[1][4] + Dhat[5][2]*P[1][5]) + P[4][3]*(Dhat[0][3]*P[1][0] + Dhat[1][3]*P[1][1] + Dhat[2][3]*P[1][2] + Dhat[3][3]*P[1][3] + Dhat[4][3]*P[1][4] + Dhat[5][3]*P[1][5]) + P[4][4]*(Dhat[0][4]*P[1][0] + Dhat[1][4]*P[1][1] + Dhat[2][4]*P[1][2] + Dhat[3][4]*P[1][3] + Dhat[4][4]*P[1][4] + Dhat[5][4]*P[1][5]) + P[4][5]*(Dhat[0][5]*P[1][0] + Dhat[1][5]*P[1][1] + Dhat[2][5]*P[1][2] + Dhat[3][5]*P[1][3] + Dhat[4][5]*P[1][4] + Dhat[5][5]*P[1][5]);
  Diso[1][5] = P[5][0]*(Dhat[0][0]*P[1][0] + Dhat[1][0]*P[1][1] + Dhat[2][0]*P[1][2] + Dhat[3][0]*P[1][3] + Dhat[4][0]*P[1][4] + Dhat[5][0]*P[1][5]) + P[5][1]*(Dhat[0][1]*P[1][0] + Dhat[1][1]*P[1][1] + Dhat[2][1]*P[1][2] + Dhat[3][1]*P[1][3] + Dhat[4][1]*P[1][4] + Dhat[5][1]*P[1][5]) + P[5][2]*(Dhat[0][2]*P[1][0] + Dhat[1][2]*P[1][1] + Dhat[2][2]*P[1][2] + Dhat[3][2]*P[1][3] + Dhat[4][2]*P[1][4] + Dhat[5][2]*P[1][5]) + P[5][3]*(Dhat[0][3]*P[1][0] + Dhat[1][3]*P[1][1] + Dhat[2][3]*P[1][2] + Dhat[3][3]*P[1][3] + Dhat[4][3]*P[1][4] + Dhat[5][3]*P[1][5]) + P[5][4]*(Dhat[0][4]*P[1][0] + Dhat[1][4]*P[1][1] + Dhat[2][4]*P[1][2] + Dhat[3][4]*P[1][3] + Dhat[4][4]*P[1][4] + Dhat[5][4]*P[1][5]) + P[5][5]*(Dhat[0][5]*P[1][0] + Dhat[1][5]*P[1][1] + Dhat[2][5]*P[1][2] + Dhat[3][5]*P[1][3] + Dhat[4][5]*P[1][4] + Dhat[5][5]*P[1][5]);
  Diso[2][2] = P[2][0]*(Dhat[0][0]*P[2][0] + Dhat[1][0]*P[2][1] + Dhat[2][0]*P[2][2] + Dhat[3][0]*P[2][3] + Dhat[4][0]*P[2][4] + Dhat[5][0]*P[2][5]) + P[2][1]*(Dhat[0][1]*P[2][0] + Dhat[1][1]*P[2][1] + Dhat[2][1]*P[2][2] + Dhat[3][1]*P[2][3] + Dhat[4][1]*P[2][4] + Dhat[5][1]*P[2][5]) + P[2][2]*(Dhat[0][2]*P[2][0] + Dhat[1][2]*P[2][1] + Dhat[2][2]*P[2][2] + Dhat[3][2]*P[2][3] + Dhat[4][2]*P[2][4] + Dhat[5][2]*P[2][5]) + P[2][3]*(Dhat[0][3]*P[2][0] + Dhat[1][3]*P[2][1] + Dhat[2][3]*P[2][2] + Dhat[3][3]*P[2][3] + Dhat[4][3]*P[2][4] + Dhat[5][3]*P[2][5]) + P[2][4]*(Dhat[0][4]*P[2][0] + Dhat[1][4]*P[2][1] + Dhat[2][4]*P[2][2] + Dhat[3][4]*P[2][3] + Dhat[4][4]*P[2][4] + Dhat[5][4]*P[2][5]) + P[2][5]*(Dhat[0][5]*P[2][0] + Dhat[1][5]*P[2][1] + Dhat[2][5]*P[2][2] + Dhat[3][5]*P[2][3] + Dhat[4][5]*P[2][4] + Dhat[5][5]*P[2][5]);
  Diso[2][3] = P[3][0]*(Dhat[0][0]*P[2][0] + Dhat[1][0]*P[2][1] + Dhat[2][0]*P[2][2] + Dhat[3][0]*P[2][3] + Dhat[4][0]*P[2][4] + Dhat[5][0]*P[2][5]) + P[3][1]*(Dhat[0][1]*P[2][0] + Dhat[1][1]*P[2][1] + Dhat[2][1]*P[2][2] + Dhat[3][1]*P[2][3] + Dhat[4][1]*P[2][4] + Dhat[5][1]*P[2][5]) + P[3][2]*(Dhat[0][2]*P[2][0] + Dhat[1][2]*P[2][1] + Dhat[2][2]*P[2][2] + Dhat[3][2]*P[2][3] + Dhat[4][2]*P[2][4] + Dhat[5][2]*P[2][5]) + P[3][3]*(Dhat[0][3]*P[2][0] + Dhat[1][3]*P[2][1] + Dhat[2][3]*P[2][2] + Dhat[3][3]*P[2][3] + Dhat[4][3]*P[2][4] + Dhat[5][3]*P[2][5]) + P[3][4]*(Dhat[0][4]*P[2][0] + Dhat[1][4]*P[2][1] + Dhat[2][4]*P[2][2] + Dhat[3][4]*P[2][3] + Dhat[4][4]*P[2][4] + Dhat[5][4]*P[2][5]) + P[3][5]*(Dhat[0][5]*P[2][0] + Dhat[1][5]*P[2][1] + Dhat[2][5]*P[2][2] + Dhat[3][5]*P[2][3] + Dhat[4][5]*P[2][4] + Dhat[5][5]*P[2][5]);
  Diso[2][4] = P[4][0]*(Dhat[0][0]*P[2][0] + Dhat[1][0]*P[2][1] + Dhat[2][0]*P[2][2] + Dhat[3][0]*P[2][3] + Dhat[4][0]*P[2][4] + Dhat[5][0]*P[2][5]) + P[4][1]*(Dhat[0][1]*P[2][0] + Dhat[1][1]*P[2][1] + Dhat[2][1]*P[2][2] + Dhat[3][1]*P[2][3] + Dhat[4][1]*P[2][4] + Dhat[5][1]*P[2][5]) + P[4][2]*(Dhat[0][2]*P[2][0] + Dhat[1][2]*P[2][1] + Dhat[2][2]*P[2][2] + Dhat[3][2]*P[2][3] + Dhat[4][2]*P[2][4] + Dhat[5][2]*P[2][5]) + P[4][3]*(Dhat[0][3]*P[2][0] + Dhat[1][3]*P[2][1] + Dhat[2][3]*P[2][2] + Dhat[3][3]*P[2][3] + Dhat[4][3]*P[2][4] + Dhat[5][3]*P[2][5]) + P[4][4]*(Dhat[0][4]*P[2][0] + Dhat[1][4]*P[2][1] + Dhat[2][4]*P[2][2] + Dhat[3][4]*P[2][3] + Dhat[4][4]*P[2][4] + Dhat[5][4]*P[2][5]) + P[4][5]*(Dhat[0][5]*P[2][0] + Dhat[1][5]*P[2][1] + Dhat[2][5]*P[2][2] + Dhat[3][5]*P[2][3] + Dhat[4][5]*P[2][4] + Dhat[5][5]*P[2][5]);
  Diso[2][5] = P[5][0]*(Dhat[0][0]*P[2][0] + Dhat[1][0]*P[2][1] + Dhat[2][0]*P[2][2] + Dhat[3][0]*P[2][3] + Dhat[4][0]*P[2][4] + Dhat[5][0]*P[2][5]) + P[5][1]*(Dhat[0][1]*P[2][0] + Dhat[1][1]*P[2][1] + Dhat[2][1]*P[2][2] + Dhat[3][1]*P[2][3] + Dhat[4][1]*P[2][4] + Dhat[5][1]*P[2][5]) + P[5][2]*(Dhat[0][2]*P[2][0] + Dhat[1][2]*P[2][1] + Dhat[2][2]*P[2][2] + Dhat[3][2]*P[2][3] + Dhat[4][2]*P[2][4] + Dhat[5][2]*P[2][5]) + P[5][3]*(Dhat[0][3]*P[2][0] + Dhat[1][3]*P[2][1] + Dhat[2][3]*P[2][2] + Dhat[3][3]*P[2][3] + Dhat[4][3]*P[2][4] + Dhat[5][3]*P[2][5]) + P[5][4]*(Dhat[0][4]*P[2][0] + Dhat[1][4]*P[2][1] + Dhat[2][4]*P[2][2] + Dhat[3][4]*P[2][3] + Dhat[4][4]*P[2][4] + Dhat[5][4]*P[2][5]) + P[5][5]*(Dhat[0][5]*P[2][0] + Dhat[1][5]*P[2][1] + Dhat[2][5]*P[2][2] + Dhat[3][5]*P[2][3] + Dhat[4][5]*P[2][4] + Dhat[5][5]*P[2][5]);
  Diso[3][3] = P[3][0]*(Dhat[0][0]*P[3][0] + Dhat[1][0]*P[3][1] + Dhat[2][0]*P[3][2] + Dhat[3][0]*P[3][3] + Dhat[4][0]*P[3][4] + Dhat[5][0]*P[3][5]) + P[3][1]*(Dhat[0][1]*P[3][0] + Dhat[1][1]*P[3][1] + Dhat[2][1]*P[3][2] + Dhat[3][1]*P[3][3] + Dhat[4][1]*P[3][4] + Dhat[5][1]*P[3][5]) + P[3][2]*(Dhat[0][2]*P[3][0] + Dhat[1][2]*P[3][1] + Dhat[2][2]*P[3][2] + Dhat[3][2]*P[3][3] + Dhat[4][2]*P[3][4] + Dhat[5][2]*P[3][5]) + P[3][3]*(Dhat[0][3]*P[3][0] + Dhat[1][3]*P[3][1] + Dhat[2][3]*P[3][2] + Dhat[3][3]*P[3][3] + Dhat[4][3]*P[3][4] + Dhat[5][3]*P[3][5]) + P[3][4]*(Dhat[0][4]*P[3][0] + Dhat[1][4]*P[3][1] + Dhat[2][4]*P[3][2] + Dhat[3][4]*P[3][3] + Dhat[4][4]*P[3][4] + Dhat[5][4]*P[3][5]) + P[3][5]*(Dhat[0][5]*P[3][0] + Dhat[1][5]*P[3][1] + Dhat[2][5]*P[3][2] + Dhat[3][5]*P[3][3] + Dhat[4][5]*P[3][4] + Dhat[5][5]*P[3][5]);
  Diso[3][4] = P[4][0]*(Dhat[0][0]*P[3][0] + Dhat[1][0]*P[3][1] + Dhat[2][0]*P[3][2] + Dhat[3][0]*P[3][3] + Dhat[4][0]*P[3][4] + Dhat[5][0]*P[3][5]) + P[4][1]*(Dhat[0][1]*P[3][0] + Dhat[1][1]*P[3][1] + Dhat[2][1]*P[3][2] + Dhat[3][1]*P[3][3] + Dhat[4][1]*P[3][4] + Dhat[5][1]*P[3][5]) + P[4][2]*(Dhat[0][2]*P[3][0] + Dhat[1][2]*P[3][1] + Dhat[2][2]*P[3][2] + Dhat[3][2]*P[3][3] + Dhat[4][2]*P[3][4] + Dhat[5][2]*P[3][5]) + P[4][3]*(Dhat[0][3]*P[3][0] + Dhat[1][3]*P[3][1] + Dhat[2][3]*P[3][2] + Dhat[3][3]*P[3][3] + Dhat[4][3]*P[3][4] + Dhat[5][3]*P[3][5]) + P[4][4]*(Dhat[0][4]*P[3][0] + Dhat[1][4]*P[3][1] + Dhat[2][4]*P[3][2] + Dhat[3][4]*P[3][3] + Dhat[4][4]*P[3][4] + Dhat[5][4]*P[3][5]) + P[4][5]*(Dhat[0][5]*P[3][0] + Dhat[1][5]*P[3][1] + Dhat[2][5]*P[3][2] + Dhat[3][5]*P[3][3] + Dhat[4][5]*P[3][4] + Dhat[5][5]*P[3][5]);
  Diso[3][5] = P[5][0]*(Dhat[0][0]*P[3][0] + Dhat[1][0]*P[3][1] + Dhat[2][0]*P[3][2] + Dhat[3][0]*P[3][3] + Dhat[4][0]*P[3][4] + Dhat[5][0]*P[3][5]) + P[5][1]*(Dhat[0][1]*P[3][0] + Dhat[1][1]*P[3][1] + Dhat[2][1]*P[3][2] + Dhat[3][1]*P[3][3] + Dhat[4][1]*P[3][4] + Dhat[5][1]*P[3][5]) + P[5][2]*(Dhat[0][2]*P[3][0] + Dhat[1][2]*P[3][1] + Dhat[2][2]*P[3][2] + Dhat[3][2]*P[3][3] + Dhat[4][2]*P[3][4] + Dhat[5][2]*P[3][5]) + P[5][3]*(Dhat[0][3]*P[3][0] + Dhat[1][3]*P[3][1] + Dhat[2][3]*P[3][2] + Dhat[3][3]*P[3][3] + Dhat[4][3]*P[3][4] + Dhat[5][3]*P[3][5]) + P[5][4]*(Dhat[0][4]*P[3][0] + Dhat[1][4]*P[3][1] + Dhat[2][4]*P[3][2] + Dhat[3][4]*P[3][3] + Dhat[4][4]*P[3][4] + Dhat[5][4]*P[3][5]) + P[5][5]*(Dhat[0][5]*P[3][0] + Dhat[1][5]*P[3][1] + Dhat[2][5]*P[3][2] + Dhat[3][5]*P[3][3] + Dhat[4][5]*P[3][4] + Dhat[5][5]*P[3][5]);
  Diso[4][4] = P[4][0]*(Dhat[0][0]*P[4][0] + Dhat[1][0]*P[4][1] + Dhat[2][0]*P[4][2] + Dhat[3][0]*P[4][3] + Dhat[4][0]*P[4][4] + Dhat[5][0]*P[4][5]) + P[4][1]*(Dhat[0][1]*P[4][0] + Dhat[1][1]*P[4][1] + Dhat[2][1]*P[4][2] + Dhat[3][1]*P[4][3] + Dhat[4][1]*P[4][4] + Dhat[5][1]*P[4][5]) + P[4][2]*(Dhat[0][2]*P[4][0] + Dhat[1][2]*P[4][1] + Dhat[2][2]*P[4][2] + Dhat[3][2]*P[4][3] + Dhat[4][2]*P[4][4] + Dhat[5][2]*P[4][5]) + P[4][3]*(Dhat[0][3]*P[4][0] + Dhat[1][3]*P[4][1] + Dhat[2][3]*P[4][2] + Dhat[3][3]*P[4][3] + Dhat[4][3]*P[4][4] + Dhat[5][3]*P[4][5]) + P[4][4]*(Dhat[0][4]*P[4][0] + Dhat[1][4]*P[4][1] + Dhat[2][4]*P[4][2] + Dhat[3][4]*P[4][3] + Dhat[4][4]*P[4][4] + Dhat[5][4]*P[4][5]) + P[4][5]*(Dhat[0][5]*P[4][0] + Dhat[1][5]*P[4][1] + Dhat[2][5]*P[4][2] + Dhat[3][5]*P[4][3] + Dhat[4][5]*P[4][4] + Dhat[5][5]*P[4][5]);
  Diso[4][5] = P[5][0]*(Dhat[0][0]*P[4][0] + Dhat[1][0]*P[4][1] + Dhat[2][0]*P[4][2] + Dhat[3][0]*P[4][3] + Dhat[4][0]*P[4][4] + Dhat[5][0]*P[4][5]) + P[5][1]*(Dhat[0][1]*P[4][0] + Dhat[1][1]*P[4][1] + Dhat[2][1]*P[4][2] + Dhat[3][1]*P[4][3] + Dhat[4][1]*P[4][4] + Dhat[5][1]*P[4][5]) + P[5][2]*(Dhat[0][2]*P[4][0] + Dhat[1][2]*P[4][1] + Dhat[2][2]*P[4][2] + Dhat[3][2]*P[4][3] + Dhat[4][2]*P[4][4] + Dhat[5][2]*P[4][5]) + P[5][3]*(Dhat[0][3]*P[4][0] + Dhat[1][3]*P[4][1] + Dhat[2][3]*P[4][2] + Dhat[3][3]*P[4][3] + Dhat[4][3]*P[4][4] + Dhat[5][3]*P[4][5]) + P[5][4]*(Dhat[0][4]*P[4][0] + Dhat[1][4]*P[4][1] + Dhat[2][4]*P[4][2] + Dhat[3][4]*P[4][3] + Dhat[4][4]*P[4][4] + Dhat[5][4]*P[4][5]) + P[5][5]*(Dhat[0][5]*P[4][0] + Dhat[1][5]*P[4][1] + Dhat[2][5]*P[4][2] + Dhat[3][5]*P[4][3] + Dhat[4][5]*P[4][4] + Dhat[5][5]*P[4][5]);
  Diso[5][5] = P[5][0]*(Dhat[0][0]*P[5][0] + Dhat[1][0]*P[5][1] + Dhat[2][0]*P[5][2] + Dhat[3][0]*P[5][3] + Dhat[4][0]*P[5][4] + Dhat[5][0]*P[5][5]) + P[5][1]*(Dhat[0][1]*P[5][0] + Dhat[1][1]*P[5][1] + Dhat[2][1]*P[5][2] + Dhat[3][1]*P[5][3] + Dhat[4][1]*P[5][4] + Dhat[5][1]*P[5][5]) + P[5][2]*(Dhat[0][2]*P[5][0] + Dhat[1][2]*P[5][1] + Dhat[2][2]*P[5][2] + Dhat[3][2]*P[5][3] + Dhat[4][2]*P[5][4] + Dhat[5][2]*P[5][5]) + P[5][3]*(Dhat[0][3]*P[5][0] + Dhat[1][3]*P[5][1] + Dhat[2][3]*P[5][2] + Dhat[3][3]*P[5][3] + Dhat[4][3]*P[5][4] + Dhat[5][3]*P[5][5]) + P[5][4]*(Dhat[0][4]*P[5][0] + Dhat[1][4]*P[5][1] + Dhat[2][4]*P[5][2] + Dhat[3][4]*P[5][3] + Dhat[4][4]*P[5][4] + Dhat[5][4]*P[5][5]) + P[5][5]*(Dhat[0][5]*P[5][0] + Dhat[1][5]*P[5][1] + Dhat[2][5]*P[5][2] + Dhat[3][5]*P[5][3] + Dhat[4][5]*P[5][4] + Dhat[5][5]*P[5][5]);

  Diso[0][0] += (2/3)*tempX*Pbar[0][0]-(2/3)*(Cinv[0][0]*Siso[0]+Siso[0]*Cinv[0][0]);
  Diso[0][1] += (2/3)*tempX*Pbar[0][1]-(2/3)*(Cinv[0][0]*Siso[1]+Siso[0]*Cinv[1][1]);
  Diso[0][2] += (2/3)*tempX*Pbar[0][2]-(2/3)*(Cinv[0][0]*Siso[2]+Siso[0]*Cinv[2][2]);
  Diso[0][3] += (2/3)*tempX*Pbar[0][3]-(2/3)*(Cinv[0][0]*Siso[3]+Siso[0]*Cinv[0][1]);
  Diso[0][4] += (2/3)*tempX*Pbar[0][4]-(2/3)*(Cinv[0][0]*Siso[4]+Siso[0]*Cinv[1][2]);
  Diso[0][5] += (2/3)*tempX*Pbar[0][5]-(2/3)*(Cinv[0][0]*Siso[5]+Siso[0]*Cinv[0][2]);
  Diso[1][1] += (2/3)*tempX*Pbar[1][1]-(2/3)*(Cinv[1][1]*Siso[1]+Siso[1]*Cinv[1][1]);
  Diso[1][2] += (2/3)*tempX*Pbar[1][2]-(2/3)*(Cinv[1][1]*Siso[2]+Siso[1]*Cinv[2][2]);
  Diso[1][3] += (2/3)*tempX*Pbar[1][3]-(2/3)*(Cinv[1][1]*Siso[3]+Siso[1]*Cinv[0][1]);
  Diso[1][4] += (2/3)*tempX*Pbar[1][4]-(2/3)*(Cinv[1][1]*Siso[4]+Siso[1]*Cinv[1][2]);
  Diso[1][5] += (2/3)*tempX*Pbar[1][5]-(2/3)*(Cinv[1][1]*Siso[5]+Siso[1]*Cinv[0][2]);
  Diso[2][2] += (2/3)*tempX*Pbar[2][2]-(2/3)*(Cinv[2][2]*Siso[2]+Siso[2]*Cinv[2][2]);
  Diso[2][3] += (2/3)*tempX*Pbar[2][3]-(2/3)*(Cinv[2][2]*Siso[3]+Siso[2]*Cinv[0][1]);
  Diso[2][4] += (2/3)*tempX*Pbar[2][4]-(2/3)*(Cinv[2][2]*Siso[4]+Siso[2]*Cinv[1][2]);
  Diso[2][5] += (2/3)*tempX*Pbar[2][5]-(2/3)*(Cinv[2][2]*Siso[5]+Siso[2]*Cinv[0][2]);
  Diso[3][3] += (2/3)*tempX*Pbar[3][3]-(2/3)*(Cinv[0][1]*Siso[3]+Siso[3]*Cinv[0][1]);
  Diso[3][4] += (2/3)*tempX*Pbar[3][4]-(2/3)*(Cinv[0][1]*Siso[4]+Siso[3]*Cinv[1][2]);
  Diso[3][5] += (2/3)*tempX*Pbar[3][5]-(2/3)*(Cinv[0][1]*Siso[5]+Siso[3]*Cinv[0][2]);
  Diso[4][4] += (2/3)*tempX*Pbar[4][4]-(2/3)*(Cinv[1][2]*Siso[4]+Siso[4]*Cinv[1][2]);
  Diso[4][5] += (2/3)*tempX*Pbar[4][5]-(2/3)*(Cinv[1][2]*Siso[5]+Siso[4]*Cinv[0][2]);
  Diso[5][5] += (2/3)*tempX*Pbar[5][5]-(2/3)*(Cinv[0][2]*Siso[5]+Siso[5]*Cinv[0][2]);

  PetscScalar Dvol[6][6];
  PetscScalar tempvol = J*J*tempd2J2+J*tempdJ;
  Dvol[0][0] = tempvol*Cinv[0][0]*Cinv[0][0]-2*J*tempdJ*0.5*(Cinv[0][0]*Cinv[0][0]+Cinv[0][0]*Cinv[0][0]);
  Dvol[0][1] = tempvol*Cinv[0][0]*Cinv[1][1]-2*J*tempdJ*0.5*(Cinv[0][1]*Cinv[0][1]+Cinv[0][1]*Cinv[0][1]);
  Dvol[0][2] = tempvol*Cinv[0][0]*Cinv[2][2]-2*J*tempdJ*0.5*(Cinv[0][2]*Cinv[0][2]+Cinv[0][2]*Cinv[0][2]);
  Dvol[0][3] = tempvol*Cinv[0][0]*Cinv[0][1]-2*J*tempdJ*0.5*(Cinv[0][0]*Cinv[0][1]+Cinv[0][1]*Cinv[0][0]);
  Dvol[0][4] = tempvol*Cinv[0][0]*Cinv[1][2]-2*J*tempdJ*0.5*(Cinv[0][1]*Cinv[0][2]+Cinv[0][2]*Cinv[0][1]);
  Dvol[0][5] = tempvol*Cinv[0][0]*Cinv[0][2]-2*J*tempdJ*0.5*(Cinv[0][0]*Cinv[0][2]+Cinv[0][2]*Cinv[0][0]);
  Dvol[1][1] = tempvol*Cinv[1][1]*Cinv[1][1]-2*J*tempdJ*0.5*(Cinv[1][1]*Cinv[1][1]+Cinv[1][1]*Cinv[1][1]);
  Dvol[1][2] = tempvol*Cinv[1][1]*Cinv[2][2]-2*J*tempdJ*0.5*(Cinv[1][2]*Cinv[1][2]+Cinv[1][2]*Cinv[1][2]);
  Dvol[1][3] = tempvol*Cinv[1][1]*Cinv[0][1]-2*J*tempdJ*0.5*(Cinv[1][0]*Cinv[1][1]+Cinv[1][1]*Cinv[1][0]);
  Dvol[1][4] = tempvol*Cinv[1][1]*Cinv[1][2]-2*J*tempdJ*0.5*(Cinv[1][1]*Cinv[1][2]+Cinv[1][2]*Cinv[1][1]);
  Dvol[1][5] = tempvol*Cinv[1][1]*Cinv[0][2]-2*J*tempdJ*0.5*(Cinv[1][0]*Cinv[1][2]+Cinv[1][2]*Cinv[1][0]);
  Dvol[2][2] = tempvol*Cinv[2][2]*Cinv[2][2]-2*J*tempdJ*0.5*(Cinv[2][2]*Cinv[2][2]+Cinv[2][2]*Cinv[2][2]);
  Dvol[2][3] = tempvol*Cinv[2][2]*Cinv[0][1]-2*J*tempdJ*0.5*(Cinv[2][0]*Cinv[2][1]+Cinv[2][1]*Cinv[2][0]);
  Dvol[2][4] = tempvol*Cinv[2][2]*Cinv[1][2]-2*J*tempdJ*0.5*(Cinv[2][1]*Cinv[2][2]+Cinv[2][2]*Cinv[2][1]);
  Dvol[2][5] = tempvol*Cinv[2][2]*Cinv[0][2]-2*J*tempdJ*0.5*(Cinv[2][0]*Cinv[2][2]+Cinv[2][2]*Cinv[2][0]);
  Dvol[3][3] = tempvol*Cinv[0][1]*Cinv[0][1]-2*J*tempdJ*0.5*(Cinv[0][0]*Cinv[1][1]+Cinv[0][1]*Cinv[1][0]);
  Dvol[3][4] = tempvol*Cinv[0][1]*Cinv[1][2]-2*J*tempdJ*0.5*(Cinv[0][1]*Cinv[1][2]+Cinv[0][2]*Cinv[1][1]);
  Dvol[3][5] = tempvol*Cinv[0][1]*Cinv[0][2]-2*J*tempdJ*0.5*(Cinv[0][0]*Cinv[1][2]+Cinv[0][2]*Cinv[1][0]);
  Dvol[4][4] = tempvol*Cinv[1][2]*Cinv[1][2]-2*J*tempdJ*0.5*(Cinv[1][1]*Cinv[2][2]+Cinv[1][2]*Cinv[2][1]);
  Dvol[4][5] = tempvol*Cinv[1][2]*Cinv[0][2]-2*J*tempdJ*0.5*(Cinv[1][0]*Cinv[2][2]+Cinv[1][2]*Cinv[2][0]);
  Dvol[5][5] = tempvol*Cinv[0][2]*Cinv[0][2]-2*J*tempdJ*0.5*(Cinv[0][0]*Cinv[2][2]+Cinv[0][2]*Cinv[2][0]);

  D[0][0]=Diso[0][0]+Dvol[0][0];
  D[0][1]=Diso[0][1]+Dvol[0][1];
  D[0][2]=Diso[0][2]+Dvol[0][2];
  D[0][3]=Diso[0][3]+Dvol[0][3];
  D[0][4]=Diso[0][4]+Dvol[0][4];
  D[0][5]=Diso[0][5]+Dvol[0][5];
  D[1][1]=Diso[1][1]+Dvol[1][1];
  D[1][2]=Diso[1][2]+Dvol[1][2];
  D[1][3]=Diso[1][3]+Dvol[1][3];
  D[1][4]=Diso[1][4]+Dvol[1][4];
  D[1][5]=Diso[1][5]+Dvol[1][5];
  D[2][2]=Diso[2][2]+Dvol[2][2];
  D[2][3]=Diso[2][3]+Dvol[2][3];
  D[2][4]=Diso[2][4]+Dvol[2][4];
  D[2][5]=Diso[2][5]+Dvol[2][5];
  D[3][3]=Diso[3][3]+Dvol[3][3];
  D[3][4]=Diso[3][4]+Dvol[3][4];
  D[3][5]=Diso[3][5]+Dvol[3][5];
  D[4][4]=Diso[4][4]+Dvol[4][4];
  D[4][5]=Diso[4][5]+Dvol[4][5];
  D[5][5]=Diso[5][5]+Dvol[5][5];
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

void DeltaE(PetscScalar Nx, PetscScalar Ny, PetscScalar Nz, PetscScalar (*F)[3], PetscScalar (*B)[3])
{
  // Given F and basis values, returns B
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


PetscErrorCode Residual(IGAPoint pnt,const PetscScalar *U,PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  // call user model
  PetscScalar F[3][3],S[3][3],D[6][6],B[6][3];
  user->model(pnt,U,F,S,D,ctx);

  // Get basis function gradients
  PetscReal (*N1)[3] = (PetscReal (*)[3]) pnt->shape[1];

  PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
  PetscInt a,nen=pnt->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];
    DeltaE(Na_x,Na_y,Na_z,F,B);
    R[a][0] = B[0][0]*S[0][0]+B[1][0]*S[1][1]+B[2][0]*S[2][2]+B[3][0]*S[0][1]+B[4][0]*S[1][2]+B[5][0]*S[0][2];
    R[a][1] = B[0][1]*S[0][0]+B[1][1]*S[1][1]+B[2][1]*S[2][2]+B[3][1]*S[0][1]+B[4][1]*S[1][2]+B[5][1]*S[0][2];
    R[a][2] = B[0][2]*S[0][0]+B[1][2]*S[1][1]+B[2][2]*S[2][2]+B[3][2]*S[0][1]+B[4][2]*S[1][2]+B[5][2]*S[0][2];
  }
  return 0;
}

PetscErrorCode Jacobian(IGAPoint pnt,const PetscScalar *U,PetscScalar *Je,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  // Call user model
  PetscScalar F[3][3],S[3][3],D[6][6];
  user->model(pnt,U,F,S,D,ctx);

  // Get basis function gradients
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
    DeltaE(Na_x,Na_y,Na_z,F,Ba);

    for (b=0; b<nen; b++) {

      PetscReal Nb_x  = N1[b][0];
      PetscReal Nb_y  = N1[b][1];
      PetscReal Nb_z  = N1[b][2];
      DeltaE(Nb_x,Nb_y,Nb_z,F,Bb);

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

      G=Na_x*(S[0][0]*Nb_x + S[0][1]*Nb_y + S[0][2]*Nb_z) +
        Na_y*(S[1][0]*Nb_x + S[1][1]*Nb_y + S[1][2]*Nb_z) +
        Na_z*(S[2][0]*Nb_x + S[2][1]*Nb_y + S[2][2]*Nb_z);

      K[a][0][b][0] = G+Chi[0][0];
      K[a][1][b][0] =   Chi[1][0];
      K[a][2][b][0] =   Chi[2][0];

      K[a][0][b][1] =   Chi[0][1];
      K[a][1][b][1] = G+Chi[1][1];
      K[a][2][b][1] =   Chi[2][1];

      K[a][0][b][2] =   Chi[0][2];
      K[a][1][b][2] =   Chi[1][2];
      K[a][2][b][2] = G+Chi[2][2];
    }
  }

  return 0;
}

int main(int argc, char *argv[])
{
  // Initialization of PETSc
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Application specific data (defaults to Aluminum)
  AppCtx user;
  PetscScalar E  = 70.0e9;
  PetscScalar nu = 0.35;
  user.model     = GeneralModel;
  PetscBool NeoHook,StVenant,MooneyR1,MooneyR2;
  NeoHook = StVenant = MooneyR1 = MooneyR2 = PETSC_FALSE;
  PetscInt nsteps = 1;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","HyperElasticity Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-neohook","Use the NeoHookean constitutive model",__FILE__,NeoHook,&NeoHook,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mooneyr1","Use the MooneyRivlin1 constitutive model",__FILE__,MooneyR1,&MooneyR1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mooneyr2","Use the MooneyRivlin2 constitutive model",__FILE__,MooneyR2,&MooneyR2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-stvenant","Use the StVenant constitutive model",__FILE__,StVenant,&StVenant,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nsteps","Number of load steps to take",__FILE__,nsteps,&nsteps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.lambda = E*nu/(1.+nu)/(1.-2.*nu);
  user.mu     = 0.5*E/(1.+nu);
  user.c1     = user.mu*0.5;
  user.c2     = 0;
  user.kappa  = 0;
  if(NeoHook){
    user.model = NeoHookeanModel;
  }
  if(StVenant){
    user.model = StVenantModel;
  }
  if(MooneyR1){
    user.model = MooneyRivlinModel1;
    user.a = 0.5*user.mu;
    user.b = 0;
    user.c = user.lambda*0.25;
    user.d = user.lambda*0.5+user.mu;
  }
  if(MooneyR2){
    user.c1 = user.mu*0.5;
    user.c2 = 0;
    user.kappa = 0;
  }

  // Initialize the discretization
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  if(iga->geometry == 0 && nsteps > 1){
    SETERRQ(PETSC_COMM_WORLD,
            PETSC_ERR_ARG_OUTOFRANGE,
            "You must specify a geometry to use an updated Lagrangian approach");
  }

  // Set boundary conditions
  //   u = [0,0,0] @ x = [0,:,:]
  ierr = IGASetBoundaryValue(iga,0,0,0,0.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,0,0,1,0.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,0,0,2,0.0);CHKERRQ(ierr);
  //   ux = 1 @ x = [1,:,:]
  ierr = IGASetBoundaryValue(iga,0,1,0,0.1/((PetscScalar)nsteps));CHKERRQ(ierr);

  // Setup the nonlinear solver
  SNES snes;
  Vec U,Utotal;
  ierr = IGASetFormFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormJacobian(iga,Jacobian,&user);CHKERRQ(ierr);
  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&Utotal);CHKERRQ(ierr);
  ierr = VecZeroEntries(Utotal);CHKERRQ(ierr);

  // Load stepping
  PetscInt step;
  for(step=0;step<nsteps;step++){

    PetscPrintf(PETSC_COMM_WORLD,"%d Load Step\n",step);

    // Solve step
    ierr = VecZeroEntries(U);CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,U);CHKERRQ(ierr);

    // Store total displacement
    ierr = VecAXPY(Utotal,1.0,U);CHKERRQ(ierr);

    // Update the geometry
    if(iga->geometry){
      Vec localU;
      const PetscScalar *arrayU;
      ierr = IGAGetLocalVecArray(iga,U,&localU,&arrayU);CHKERRQ(ierr);
      PetscInt i,N;
      ierr = VecGetSize(localU,&N);
      for(i=0;i<N;i++) iga->geometryX[i] += arrayU[i];
      ierr = IGARestoreLocalVecArray(iga,U,&localU,&arrayU);CHKERRQ(ierr);
    }

    // Dump solution vector
    char filename[256];
    sprintf(filename,"disp%d.dat",step);
    ierr = IGAWriteVec(iga,Utotal,filename);CHKERRQ(ierr);

  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
