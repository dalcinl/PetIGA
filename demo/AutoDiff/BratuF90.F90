#include "petscconf.h"
#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

module BratuF90

use PetIGA
implicit none

type, bind(C) :: AppCtx
   real(kind=IGA_REAL_KIND) lambda
end type AppCtx

integer, parameter :: rk = IGA_REAL_KIND

contains

integer(kind=IGA_ERRCODE_KIND) &
function Function(p,UU,FF,ctx) result (ierr) &
bind(C, name="FunctionF90")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(AppCtx),   intent(in) :: ctx
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: FF(p%neq)
  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: u, grad_u(p%dim)
  real   (kind=IGA_SCALAR_KIND ) :: lambda_exp_u
  integer(kind=IGA_INTEGER_KIND) :: a

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  u      = IGA_Value(p,UU) ! just for testing
  grad_u = IGA_Grad(p,UU)  ! just for testing

  lambda_exp_u = ctx%lambda * exp(real(u,rk))

  do a = 1, p%nen
     FF(a) = + dot_product(grad_N(:,a),real(grad_u,rk)) &
             - N(a) * lambda_exp_u
  end do

  ierr = 0
end function Function

integer(kind=IGA_ERRCODE_KIND) &
function Jacobian(p,UU,JJ,ctx) result (ierr) &
bind(C, name="JacobianF90")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(AppCtx),   intent(in) :: ctx
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: JJ(p%nen,p%neq)
  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: u, grad_u(p%dim)
  real   (kind=IGA_SCALAR_KIND ) :: lambda_exp_u
  integer(kind=IGA_INTEGER_KIND) :: a, b

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  u      = IGA_Value(p,UU)
  grad_u = IGA_Grad (p,UU)

  lambda_exp_u = ctx%lambda * exp(real(u,rk))

     ! galerkin
  do a = 1, p%nen
     do b = 1, p%nen
        JJ(b,a) = + dot_product(grad_N(:,a),grad_N(:,b)) &
                  - N(a) * N(b) * lambda_exp_u
     end do
  end do

  ierr = 0
end function Jacobian

end module BratuF90
