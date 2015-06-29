#include "petscconf.h"
#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

module BratuFJ 

use PetIGA
implicit none

type, bind(C) :: AppCtx
   real(kind=IGA_REAL_KIND) lambda
end type AppCtx

integer, parameter :: rk = IGA_REAL_KIND

contains

! --- Steady ---

integer(kind=IGA_ERRCODE_KIND) &
function Function(p,UU,FF,ctx) result (ierr) &
bind(C, name="Bratu_Function")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(AppCtx),   intent(in) :: ctx
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: FF(p%neq)
  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:), hess_N(:,:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: u, grad_u(p%dim), del2_u
  integer(kind=IGA_INTEGER_KIND) :: a

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)
  hess_N => IGA_Shape2(p)

  u      = IGA_Eval(N,UU)
  grad_u = IGA_Eval(grad_N,UU)
  del2_u = IGA_Del2(hess_N,UU)

  u      = IGA_Value(p,UU) ! just for testing
  grad_u = IGA_Grad(p,UU)  ! just for testing
  del2_u = IGA_Del2(p,UU)  ! just for testing

  if (p%neq == 1) then
     ! collocation
     FF(1) = - del2_u - ctx%lambda * exp(real(u,rk))
  else
     ! galerkin
     do a = 1, p%nen
        FF(a) = + dot_product(grad_N(:,a),real(grad_u,rk)) &
                - N(a) * ctx%lambda * exp(real(u,rk))
     end do
  end if

  ierr = 0
end function Function

integer(kind=IGA_ERRCODE_KIND) &
function Jacobian(p,UU,JJ,ctx) result (ierr) &
bind(C, name="Bratu_Jacobian")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(AppCtx),   intent(in) :: ctx
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: JJ(p%nen,p%neq)
  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:), hess_N(:,:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: u, grad_u(p%dim), del2_u, del2_N
  integer(kind=IGA_INTEGER_KIND) :: a, b, i

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)
  hess_N => IGA_Shape2(p)

  u      = IGA_Eval(N,UU)
  grad_u = IGA_Eval(grad_N,UU)
  del2_u = IGA_Del2(hess_N,UU)

  u      = IGA_Value(p,UU) ! just for testing
  grad_u = IGA_Grad(p,UU)  ! just for testing
  del2_u = IGA_Del2(p,UU)  ! just for testing

  if (p%neq == 1) then
     ! collocation
     do a = 1, p%nen
        del2_N = 0
        do i = 1, p%dim
           del2_N = del2_N + hess_N(i,i,a)
        end do
        JJ(a,1) = - del2_N - N(a) * ctx%lambda * exp(real(u,rk))
     end do
  else
     ! galerkin
     do a = 1, p%nen
        do b = 1, p%nen
           JJ(b,a) = + dot_product(grad_N(:,a),grad_N(:,b)) &
                     - N(a) * N(b) * ctx%lambda * exp(real(u,rk))
        end do
     end do
  end if

  ierr = 0
end function Jacobian

! --- Time-dependent ---

integer(kind=IGA_ERRCODE_KIND) &
function IFunction(p,shift,VV,t,UU,FF,ctx) result (ierr) &
bind(C, name="Bratu_IFunction")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(AppCtx),   intent(in) :: ctx
  real   (kind=IGA_REAL_KIND   ), intent(in), value :: shift, t
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: VV(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: FF(p%neq)

  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: v, u, grad_u(p%dim)
  integer(kind=IGA_INTEGER_KIND) :: a

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  v      = IGA_Eval(N,VV)
  u      = IGA_Eval(N,UU)
  grad_u = IGA_Eval(grad_N,UU)

  do a = 1, p%nen
     FF(a) = + N(a) * real(v,rk) &
             + dot_product(grad_N(:,a),real(grad_u,rk)) &
             - N(a) * ctx%lambda * exp(real(u,rk))
  end do

  ierr = 0
end function IFunction

integer(kind=IGA_ERRCODE_KIND) &
function IJacobian(p,shift,VV,t,UU,JJ,ctx) result (ierr) &
bind(C, name="Bratu_IJacobian")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(AppCtx),   intent(in) :: ctx
  real   (kind=IGA_REAL_KIND   ), intent(in), value :: shift, t
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: VV(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: JJ(p%nen,p%neq)

  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: v, u, grad_u(p%dim)
  integer(kind=IGA_INTEGER_KIND) :: a, b

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  v      = IGA_Eval(N,VV)
  u      = IGA_Eval(N,UU)
  grad_u = IGA_Eval(grad_N,UU)

  do a = 1, p%nen
     do b = 1, p%nen
        JJ(b,a) = + shift * N(a) * N(b) &
                  + dot_product(grad_N(:,a),grad_N(:,b)) &
                  - N(a) * N(b) * ctx%lambda  * exp(real(u,rk))
     end do
  end do

  ierr = 0
end function IJacobian

end module BratuFJ
