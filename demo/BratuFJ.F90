#include "petscconf.h"
#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

module BratuFJ 

use PetIGA
implicit none

type, bind(C) :: IGAUser
   real(kind=IGA_REAL_KIND) lambda
end type IGAUser

contains

! --- Steady ---

integer(kind=IGA_ERRCODE_KIND) &
function Function(p,UU,FF,ctx) result (ierr) &
bind(C, name="Bratu_Function")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(IGAUser),  intent(in) :: ctx
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: FF(p%nen)
  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: u, grad_u(p%dim)
  integer(kind=IGA_INTEGER_KIND) :: a

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  u      = IGA_Eval(N,UU)
  grad_u = IGA_Eval(grad_N,UU)

  u      = IGA_Value(p,UU) ! just for testing
  grad_u = IGA_Grad(p,UU)  ! just for testing

  do a = 1, p%nen
     FF(a) = dot_product(grad_N(:,a),real(grad_u)) - &
             N(a) * ctx%lambda * exp(real(u))
  end do

  ierr = 0
end function Function

integer(kind=IGA_ERRCODE_KIND) &
function Jacobian(p,UU,JJ,ctx) result (ierr) &
bind(C, name="Bratu_Jacobian")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(IGAUser),  intent(in) :: ctx
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: JJ(p%nen,p%nen)
  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: u
  integer(kind=IGA_INTEGER_KIND) :: a, b

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  u = IGA_Eval(N,UU)

  u = IGA_Value(p,UU)

  do a = 1, p%nen
     do b = 1, p%nen
        JJ(b,a) = dot_product(grad_N(:,a),grad_N(:,b)) - &
                  N(a) * N(b) * ctx%lambda  * exp(real(u))
     end do
  end do

  ierr = 0
end function Jacobian

! --- Time-dependent ---

integer(kind=IGA_ERRCODE_KIND) &
function IFunction(p,dt,shift,VV,t,UU,FF,ctx) result (ierr) &
bind(C, name="Bratu_IFunction")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(IGAUser),  intent(in) :: ctx
  real   (kind=IGA_REAL_KIND   ), intent(in), value :: dt, shift, t
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: VV(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: FF(p%nen)

  real   (kind=IGA_REAL_KIND   ), pointer :: N(:), grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND ) :: v, u, grad_u(p%dim)
  integer(kind=IGA_INTEGER_KIND) :: a

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  v      = IGA_Eval(N,VV)
  u      = IGA_Eval(N,UU)
  grad_u = IGA_Eval(grad_N,UU)

  do a = 1, p%nen
     FF(a) = N(a) * real(v) + &
             dot_product(grad_N(:,a),real(grad_u)) - &
             N(a) * ctx%lambda * exp(real(u))
  end do

  ierr = 0
end function IFunction

integer(kind=IGA_ERRCODE_KIND) &
function IJacobian(p,dt,shift,VV,t,UU,JJ,ctx) result (ierr) &
bind(C, name="Bratu_IJacobian")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(IGAUser),  intent(in) :: ctx
  real   (kind=IGA_REAL_KIND   ), intent(in), value :: dt, shift, t
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: VV(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: JJ(p%nen,p%nen)

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
        JJ(b,a) = shift * N(a) * N(b) + &
                  dot_product(grad_N(:,a),grad_N(:,b)) - &
                  N(a) * N(b) * ctx%lambda  * exp(real(u))
     end do
  end do

  ierr = 0
end function IJacobian

end module BratuFJ
