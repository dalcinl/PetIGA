#include "petscconf.h"
#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

module ElasticRodFJ

use PetIGA
implicit none

type, bind(C) :: IGAUser
   real(kind=IGA_REAL_KIND) rho ! density
   real(kind=IGA_REAL_KIND) E   ! Young modulus
end type IGAUser

contains

integer(kind=IGA_ERRCODE_KIND) &
function IFunction(p,shiftA,AA,shiftV,VV,t,UU,FF,user) result (ierr) &
bind(C, name="ElasticRod_IFunction")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(IGAUser),  intent(in) :: user
  real   (kind=IGA_REAL_KIND   ), intent(in), value :: shiftA, shiftV, t
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: AA(p%dof,p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: VV(p%dof,p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%dof,p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: FF(p%dof,p%nen)

  real   (kind=IGA_REAL_KIND   ), pointer :: N(:)
  real   (kind=IGA_REAL_KIND   ), pointer :: grad_N(:,:)
  scalar (kind=IGA_SCALAR_KIND )  :: A(p%dof), V(p%dof), U(p%dof)
  scalar (kind=IGA_SCALAR_KIND )  :: grad_U(p%dof,p%dim)
  integer(kind=IGA_INTEGER_KIND)  :: ia

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  A      = IGA_Value(p,AA)
  V      = IGA_Value(p,VV)
  U      = IGA_Value(p,UU)
  grad_U = IGA_Grad (p,UU)

  do ia = 1, p%nen
     FF(1,ia) = N(ia) * user%rho * A(1) &
              + user%E * dot_product(grad_N(:,ia),grad_U(1,:))
  end do

  ierr = 0
end function IFunction

integer(kind=IGA_ERRCODE_KIND) &
function IJacobian(p,shiftA,AA,shiftV,VV,t,UU,JJ,user) result (ierr) &
bind(C, name="ElasticRod_IJacobian")
  use PetIGA
  implicit none
  type(IGAPoint), intent(in) :: p
  type(IGAUser),  intent(in) :: user
  real   (kind=IGA_REAL_KIND   ), intent(in), value :: shiftA, shiftV, t
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: AA(p%dof,p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: VV(p%dof,p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)  :: UU(p%dof,p%nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out) :: JJ(p%dof,p%nen,p%dof,p%nen)

  real   (kind=IGA_REAL_KIND   ), pointer :: N(:)
  real   (kind=IGA_REAL_KIND   ), pointer :: grad_N(:,:)
  !scalar (kind=IGA_SCALAR_KIND )  :: A(p%dof), V(p%dof), U(p%dof)
  !scalar (kind=IGA_SCALAR_KIND )  :: grad_U(p%dof,p%dim)
  integer(kind=IGA_INTEGER_KIND)  :: ia, ib

  N      => IGA_Shape0(p)
  grad_N => IGA_Shape1(p)

  !A      = IGA_Eval(N,AA)
  !V      = IGA_Eval(N,VV)
  !U      = IGA_Eval(N,UU)
  !grad_U = IGA_Eval(grad_N,UU)

  do ia = 1, p%nen
     do ib = 1, p%nen
        JJ(1,ib,1,ia) = shiftA * user%rho * N(ia) * N(ib) &
                      + user%E * dot_product(grad_N(:,ia),grad_N(:,ib))
     end do
  end do

  ierr = 0
end function IJacobian

end module ElasticRodFJ
