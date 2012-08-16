#include "petscconf.h"
#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

!module Bratu 
!contains

integer(kind=IGA_ERRCODE) &
function Function(p,UU,FF,ctx) result (ierr) &
bind(C, name="Bratu_Function")
  use PetIGA
  implicit none
  type(C_PTR), intent(in), value :: ctx
  type(IGAPoint), intent(in)     :: p
  scalar (kind=IGA_SCALAR), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR), intent(out) :: FF(p%nen)
  integer(kind=IGA_INT)    :: a
  scalar (kind=IGA_SCALAR) :: u, grad_u(p%dim)
  real   (kind=IGA_REAL)   :: N(p%nen), grad_N(p%dim,p%nen)
  ierr = IGAPointFormValue(p,UU,u)
  ierr = IGAPointFormGrad (p,UU,grad_u)
  ierr = IGAPointFormShapeFuns(p,0,N)
  ierr = IGAPointFormShapeFuns(p,1,grad_N)
  do a = 1, p%nen
     FF(a) = dot_product(grad_N(:,a),real(grad_u)) - N(a) * 1 * exp(real(u))
  end do
  ierr = 0
end function Function

integer(kind=IGA_ERRCODE) &
function Jacobian(p,UU,JJ,ctx) result (ierr) &
bind(C, name="Bratu_Jacobian")
  use PetIGA
  implicit none
  type(C_PTR), intent(in), value :: ctx
  type(IGAPoint), intent(in)     :: p
  scalar (kind=IGA_SCALAR), intent(in)  :: UU(p%nen)
  scalar (kind=IGA_SCALAR), intent(out) :: JJ(p%nen,p%nen)
  integer(kind=IGA_INT )    :: a, b
  scalar (kind=IGA_SCALAR)  :: u
  real   (kind=IGA_REAL)    :: N(p%nen), grad_N(p%dim,p%nen)
  ierr = IGAPointFormValue(p,UU,u)
  ierr = IGAPointFormShapeFuns(p,0,N)
  ierr = IGAPointFormShapeFuns(p,1,grad_N)
  do a = 1, p%nen
     do b = 1, p%nen
        JJ(b,a) = dot_product(grad_N(:,a),grad_N(:,b)) - N(a) * N(b) * 1 * exp(real(u))
     end do
  end do
  ierr = 0
end function Jacobian

!end module Bratu 
