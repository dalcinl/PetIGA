! -*- f90 -*-

#include "petscconf.h"
#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

subroutine IGA_GetGeomMap(nen,nsd,N,C,X) &
  bind(C, name="IGA_GetGeomMap")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,nsd
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: C(nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: X(nsd)
  X = matmul(C,N)
end subroutine IGA_GetGeomMap

subroutine IGA_GetGradGeomMap(nen,nsd,dim,N,C,F) &
  bind(C, name="IGA_GetGradGeomMap")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,nsd,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: C(nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: F(dim,nsd)
  F = matmul(N,transpose(C))
end subroutine IGA_GetGradGeomMap

subroutine IGA_GetInvGradGeomMap(nen,nsd,dim,N,C,G) &
  bind(C, name="IGA_GetInvGradGeomMap")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,nsd,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: C(nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: G(nsd,dim)
  real   (kind=IGA_REAL_KIND   )  :: F(nsd,dim)
  real   (kind=IGA_REAL_KIND   )  :: M(dim,dim), detM, invM(dim,dim)
  F = matmul(C,transpose(N))
  M = matmul(transpose(F),F)
  detM = Determinant(dim,M)
  invM = Inverse(dim,detM,M)
  G = transpose(matmul(invM,transpose(F)))
contains
include 'petigainv.f90.in'
end subroutine IGA_GetInvGradGeomMap


subroutine IGA_GetValue(nen,dof,N,U,V) &
  bind(C, name="IGA_GetValue")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = matmul(N,transpose(U))
  V = 0
  do a = 1, nen
     V = V + N(a) * U(:,a)
  end do
end subroutine IGA_GetValue

subroutine IGA_GetGrad(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_GetGrad")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, c
  ! V = matmul(N,transpose(U))
  V = 0
  do a = 1, nen
     do c = 1, dof
        V(:,c) = V(:,c) + N(:,a) * U(c,a)
     end do
  end do
end subroutine IGA_GetGrad

subroutine IGA_GetHess(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_GetHess")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim*dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim*dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = matmul(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_GetHess

subroutine IGA_GetDel2(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_GetDel2")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, c, i
  V = 0
  do a = 1, nen
     do c = 1, dof
        do i = 1, dim
           V(c) = V(c) + N(i,i,a) * U(c,a)
        end do
     end do
  end do
end subroutine IGA_GetDel2

subroutine IGA_GetDer3(nen,dof,dim,N,U,V) &
     bind(C, name="IGA_GetDer3")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim*dim*dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim*dim*dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = matmul(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_GetDer3

!subroutine IGA_GetDerivative(nen,dof,dim,der,N,U,V) &
!  bind(C, name="IGA_GetDerivative")
!  use PetIGA
!  implicit none
!  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof
!  integer(kind=IGA_INTEGER_KIND), intent(in),value :: dim,der
!  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim**der,nen)
!  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
!  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim**der,dof)
!  integer(kind=IGA_INTEGER_KIND)  :: a, i
!  ! V = matmul(N,transpose(U))
!  V = 0
!  do a = 1, nen
!     do i = 1, dof
!        V(:,i) = V(:,i) + N(:,a) * U(i,a)
!     end do
!  end do
!end subroutine IGA_GetDerivative

subroutine IGA_Interpolate(nen,dof,dim,der,N,U,V) &
  bind(C, name="IGA_Interpolate")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: dim,der
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim**der,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim**der,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = matmul(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_Interpolate
