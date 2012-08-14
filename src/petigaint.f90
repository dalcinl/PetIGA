! -*- f90 -*-

subroutine IGA_GetPoint(nen,dim,N,C,X) &
  bind(C, name="IGA_GetPoint")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), intent(in),value :: nen,dim
  real   (kind=IGA_REAL), intent(in)       :: N(nen)
  real   (kind=IGA_REAL), intent(in)       :: C(dim,nen)
  real   (kind=IGA_REAL), intent(out)      :: X(dim)
  integer(kind=IGA_INT )  :: a
  ! X = matmul(C,N)
  X = 0
  do a = 1, nen
     X = X + N(a) * C(:,a)
  end do
end subroutine IGA_GetPoint

subroutine IGA_GetValue(nen,dof,N,U,V) &
  bind(C, name="IGA_GetValue")
  use PetIGA
  implicit none
  integer(kind=IGA_INT   ), intent(in),value :: nen,dof
  real   (kind=IGA_REAL  ), intent(in)       :: N(nen)
  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
  real   (kind=IGA_SCALAR), intent(out)      :: V(dof)
  integer(kind=IGA_INT   )  :: a, i
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
  integer(kind=IGA_INT   ), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL  ), intent(in)       :: N(dim,nen)
  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
  real   (kind=IGA_SCALAR), intent(out)      :: V(dim,dof)
  integer(kind=IGA_INT   )  :: a, c
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
  integer(kind=IGA_INT   ), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL  ), intent(in)       :: N(dim*dim,nen)
  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
  real   (kind=IGA_SCALAR), intent(out)      :: V(dim*dim,dof)
  integer(kind=IGA_INT   )  :: a, i
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
  integer(kind=IGA_INT   ), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL  ), intent(in)       :: N(dim,dim,nen)
  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
  real   (kind=IGA_SCALAR), intent(out)      :: V(dof)
  integer(kind=IGA_INT   )  :: a, c, i
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
  integer(kind=IGA_INT   ), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL  ), intent(in)       :: N(dim*dim*dim,nen)
  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
  real   (kind=IGA_SCALAR), intent(out)      :: V(dim*dim*dim,dof)
  integer(kind=IGA_INT   )  :: a, i
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
!  integer(kind=IGA_INT   ), intent(in),value :: nen,dof
!  integer(kind=IGA_INT   ), intent(in),value :: dim,der
!  real   (kind=IGA_REAL  ), intent(in)       :: N(dim**der,nen)
!  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
!  real   (kind=IGA_SCALAR), intent(out)      :: V(dim**der,dof)
!  integer(kind=IGA_INT   )  :: a, i
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
  integer(kind=IGA_INT   ), intent(in),value :: nen,dof
  integer(kind=IGA_INT   ), intent(in),value :: dim,der
  real   (kind=IGA_REAL  ), intent(in)       :: N(dim**der,nen)
  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
  real   (kind=IGA_SCALAR), intent(out)      :: V(dim**der,dof)
  integer(kind=IGA_INT   )  :: a, i
  ! V = matmul(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_Interpolate
