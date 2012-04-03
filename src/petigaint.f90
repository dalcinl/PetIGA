! -*- f90 -*-

subroutine IGA_Interpolate(nen,dof,dim,der,N,U,V) &
  bind(C, name="IGA_Interpolate")
  use PetIGA
  implicit none
  integer(kind=IGA_INT   ), intent(in),value :: nen,dof
  integer(kind=IGA_INT   ), intent(in),value :: dim,der
  real   (kind=IGA_REAL  ), intent(in)       :: N(dim**der,nen)
  real   (kind=IGA_SCALAR), intent(in)       :: U(dof,nen)
  real   (kind=IGA_SCALAR), intent(out)      :: V(dim**der,dof)
  integer(kind=IGA_INT   ) :: a, i
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_Interpolate
