! -*- f90 -*-

subroutine IGA_Interpolate(nen,dof,dim,der,N,U,V) &
  bind(C, name="IGA_Interpolate")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  use ISO_C_BINDING, only: C_FLOAT_COMPLEX, C_DOUBLE_COMPLEX
  implicit none
  integer(kind=C_INT),    intent(in),value :: nen,dof
  integer(kind=C_INT),    intent(in),value :: dim,der
  real   (kind=C_DOUBLE), intent(in)       :: N(dim**der,nen)
  real   (kind=C_DOUBLE), intent(in)       :: U(dof,nen)
  real   (kind=C_DOUBLE), intent(out)      :: V(dim**der,dof)
  integer a, i
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_Interpolate
