subroutine IGA_Quadrature_1D(&
     inq,iX,iW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_1D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer, parameter :: dim = 1
  integer(kind=C_INT   ), intent(in),value :: inq
  real   (kind=C_DOUBLE), intent(in)  :: iX(inq), iW(inq)
  real   (kind=C_DOUBLE), intent(out) :: X(dim,inq)
  real   (kind=C_DOUBLE), intent(out) :: W(inq)
  integer :: iq
  forall (iq=1:inq)
     X(:,iq) = (/ iX(iq) /)
     W(iq)   = iW(iq)
  end forall
end subroutine IGA_Quadrature_1D

subroutine IGA_ShapeFuns_1D(&
     inq,ina,ind,iX,iJ,iN, &
     detJ,X,J,N0,N1,N2,N3) &
  bind(C, name="IGA_ShapeFuns_1D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer, parameter :: dim = 1
  integer(kind=C_INT   ), intent(in),value :: inq, ina, ind
  real   (kind=C_DOUBLE), intent(in)  :: iX(inq), iJ, iN(0:ind,ina,inq)
  real   (kind=C_DOUBLE), intent(out) :: detJ(     inq)
  real   (kind=C_DOUBLE), intent(out) :: X(    dim,inq)
  real   (kind=C_DOUBLE), intent(out) :: J(dim,dim,inq)
  real   (kind=C_DOUBLE), intent(out) :: N0(            ina,inq)
  real   (kind=C_DOUBLE), intent(out) :: N1(        dim,ina,inq)
  real   (kind=C_DOUBLE), intent(out) :: N2(    dim,dim,ina,inq)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim,dim,dim,ina,inq)
  integer :: ia,iq

  forall (iq=1:inq)
     forall (ia=1:ina)
        N0(ia,iq)   = iN(0,ia,iq)
        N1(1,ia,iq) = iN(1,ia,iq)
     end forall
     X(:,  iq)  = (/ iX(iq) /)
     J(1,1,iq) = iJ
     detJ( iq) = iJ
  end forall

  forall (iq=1:inq)
     forall (ia=1:ina)
        N2(1,1,ia,iq) = iN(2,ia,iq)
     end forall
  end forall

  !forall (iq=1:inq)
  !   forall (ia=1:ina)
  !      N3(1,1,1,ia,iq) = iN(3,ia,iq)
  !   end forall
  !end forall

end subroutine IGA_ShapeFuns_1D
