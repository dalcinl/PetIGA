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
     rational,geometry,  &
     inq,ina,ind,iJ,iN,  &
     Cw,                 &
     detJ,J,N0,N1,N2,N3) &
  bind(C, name="IGA_ShapeFuns_1D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 1
  integer(kind=C_INT   ), intent(in),value :: rational
  integer(kind=C_INT   ), intent(in),value :: geometry
  integer(kind=C_INT   ), intent(in),value :: inq, ina, ind
  real   (kind=C_DOUBLE), intent(in)  :: iJ, iN(0:ind,ina,inq)
  real   (kind=C_DOUBLE), intent(in)  :: Cw(dim+1,ina)
  real   (kind=C_DOUBLE), intent(out) :: detJ(     inq)
  real   (kind=C_DOUBLE), intent(out) :: J(dim,dim,inq)
  real   (kind=C_DOUBLE), intent(out) :: N0(       ina,inq)
  real   (kind=C_DOUBLE), intent(out) :: N1(   dim,ina,inq)
  real   (kind=C_DOUBLE), intent(out) :: N2(dim**2,ina,inq)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim**3,ina,inq)
  integer :: ia,iq
  integer :: na,nd

  real(kind=C_DOUBLE) :: C(dim,ina)
  real(kind=C_DOUBLE) :: w(    ina)
  if (rational /= 0) then
     w = Cw(dim+1,:) 
  end if
  if (geometry /= 0) then
     C = Cw(1:dim,:)
  end if

  nd = max(1,min(ind,3))
  na = ina
  do iq=1,inq
     call TensorBasisFuns(&
          ina,ind,iN(:,:,iq),&
          nd,&
          N0(  :,iq),&
          N1(:,:,iq),&
          N2(:,:,iq),&
          N3(:,:,iq))
     if (rational /= 0) then
        call Rationalize(&
             nd,na,w,&
             N0(  :,iq),&
             N1(:,:,iq),&
             N2(:,:,iq),&
             N3(:,:,iq))
     endif
     if (geometry /= 0) then
        call GeometryMap(&
             nd,na,C,&
             detJ( iq),&
             J(:,:,iq),&
             N0(  :,iq),&
             N1(:,:,iq),&
             N2(:,:,iq),&
             N3(:,:,iq))
        detJ( iq) = detJ( iq) * iJ 
        J(1,:,iq) = J(1,:,iq) * iJ
     else
        detJ( iq) = iJ
        J(1,1,iq) = iJ
     end if
  end do

contains

subroutine TensorBasisFuns(&
     ina,ind,iN,&
     nd,N0,N1,N2,N3)
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 1
  integer(kind=C_INT   ), intent(in),value :: ina, ind
  real   (kind=C_DOUBLE), intent(in)  :: iN(0:ind,ina)
  integer(kind=C_INT   ), intent(in)  :: nd
  real   (kind=C_DOUBLE), intent(out) :: N0(            ina)
  real   (kind=C_DOUBLE), intent(out) :: N1(        dim,ina)
  real   (kind=C_DOUBLE), intent(out) :: N2(    dim,dim,ina)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim,dim,dim,ina)
  integer :: ia
  !
  forall (ia=1:ina)
     N0(ia) = iN(0,ia)
  end forall
  !
  forall (ia=1:ina)
     N1(1,ia) = iN(1,ia)
  end forall
  !
  if (nd < 2) return
  forall (ia=1:ina)
     N2(1,1,ia) = iN(2,ia)
  end forall
  !
  if (nd < 3) return
  forall (ia=1:ina)
     N3(1,1,1,ia) = iN(3,ia)
  end forall
  !
end subroutine TensorBasisFuns

include 'petigarat.f90.in'
include 'petigamap.f90.in'

end subroutine IGA_ShapeFuns_1D
