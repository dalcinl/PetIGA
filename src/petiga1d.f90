subroutine IGA_Quadrature_1D(&
     inq,iX,iW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 1
  integer(kind=IGA_INT ), intent(in),value :: inq
  real   (kind=IGA_REAL), intent(in)  :: iX(inq), iW(inq)
  real   (kind=IGA_REAL), intent(out) :: X(dim,inq)
  real   (kind=IGA_REAL), intent(out) :: W(inq)
  integer(kind=IGA_INT )  :: iq
  forall (iq=1:inq)
     X(:,iq) = (/ iX(iq) /)
     W(iq)   = iW(iq)
  end forall
end subroutine IGA_Quadrature_1D

subroutine IGA_ShapeFuns_1D(&
     geometry,rational,     &
     inq,ina,ind,iJ,iN,     &
     Cw,detJac,Jac,         &
     N0,N1,N2,N3)           &
  bind(C, name="IGA_ShapeFuns_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 1
  integer(kind=IGA_INT ), intent(in),value :: geometry
  integer(kind=IGA_INT ), intent(in),value :: rational
  integer(kind=IGA_INT ), intent(in),value :: inq, ina, ind
  real   (kind=IGA_REAL), intent(in)  :: iJ, iN(0:ind,ina,inq)
  real   (kind=IGA_REAL), intent(in)  :: Cw(dim+1,ina)
  real   (kind=IGA_REAL), intent(out) :: detJac(     inq)
  real   (kind=IGA_REAL), intent(out) :: Jac(dim,dim,inq)
  real   (kind=IGA_REAL), intent(out) :: N0(       ina,inq)
  real   (kind=IGA_REAL), intent(out) :: N1(   dim,ina,inq)
  real   (kind=IGA_REAL), intent(out) :: N2(dim**2,ina,inq)
  real   (kind=IGA_REAL), intent(out) :: N3(dim**3,ina,inq)

  integer(kind=IGA_INT ) :: ia,iq
  integer(kind=IGA_INT ) :: i,nen,ord
  real   (kind=IGA_REAL) :: C(dim,ina)
  real   (kind=IGA_REAL) :: W(    ina)

  if (geometry /= 0) then
     W = Cw(dim+1,:)
     forall (i=1:dim)
        C(i,:) = Cw(i,:) / W
     end forall
  end if

  nen = ina
  ord = max(1,min(ind,3))
  do iq=1,inq
     call TensorBasisFuns(&
          ord,&
          ina,ind,iN(:,:,iq),&
          N0(  :,iq),&
          N1(:,:,iq),&
          N2(:,:,iq),&
          N3(:,:,iq))
     if (rational /= 0) then
        call Rationalize(&
             ord,&
             nen,W,&
             N0(  :,iq),&
             N1(:,:,iq),&
             N2(:,:,iq),&
             N3(:,:,iq))
     endif
     if (geometry /= 0) then
        call GeometryMap(&
             ord,&
             nen,C,&
             detJac( iq),&
             Jac(:,:,iq),&
             N0(  :,iq),&
             N1(:,:,iq),&
             N2(:,:,iq),&
             N3(:,:,iq))
        detJac( iq) = detJac( iq) * iJ
        Jac(1,:,iq) = Jac(1,:,iq) * iJ
     else
        detJac( iq) = iJ
        Jac(1,1,iq) = iJ
     end if
  end do

contains

pure subroutine TensorBasisFuns(&
     ord,&
     ina,ind,iN,&
     N0,N1,N2,N3)
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 1
  integer(kind=IGA_INT ), intent(in),value :: ord
  integer(kind=IGA_INT ), intent(in),value :: ina, ind
  real   (kind=IGA_REAL), intent(in)  :: iN(0:ind,ina)
  real   (kind=IGA_REAL), intent(out) :: N0(            ina)
  real   (kind=IGA_REAL), intent(out) :: N1(        dim,ina)
  real   (kind=IGA_REAL), intent(out) :: N2(    dim,dim,ina)
  real   (kind=IGA_REAL), intent(out) :: N3(dim,dim,dim,ina)
  integer(kind=IGA_INT ) :: ia
  !
  forall (ia=1:ina)
     N0(ia) = iN(0,ia)
  end forall
  !
  forall (ia=1:ina)
     N1(1,ia) = iN(1,ia)
  end forall
  !
  if (ord < 2) return
  forall (ia=1:ina)
     N2(1,1,ia) = iN(2,ia)
  end forall
  !
  if (ord < 3) return
  forall (ia=1:ina)
     N3(1,1,1,ia) = iN(3,ia)
  end forall
  !
end subroutine TensorBasisFuns

include 'petigarat.f90.in'
include 'petigageo.f90.in'

end subroutine IGA_ShapeFuns_1D
