subroutine IGA_Quadrature_1D(&
     inq,iX,iW,iL,           &
     W,J,X,L)                &
  bind(C, name="IGA_Quadrature_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 1
  integer(kind=IGA_INT ), intent(in),value :: inq
  real   (kind=IGA_REAL), intent(in)  :: iX(inq), iW(inq), iL
  real   (kind=IGA_REAL), intent(out) :: W(inq)
  real   (kind=IGA_REAL), intent(out) :: J(inq)
  real   (kind=IGA_REAL), intent(out) :: X(dim,inq)
  real   (kind=IGA_REAL), intent(out) :: L(dim,inq)
  integer(kind=IGA_INT )  :: iq
  forall (iq=1:inq)
     !
     W(iq) = iW(iq)
     J(iq) = iL
     !
     X(1,iq) = iX(iq)
     L(1,iq) = iL
  end forall
end subroutine IGA_Quadrature_1D


subroutine IGA_BasisFuns_1D(&
     order,                 &
     rational,W,            &
     inq,ina,ind,iN,        &
     N0,N1,N2,N3)           &
  bind(C, name="IGA_BasisFuns_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 1
  integer(kind=IGA_INT ), intent(in),value :: order
  integer(kind=IGA_INT ), intent(in),value :: rational
  integer(kind=IGA_INT ), intent(in),value :: inq, ina, ind
  real   (kind=IGA_REAL), intent(in)  :: iN(0:ind,ina,inq)
  real   (kind=IGA_REAL), intent(in)  :: W(dim+1,  ina)
  real   (kind=IGA_REAL), intent(out) :: N0(       ina,inq)
  real   (kind=IGA_REAL), intent(out) :: N1(   dim,ina,inq)
  real   (kind=IGA_REAL), intent(out) :: N2(dim**2,ina,inq)
  real   (kind=IGA_REAL), intent(out) :: N3(dim**3,ina,inq)
  integer(kind=IGA_INT ) :: ia,iq
  integer(kind=IGA_INT ) :: nen
  nen = ina
  do iq=1,inq
     call TensorBasisFuns(&
          order,&
          ina,ind,iN(:,:,iq),&
          N0(  :,iq),&
          N1(:,:,iq),&
          N2(:,:,iq),&
          N3(:,:,iq))
     if (rational /= 0) then
        call Rationalize(&
             order,&
             nen,W,&
             N0(  :,iq),&
             N1(:,:,iq),&
             N2(:,:,iq),&
             N3(:,:,iq))
     end if
  end do

contains

subroutine TensorBasisFuns(&
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

end subroutine IGA_BasisFuns_1D


subroutine IGA_ShapeFuns_1D(&
     order,                 &
     nqp,nen,X,             &
     M0,M1,M2,M3,           &
     N0,N1,N2,N3,           &
     DetJac,F,G)            &
  bind(C, name="IGA_ShapeFuns_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 1
  integer(kind=IGA_INT ), intent(in),value :: order
  integer(kind=IGA_INT ), intent(in),value :: nqp
  integer(kind=IGA_INT ), intent(in),value :: nen
  real   (kind=IGA_REAL), intent(in)    :: X(dim+1,nen)
  real   (kind=IGA_REAL), intent(in)    :: M0(       nen,nqp)
  real   (kind=IGA_REAL), intent(in)    :: M1(dim,   nen,nqp)
  real   (kind=IGA_REAL), intent(in)    :: M2(dim**2,nen,nqp)
  real   (kind=IGA_REAL), intent(in)    :: M3(dim**3,nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N0(       nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N1(dim,   nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N2(dim**2,nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N3(dim**3,nen,nqp)
  real   (kind=IGA_REAL), intent(inout) :: DetJac(nqp)
  real   (kind=IGA_REAL), intent(out)   :: F(dim,dim,nqp)
  real   (kind=IGA_REAL), intent(out)   :: G(dim,dim,nqp)
  integer(kind=IGA_INT )  :: q
  real   (kind=IGA_REAL)  :: DetF
  do q=1,nqp
     call GeometryMap(&
          order,&
          nen,X,&
          M0(:,q),M1(:,:,q),M2(:,:,q),M3(:,:,q),&
          N0(:,q),N1(:,:,q),N2(:,:,q),N3(:,:,q),&
          DetF,F(:,:,q),G(:,:,q))
     DetJac(q) = DetJac(q) * DetF
  end do
contains
include 'petigageo.f90.in'
end subroutine IGA_ShapeFuns_1D
