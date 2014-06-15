pure subroutine IGA_Quadrature_1D(&
     inq,iX,iW,iJ,                &
     X,W,J)                       &
  bind(C, name="IGA_Quadrature_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iX(inq), iW(inq), iJ
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(dim,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: W(    inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: J(    inq)
  integer(kind=IGA_INTEGER_KIND)  :: iq
  do iq=1,inq
     X(1,iq) = iX(iq)
     W(  iq) = iW(iq)
  end do
  J = iJ
end subroutine IGA_Quadrature_1D


pure subroutine IGA_BasisFuns_1D(&
     order,                      &
     rational,W,                 &
     inq,ina,ind,iN,             &
     N0,N1,N2,N3)                &
  bind(C, name="IGA_BasisFuns_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: rational
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq, ina, ind
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: W(ina)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:ind,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(       ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(   dim,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(dim**2,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(dim**3,ina,inq)
  integer(kind=IGA_INTEGER_KIND)  :: ia, iq
  integer(kind=IGA_INTEGER_KIND)  :: nen
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

pure subroutine TensorBasisFuns(&
     ord,&
     ina,ind,iN,&
     N0,N1,N2,N3)
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: ord
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: ina, ind
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:ind,ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(            ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(        dim,ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(    dim,dim,ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(dim,dim,dim,ina)
  integer(kind=IGA_INTEGER_KIND)  :: ia
  !
  do ia=1,ina
     N0(ia) = iN(0,ia)
  end do
  !
  do ia=1,ina
     N1(1,ia) = iN(1,ia)
  end do
  !
  if (ord < 2) return
  do ia=1,ina
     N2(1,1,ia) = iN(2,ia)
  end do
  !
  if (ord < 3) return
  do ia=1,ina
     N3(1,1,1,ia) = iN(3,ia)
  end do
  !
end subroutine TensorBasisFuns

include 'petigarat.f90.in'

end subroutine IGA_BasisFuns_1D


pure subroutine IGA_ShapeFuns_1D(&
     order,                      &
     nqp,nen,X,                  &
     M0,M1,M2,M3,                &
     N0,N1,N2,N3,                &
     dX,G0,G1,H0,H1,I0,I1)      &
  bind(C, name="IGA_ShapeFuns_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nqp
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: X(dim,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M0(       nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M1(dim,   nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M2(dim**2,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M3(dim**3,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N0(       nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N1(dim,   nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N2(dim**2,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N3(dim**3,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: dX(nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: G0(dim,dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: G1(dim,dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: H0(dim,dim,dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: H1(dim,dim,dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: I0(dim,dim,dim,dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: I1(dim,dim,dim,dim,nqp)
  integer(kind=IGA_INTEGER_KIND)  :: q
  do q=1,nqp
     call GeometryMap(&
          order,&
          nen,X,&
          M0(:,q),M1(:,:,q),M2(:,:,q),M3(:,:,q),&
          N0(:,q),N1(:,:,q),N2(:,:,q),N3(:,:,q),&
          dX(q),&
          G0(:,:,q),G1(:,:,q),&
          H0(:,:,:,q),H1(:,:,:,q),&
          I0(:,:,:,:,q),I1(:,:,:,:,q))
  end do
contains
include 'petigageo.f90.in'
end subroutine IGA_ShapeFuns_1D
