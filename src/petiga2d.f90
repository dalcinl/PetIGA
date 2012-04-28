subroutine IGA_Quadrature_2D(&
     inq,iX,iW,iJ,           &
     jnq,jX,jW,jJ,           &
     X,W,detJ,J)             &
  bind(C, name="IGA_Quadrature_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
  integer(kind=IGA_INT ), intent(in),value :: inq
  integer(kind=IGA_INT ), intent(in),value :: jnq
  real   (kind=IGA_REAL), intent(in)  :: iX(inq), iW(inq), iJ
  real   (kind=IGA_REAL), intent(in)  :: jX(jnq), jW(jnq), jJ
  real   (kind=IGA_REAL), intent(out) :: X(dim,    inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: W(        inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: detJ(     inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: J(dim,dim,inq,jnq)
  integer(kind=IGA_INT ) :: iq
  integer(kind=IGA_INT ) :: jq
  forall (iq=1:inq, jq=1:jnq)
     X(1,iq,jq) = iX(iq)
     X(2,iq,jq) = jX(jq)
     W(  iq,jq) =    iW(iq) * jW(jq)
     detJ( iq,jq) = iJ * jJ
     J(:,:,iq,jq) = 0
     J(1,1,iq,jq) = iJ
     J(2,2,iq,jq) = jJ
  end forall
end subroutine IGA_Quadrature_2D


subroutine IGA_BasisFuns_2D(&
     order,                 &
     rational,W,            &
     inq,ina,ind,iN,        &
     jnq,jna,jnd,jN,        &
     N0,N1,N2,N3)           &
  bind(C, name="IGA_BasisFuns_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
  integer(kind=IGA_INT ), intent(in),value :: order
  integer(kind=IGA_INT ), intent(in),value :: rational
  integer(kind=IGA_INT ), intent(in),value :: inq, ina, ind
  integer(kind=IGA_INT ), intent(in),value :: jnq, jna, jnd
  real   (kind=IGA_REAL), intent(in)  :: iN(0:ind,ina,inq)
  real   (kind=IGA_REAL), intent(in)  :: jN(0:jnd,jna,jnq)
  real   (kind=IGA_REAL), intent(in)  :: W(dim+1,  ina*jna)
  real   (kind=IGA_REAL), intent(out) :: N0(       ina*jna,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: N1(   dim,ina*jna,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: N2(dim**2,ina*jna,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: N3(dim**3,ina*jna,inq,jnq)
  integer(kind=IGA_INT ) :: ia,iq
  integer(kind=IGA_INT ) :: ja,jq
  integer(kind=IGA_INT ) :: ka,kq
  integer(kind=IGA_INT ) :: nen
  nen = ina*jna
  do jq=1,jnq
     do iq=1,inq
        call TensorBasisFuns(&
             order,&
             ina,ind,iN(:,:,iq),&
             jna,jnd,jN(:,:,jq),&
             N0(  :,iq,jq),&
             N1(:,:,iq,jq),&
             N2(:,:,iq,jq),&
             N3(:,:,iq,jq))
        if (rational /= 0) then
           call Rationalize(&
                order,&
                nen,W,&
                N0(  :,iq,jq),&
                N1(:,:,iq,jq),&
                N2(:,:,iq,jq),&
                N3(:,:,iq,jq))
        end if
     end do
  end do

contains

subroutine TensorBasisFuns(&
     ord,&
     ina,ind,iN,&
     jna,jnd,jN,&
     N0,N1,N2,N3)
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
  integer(kind=IGA_INT ), intent(in),value :: ord
  integer(kind=IGA_INT ), intent(in),value :: ina, ind
  integer(kind=IGA_INT ), intent(in),value :: jna, jnd
  real   (kind=IGA_REAL), intent(in)  :: iN(0:ind,ina)
  real   (kind=IGA_REAL), intent(in)  :: jN(0:jnd,jna)
  real   (kind=IGA_REAL), intent(out) :: N0(            ina,jna)
  real   (kind=IGA_REAL), intent(out) :: N1(        dim,ina,jna)
  real   (kind=IGA_REAL), intent(out) :: N2(    dim,dim,ina,jna)
  real   (kind=IGA_REAL), intent(out) :: N3(dim,dim,dim,ina,jna)
  integer(kind=IGA_INT ) :: ia, ja
  !
  forall (ia=1:ina, ja=1:jna)
     N0(ia,ja) = iN(0,ia) * jN(0,ja)
  end forall
  !
  forall (ia=1:ina, ja=1:jna)
     N1(1,ia,ja) = iN(1,ia) * jN(0,ja)
     N1(2,ia,ja) = iN(0,ia) * jN(1,ja)
  end forall
  !
  if (ord < 2) return ! XXX Optimize!
  forall (ia=1:ina, ja=1:jna)
     N2(1,1,ia,ja) = iN(2,ia) * jN(0,ja)
     N2(2,1,ia,ja) = iN(1,ia) * jN(1,ja)
     N2(1,2,ia,ja) = iN(1,ia) * jN(1,ja)
     N2(2,2,ia,ja) = iN(0,ia) * jN(2,ja)
  end forall
  !
  if (ord < 3) return ! XXX Optimize!
  forall (ia=1:ina, ja=1:jna)
     N3(1,1,1,ia,ja) = iN(3,ia) * jN(0,ja)
     N3(2,1,1,ia,ja) = iN(2,ia) * jN(1,ja)
     N3(1,2,1,ia,ja) = iN(2,ia) * jN(1,ja)
     N3(2,2,1,ia,ja) = iN(1,ia) * jN(2,ja)
     N3(1,1,2,ia,ja) = iN(2,ia) * jN(1,ja)
     N3(2,1,2,ia,ja) = iN(1,ia) * jN(2,ja)
     N3(1,2,2,ia,ja) = iN(1,ia) * jN(2,ja)
     N3(2,2,2,ia,ja) = iN(0,ia) * jN(3,ja)
  end forall
  !
end subroutine TensorBasisFuns

include 'petigarat.f90.in'

end subroutine IGA_BasisFuns_2D


subroutine IGA_ShapeFuns_2D(&
     order,                 &
     nqp,nen,X,             &
     M0,M1,M2,M3,           &
     N0,N1,N2,N3,           &
     DetF,F)                &
  bind(C, name="IGA_ShapeFuns_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
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
  real   (kind=IGA_REAL), intent(inout) :: DetF(nqp)
  real   (kind=IGA_REAL), intent(inout) :: F(dim,dim,nqp)
  call GeometryMapping(&
       order,&
       nqp,nen,X,&
       M0,M1,M2,M3,&
       N0,N1,N2,N3,&
       DetF,F)
contains
include 'petigageo.f90.in'
end subroutine IGA_ShapeFuns_2D
