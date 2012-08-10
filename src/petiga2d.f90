subroutine IGA_Quadrature_2D(&
     inq,iX,iW,iL,           &
     jnq,jX,jW,jL,           &
     W,J,X,L)                &
  bind(C, name="IGA_Quadrature_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
  integer(kind=IGA_INT ), intent(in),value :: inq
  integer(kind=IGA_INT ), intent(in),value :: jnq
  real   (kind=IGA_REAL), intent(in)  :: iX(inq), iW(inq), iL
  real   (kind=IGA_REAL), intent(in)  :: jX(jnq), jW(jnq), jL
  real   (kind=IGA_REAL), intent(out) :: W(    inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: J(    inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: X(dim,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: L(dim,inq,jnq)
  integer(kind=IGA_INT ) :: iq
  integer(kind=IGA_INT ) :: jq
  forall (iq=1:inq, jq=1:jnq)
     !
     W(iq,jq) = iW(iq) * jW(jq)
     J(iq,jq) = iL * jL
     !
     X(1,iq,jq) = iX(iq)
     X(2,iq,jq) = jX(jq)
     L(1,iq,jq) = iL
     L(2,iq,jq) = jL
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
     DetJac,F,G)            &
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
end subroutine IGA_ShapeFuns_2D
