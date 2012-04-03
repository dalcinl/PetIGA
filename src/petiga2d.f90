subroutine IGA_Quadrature_2D(&
     inq,iX,iW, &
     jnq,jX,jW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
  integer(kind=IGA_INT ), intent(in),value :: inq
  integer(kind=IGA_INT ), intent(in),value :: jnq
  real   (kind=IGA_REAL), intent(in)  :: iX(inq), iW(inq)
  real   (kind=IGA_REAL), intent(in)  :: jX(jnq), jW(jnq)
  real   (kind=IGA_REAL), intent(out) :: X(dim,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: W(inq,jnq)
  integer(kind=IGA_INT ) :: iq
  integer(kind=IGA_INT ) :: jq
  forall (iq=1:inq, jq=1:jnq)
     X(:,iq,jq) = (/ iX(iq),  jX(jq)/)
     W(  iq,jq) =    iW(iq) * jW(jq)
  end forall
end subroutine IGA_Quadrature_2D

subroutine IGA_ShapeFuns_2D(&
     geometry,rational,     &
     inq,ina,ind,iJ,iN,     &
     jnq,jna,jnd,jJ,jN,     &
     Cw,detJac,Jac,         &
     N0,N1,N2,N3)           &
  bind(C, name="IGA_ShapeFuns_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
  integer(kind=IGA_INT ), intent(in),value :: geometry
  integer(kind=IGA_INT ), intent(in),value :: rational
  integer(kind=IGA_INT ), intent(in),value :: inq, ina, ind
  integer(kind=IGA_INT ), intent(in),value :: jnq, jna, jnd
  real   (kind=IGA_REAL), intent(in)  :: iJ, iN(0:ind,ina,inq)
  real   (kind=IGA_REAL), intent(in)  :: jJ, jN(0:jnd,jna,jnq)
  real   (kind=IGA_REAL), intent(in)  :: Cw(dim+1,ina,jna)
  real   (kind=IGA_REAL), intent(out) :: detJac(     inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: Jac(dim,dim,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: N0(       ina,jna,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: N1(   dim,ina,jna,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: N2(dim**2,ina,jna,inq,jnq)
  real   (kind=IGA_REAL), intent(out) :: N3(dim**3,ina,jna,inq,jnq)

  integer(kind=IGA_INT ) :: ia,iq
  integer(kind=IGA_INT ) :: ja,jq
  integer(kind=IGA_INT ) :: na,nd
  real   (kind=IGA_REAL) :: C(dim,ina,jna)
  real   (kind=IGA_REAL) :: w(    ina,jna)

  if (geometry /= 0) then
     C = Cw(1:dim,:,:)
  end if
  if (rational /= 0) then
     w = Cw(dim+1,:,:)
  end if

  nd = max(1,min(ind,jnd,3))
  na = ina*jna
  do jq=1,jnq
     do iq=1,inq
        call TensorBasisFuns(&
             ina,ind,iN(:,:,iq),&
             jna,jnd,jN(:,:,jq),&
             nd,&
             N0(  :,:,iq,jq),&
             N1(:,:,:,iq,jq),&
             N2(:,:,:,iq,jq),&
             N3(:,:,:,iq,jq))
        if (rational /= 0) then
           call Rationalize(&
                nd,na,w,&
                N0(  :,:,iq,jq),&
                N1(:,:,:,iq,jq),&
                N2(:,:,:,iq,jq),&
                N3(:,:,:,iq,jq))
        endif
        if (geometry /= 0) then
           call GeometryMap(&
                nd,na,C,&
                detJac( iq,jq),&
                Jac(:,:,iq,jq),&
                N0(  :,:,iq,jq),&
                N1(:,:,:,iq,jq),&
                N2(:,:,:,iq,jq),&
                N3(:,:,:,iq,jq))
           detJac( iq,jq) = detJac( iq,jq) * (iJ*jJ)
           Jac(1,:,iq,jq) = Jac(1,:,iq,jq) * iJ
           Jac(2,:,iq,jq) = Jac(2,:,iq,jq) * jJ
        else
           detJac( iq,jq) = (iJ*jJ)
           Jac(:,:,iq,jq) = 0
           Jac(1,1,iq,jq) = iJ
           Jac(2,2,iq,jq) = jJ
        end if
     end do
  end do

contains

pure subroutine TensorBasisFuns(&
     ina,ind,iN,&
     jna,jnd,jN,&
     nd,N0,N1,N2,N3)
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 2
  integer(kind=IGA_INT ), intent(in),value :: ina, ind
  integer(kind=IGA_INT ), intent(in),value :: jna, jnd
  real   (kind=IGA_REAL), intent(in)  :: iN(0:ind,ina)
  real   (kind=IGA_REAL), intent(in)  :: jN(0:jnd,jna)
  integer(kind=IGA_INT ), intent(in)  :: nd
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
  if (nd < 2) return ! XXX Optimize!
  forall (ia=1:ina, ja=1:jna)
     N2(1,1,ia,ja) = iN(2,ia) * jN(0,ja)
     N2(2,1,ia,ja) = iN(1,ia) * jN(1,ja)
     N2(1,2,ia,ja) = iN(1,ia) * jN(1,ja)
     N2(2,2,ia,ja) = iN(0,ia) * jN(2,ja)
  end forall
  !
  if (nd < 3) return ! XXX Optimize!
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
include 'petigageo.f90.in'

end subroutine IGA_ShapeFuns_2D
