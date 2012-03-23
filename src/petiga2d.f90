subroutine IGA_Quadrature_2D(&
     inq,iX,iW, &
     jnq,jX,jW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_2D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer, parameter :: dim = 2
  integer(kind=C_INT   ), intent(in),value :: inq
  integer(kind=C_INT   ), intent(in),value :: jnq
  real   (kind=C_DOUBLE), intent(in)  :: iX(inq), iW(inq)
  real   (kind=C_DOUBLE), intent(in)  :: jX(jnq), jW(jnq)
  real   (kind=C_DOUBLE), intent(out) :: X(dim,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: W(inq,jnq)
  integer :: iq
  integer :: jq
  forall (iq=1:inq, jq=1:jnq)
     X(:,iq,jq) = (/ iX(iq),  jX(jq)/)
     W(  iq,jq) =    iW(iq) * jW(jq)
  end forall
end subroutine IGA_Quadrature_2D

subroutine IGA_ShapeFuns_2D(&
     geometry,rational,  &
     inq,ina,ind,iJ,iN,  &
     jnq,jna,jnd,jJ,jN,  &
     Cw,                 &
     detJ,J,N0,N1,N2,N3) &
  bind(C, name="IGA_ShapeFuns_2D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 2
  integer(kind=C_INT   ), intent(in),value :: geometry
  integer(kind=C_INT   ), intent(in),value :: rational
  integer(kind=C_INT   ), intent(in),value :: inq, ina, ind
  integer(kind=C_INT   ), intent(in),value :: jnq, jna, jnd
  real   (kind=C_DOUBLE), intent(in)  :: iJ, iN(0:ind,ina,inq)
  real   (kind=C_DOUBLE), intent(in)  :: jJ, jN(0:jnd,jna,jnq)
  real   (kind=C_DOUBLE), intent(in)  :: Cw(dim+1,ina,jna)
  real   (kind=C_DOUBLE), intent(out) :: detJ(     inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: J(dim,dim,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N0(       ina,jna,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N1(   dim,ina,jna,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N2(dim**2,ina,jna,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim**3,ina,jna,inq,jnq)
  integer :: ia,iq
  integer :: ja,jq
  integer :: na,nd

  real(kind=C_DOUBLE) :: C(dim,ina,jna)
  real(kind=C_DOUBLE) :: w(    ina,jna)
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
                detJ( iq,jq),&
                J(:,:,iq,jq),&
                N0(  :,:,iq,jq),&
                N1(:,:,:,iq,jq),&
                N2(:,:,:,iq,jq),&
                N3(:,:,:,iq,jq))
           detJ( iq,jq) = detJ( iq,jq) * (iJ*jJ) 
           J(1,:,iq,jq) = J(1,:,iq,jq) * iJ
           J(2,:,iq,jq) = J(2,:,iq,jq) * jJ
        else
           detJ( iq,jq) = (iJ*jJ)
           J(:,:,iq,jq) = 0
           J(1,1,iq,jq) = iJ
           J(2,2,iq,jq) = jJ
        end if
     end do
  end do

contains

subroutine TensorBasisFuns(&
     ina,ind,iN,&
     jna,jnd,jN,&
     nd,N0,N1,N2,N3)
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 2
  integer(kind=C_INT   ), intent(in),value :: ina, ind
  integer(kind=C_INT   ), intent(in),value :: jna, jnd
  real   (kind=C_DOUBLE), intent(in)  :: iN(0:ind,ina)
  real   (kind=C_DOUBLE), intent(in)  :: jN(0:jnd,jna)
  integer(kind=C_INT   ), intent(in)  :: nd
  real   (kind=C_DOUBLE), intent(out) :: N0(            ina,jna)
  real   (kind=C_DOUBLE), intent(out) :: N1(        dim,ina,jna)
  real   (kind=C_DOUBLE), intent(out) :: N2(    dim,dim,ina,jna)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim,dim,dim,ina,jna)
  integer :: ia, ja
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
include 'petigamap.f90.in'

end subroutine IGA_ShapeFuns_2D
