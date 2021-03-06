! -*- f90 -*-

subroutine IGA_Rule_GaussLegendre(q,X,W) &
  bind(C, name="IGA_Rule_GaussLegendre")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in), value :: q
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(0:q-1)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: W(0:q-1)
  integer, parameter :: rk = IGA_REAL_KIND
  select case (q)
  case (1) ! p <= 1
     X(0) = 0.0_rk
     W(0) = 2.0_rk
  case (2) ! p <= 3
     X(0) = -0.577350269189625764509148780501957456_rk ! 1/sqrt(3)
     X(1) = -X(0)
     W(0) =  1.0_rk                                    ! 1
     W(1) =  W(0)
  case (3) ! p <= 5
     X(0) = -0.774596669241483377035853079956479922_rk ! sqrt(3/5)
     X(1) =  0.0_rk                                    ! 0
     X(2) = -X(0)
     W(0) =  0.555555555555555555555555555555555556_rk ! 5/9
     W(1) =  0.888888888888888888888888888888888889_rk ! 8/9
     W(2) =  W(0)
  case (4) ! p <= 7
     X(0) = -0.861136311594052575223946488892809505_rk ! sqrt((3+2*sqrt(6/5))/7)
     X(1) = -0.339981043584856264802665759103244687_rk ! sqrt((3-2*sqrt(6/5))/7)
     X(2) = -X(1)
     X(3) = -X(0)
     W(0) =  0.347854845137453857373063949221999407_rk ! (18-sqrt(30))/36
     W(1) =  0.652145154862546142626936050778000593_rk ! (18+sqrt(30))/36
     W(2) =  W(1)
     W(3) =  W(0)
  case (5) ! p <= 9
     X(0) = -0.906179845938663992797626878299392965_rk ! 1/3*sqrt(5+2*sqrt(10/7))
     X(1) = -0.538469310105683091036314420700208805_rk ! 1/3*sqrt(5-2*sqrt(10/7))
     X(2) =  0.0_rk                                    ! 0
     X(3) = -X(1)
     X(4) = -X(0)
     W(0) =  0.236926885056189087514264040719917363_rk ! (322-13*sqrt(70))/900
     W(1) =  0.478628670499366468041291514835638193_rk ! (322+13*sqrt(70))/900
     W(2) =  0.568888888888888888888888888888888889_rk ! 128/225
     W(3) =  W(1)
     W(4) =  W(0)
  case (6) ! p <= 11
     X(0) = -0.932469514203152027812301554493994609_rk
     X(1) = -0.661209386466264513661399595019905347_rk
     X(2) = -0.238619186083196908630501721680711935_rk
     X(3) = -X(2)
     X(4) = -X(1)
     X(5) = -X(0)
     W(0) =  0.171324492379170345040296142172732894_rk
     W(1) =  0.360761573048138607569833513837716112_rk
     W(2) =  0.467913934572691047389870343989550995_rk
     W(3) =  W(2)
     W(4) =  W(1)
     W(5) =  W(0)
  case (7) ! p <= 13
     X(0) = -0.949107912342758524526189684047851262_rk
     X(1) = -0.741531185599394439863864773280788407_rk
     X(2) = -0.405845151377397166906606412076961463_rk
     X(3) =  0.0_rk
     X(4) = -X(2)
     X(5) = -X(1)
     X(6) = -X(0)
     W(0) =  0.129484966168869693270611432679082018_rk
     W(1) =  0.279705391489276667901467771423779582_rk
     W(2) =  0.381830050505118944950369775488975134_rk
     W(3) =  0.417959183673469387755102040816326531_rk
     W(4) =  W(2)
     W(5) =  W(1)
     W(6) =  W(0)
  case (8) ! p <= 15
     X(0) = -0.960289856497536231683560868569472990_rk
     X(1) = -0.796666477413626739591553936475830437_rk
     X(2) = -0.525532409916328985817739049189246349_rk
     X(3) = -0.183434642495649804939476142360183981_rk
     X(4) = -X(3)
     X(5) = -X(2)
     X(6) = -X(1)
     X(7) = -X(0)
     W(0) =  0.101228536290376259152531354309962190_rk
     W(1) =  0.222381034453374470544355994426240884_rk
     W(2) =  0.313706645877887287337962201986601313_rk
     W(3) =  0.362683783378361982965150449277195612_rk
     W(4) =  W(3)
     W(5) =  W(2)
     W(6) =  W(1)
     W(7) =  W(0)
  case (9) ! p <= 17
     X(0) = -0.968160239507626089835576202903672870_rk
     X(1) = -0.836031107326635794299429788069734877_rk
     X(2) = -0.613371432700590397308702039341474185_rk
     X(3) = -0.324253423403808929038538014643336609_rk
     X(4) =  0.0_rk
     X(5) = -X(3)
     X(6) = -X(2)
     X(7) = -X(1)
     X(8) = -X(0)
     W(0) =  0.081274388361574411971892158110523651_rk
     W(1) =  0.180648160694857404058472031242912810_rk
     W(2) =  0.260610696402935462318742869418632850_rk
     W(3) =  0.312347077040002840068630406584443666_rk
     W(4) =  0.330239355001259763164525069286974049_rk
     W(5) =  W(3)
     W(6) =  W(2)
     W(7) =  W(1)
     W(8) =  W(0)
  case (10) ! p <= 19
     X(0) = -0.973906528517171720077964012084452053_rk
     X(1) = -0.865063366688984510732096688423493049_rk
     X(2) = -0.679409568299024406234327365114873576_rk
     X(3) = -0.433395394129247190799265943165784162_rk
     X(4) = -0.148874338981631210884826001129719985_rk
     X(5) = -X(4)
     X(6) = -X(3)
     X(7) = -X(2)
     X(8) = -X(1)
     X(9) = -X(0)
     W(0) =  0.066671344308688137593568809893331793_rk
     W(1) =  0.149451349150580593145776339657697332_rk
     W(2) =  0.219086362515982043995534934228163192_rk
     W(3) =  0.269266719309996355091226921569469353_rk
     W(4) =  0.295524224714752870173892994651338329_rk
     W(5) =  W(4)
     W(6) =  W(3)
     W(7) =  W(2)
     W(8) =  W(1)
     W(9) =  W(0)
  case default
     X = 0.0_rk
     W = 0.0_rk
  end select
end subroutine IGA_Rule_GaussLegendre

subroutine IGA_Rule_GaussLobatto(q,X,W) &
  bind(C, name="IGA_Rule_GaussLobatto")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in), value :: q
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(0:q-1)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: W(0:q-1)
  integer, parameter :: rk = IGA_REAL_KIND
  select case (q)
  case (2) ! p <= 1
     X(0) = -1.0_rk
     X(1) = -X(0)
     W(0) =  1.0_rk
     W(1) =  W(0)
  case (3) ! p <= 3
     X(0) = -1.0_rk                                    ! 1
     X(1) =  0.0_rk                                    ! 0
     X(2) = -X(0)
     W(0) =  0.333333333333333333333333333333333333_rk ! 1/3
     W(1) =  1.333333333333333333333333333333333333_rk ! 4/3
     W(2) =  W(0)
  case (4) ! p <= 5
     X(0) = -1.0_rk                                    ! 1
     X(1) = -0.447213595499957939281834733746255247_rk ! 1/sqrt(5)
     X(2) = -X(1)
     X(3) = -X(0)
     W(0) =  0.166666666666666666666666666666666667_rk ! 1/6
     W(1) =  0.833333333333333333333333333333333343_rk ! 5/6
     W(2) =  W(1)
     W(3) =  W(0)
  case (5) ! p <= 7
     X(0) = -1.0_rk                                    ! 1
     X(1) = -0.654653670707977143798292456246858356_rk ! sqrt(3/7)
     X(2) =  0.0_rk                                    ! 0
     X(3) = -X(1)
     X(4) = -X(0)
     W(0) =  0.1_rk                                    !  1/10
     W(1) =  0.544444444444444444444444444444444444_rk ! 49/90
     W(2) =  0.711111111111111111111111111111111111_rk ! 32/45
     W(3) =  W(1)
     W(4) =  W(0)
  case (6) ! p <= 9
     X(0) = -1.0_rk                                    ! 1
     X(1) = -0.765055323929464692851002973959338150_rk ! sqrt((7+2*sqrt(7))/21)
     X(2) = -0.285231516480645096314150994040879072_rk ! sqrt((7-2*sqrt(7))/21)
     X(3) = -X(2)
     X(4) = -X(1)
     X(5) = -X(0)
     W(0) =  0.066666666666666666666666666666666667_rk ! 1/15
     W(1) =  0.378474956297846980316612808212024652_rk ! (14-sqrt(7))/30
     W(2) =  0.554858377035486353016720525121308681_rk ! (14+sqrt(7))/30
     W(3) =  W(2)
     W(4) =  W(1)
     W(5) =  W(0)
  case (7) ! p <= 11
     X(0) = -1.0_rk
     X(1) = -0.830223896278566929872032213967465140_rk
     X(2) = -0.468848793470714213803771881908766329_rk
     X(3) =  0.0_rk
     X(4) = -X(2)
     X(5) = -X(1)
     X(6) = -X(0)
     W(0) =  0.047619047619047619047619047619047619_rk
     W(1) =  0.276826047361565948010700406290066293_rk
     W(2) =  0.431745381209862623417871022281362278_rk
     W(3) =  0.487619047619047619047619047619047619_rk
     W(4) =  W(2) 
     W(5) =  W(1)
     W(6) =  W(0)
  case (8) ! p <= 13
     X(0) = -1.0_rk
     X(1) = -0.871740148509606615337445761220663438_rk
     X(2) = -0.591700181433142302144510731397953190_rk
     X(3) = -0.209299217902478868768657260345351255_rk
     X(4) = -X(3)
     X(5) = -X(2)
     X(6) = -X(1)
     X(7) = -X(0)
     W(0) =  0.035714285714285714285714285714285714_rk
     W(1) =  0.210704227143506039382992065775756324_rk
     W(2) =  0.341122692483504364764240677107748172_rk
     W(3) =  0.412458794658703881567052971402209789_rk
     W(4) =  W(3)
     W(5) =  W(2)
     W(6) =  W(1)
     W(7) =  W(0)
  case (9) ! p <= 15
     X(0) = -1.0_rk
     X(1) = -0.899757995411460157312345244418337958_rk
     X(2) = -0.677186279510737753445885427091342451_rk
     X(3) = -0.363117463826178158710752068708659213_rk
     X(4) =  0.0_rk
     X(5) = -X(3)
     X(6) = -X(2)
     X(7) = -X(1)
     X(8) = -X(0)
     W(0) =  0.027777777777777777777777777777777778_rk
     W(1) =  0.165495361560805525046339720029208306_rk
     W(2) =  0.274538712500161735280705618579372726_rk
     W(3) =  0.346428510973046345115131532139718288_rk
     W(4) =  0.371519274376417233560090702947845805_rk
     W(5) =  W(3)
     W(6) =  W(2)
     W(7) =  W(1)
     W(8) =  W(0)
  case (10) ! p <= 17
     X(0) = -1.0_rk
     X(1) = -0.919533908166458813828932660822338134_rk
     X(2) = -0.738773865105505075003106174859830725_rk
     X(3) = -0.477924949810444495661175092731257998_rk
     X(4) = -0.165278957666387024626219765958173533_rk
     X(5) = -X(4)
     X(6) = -X(3)
     X(7) = -X(2)
     X(8) = -X(1)
     X(9) = -X(0)
     W(0) =  0.022222222222222222222222222222222222_rk
     W(1) =  0.133305990851070111126227170755392898_rk
     W(2) =  0.224889342063126452119457821731047843_rk
     W(3) =  0.292042683679683757875582257374443892_rk
     W(4) =  0.327539761183897456656510527916893145_rk
     W(5) =  W(4)
     W(6) =  W(3)
     W(7) =  W(2)
     W(8) =  W(1)
     W(9) =  W(0)
  case default
     X = 0.0_rk
     W = 0.0_rk
  end select
end subroutine IGA_Rule_GaussLobatto
