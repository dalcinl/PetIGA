#ifndef PETIGA_CXX_LOOKUP_HPP
#define PETIGA_CXX_LOOKUP_HPP

//#define DIM 3
//#define DIM_MIN 1
//#define DIM_MAX 3

//#define DEG 2
//#define DEG_MIN 1
//#define DEG_MAX 5

//#define DOF 1
//#define DOF_MIN 1
//#define DOF_MAX 5

#ifdef DIM
#define DIM_MIN DIM
#define DIM_MAX DIM
#endif
#ifndef DIM_MIN
#define DIM_MIN 1
#endif
#ifndef DIM_MAX
#define DIM_MAX 3
#endif

#ifdef DEG
#define DEG_MIN DEG
#define DEG_MAX DEG
#endif
#ifndef DEG_MIN
#define DEG_MIN 1
#endif
#ifndef DEG_MAX
#define DEG_MAX 3
#endif

#ifdef DOF
#define DOF_MIN DOF
#define DOF_MAX DOF
#endif
#ifndef DOF_MIN
#define DOF_MIN 1
#endif
#ifndef DOF_MAX
#define DOF_MAX 5
#endif

#define Enable(VAR,N) \
  (((VAR##_MIN) <= (N)) && ((VAR##_MAX) >= (N)))

#if Enable(DIM,1)
#define EnableDim1(arg) arg
#else
#define EnableDim1(arg)
#endif
#if Enable(DIM,2)
#define EnableDim2(arg) arg
#else
#define EnableDim2(arg)
#endif
#if Enable(DIM,3)
#define EnableDim3(arg) arg
#else
#define EnableDim3(arg)
#endif

#if Enable(DEG,1)
#define EnableDeg1(arg) arg
#else
#define EnableDeg1(arg)
#endif
#if Enable(DEG,2)
#define EnableDeg2(arg) arg
#else
#define EnableDeg2(arg)
#endif
#if Enable(DEG,3)
#define EnableDeg3(arg) arg
#else
#define EnableDeg3(arg)
#endif
#if Enable(DEG,4)
#define EnableDeg4(arg) arg
#else
#define EnableDeg4(arg)
#endif
#if Enable(DEG,5)
#define EnableDeg5(arg) arg
#else
#define EnableDeg5(arg)
#endif

#if Enable(DEG,1)
#define EnableDeg1(arg) arg
#else
#define EnableDeg1(arg)
#endif
#if Enable(DEG,2)
#define EnableDeg2(arg) arg
#else
#define EnableDeg2(arg)
#endif
#if Enable(DEG,3)
#define EnableDeg3(arg) arg
#else
#define EnableDeg3(arg)
#endif
#if Enable(DEG,4)
#define EnableDeg4(arg) arg
#else
#define EnableDeg4(arg)
#endif
#if Enable(DEG,5)
#define EnableDeg5(arg) arg
#else
#define EnableDeg5(arg)
#endif

#if Enable(DOF,1)
#define EnableDof1(arg) arg
#else
#define EnableDof1(arg)
#endif
#if Enable(DOF,2)
#define EnableDof2(arg) arg
#else
#define EnableDof2(arg)
#endif
#if Enable(DOF,3)
#define EnableDof3(arg) arg
#else
#define EnableDof3(arg)
#endif
#if Enable(DOF,4)
#define EnableDof4(arg) arg
#else
#define EnableDof4(arg)
#endif
#if Enable(DOF,5)
#define EnableDof5(arg) arg
#else
#define EnableDof5(arg)
#endif
#if Enable(DOF,6)
#define EnableDof6(arg) arg
#else
#define EnableDof6(arg)
#endif
#if Enable(DOF,7)
#define EnableDof7(arg) arg
#else
#define EnableDof7(arg)
#endif
#if Enable(DOF,8)
#define EnableDof8(arg) arg
#else
#define EnableDof8(arg)
#endif
#if Enable(DOF,9)
#define EnableDof9(arg) arg
#else
#define EnableDof9(arg)
#endif

#define SelectDim(Template,dim,nen,dof)               \
  switch (dim) {                                      \
    HandleDim(Template,3,nen,dof)                     \
    HandleDim(Template,2,nen,dof)                     \
    HandleDim(Template,1,nen,dof)                     \
  default: break; }                                  //
#define HandleDim(Template,dim,nen,dof)               \
  EnableDim##dim(                                     \
  case dim: {SelectNen##dim(Template,dim,nen,dof)}    \
  break;)                                            //

#define SelectNen1(Template,dim,nen,dof)              \
  switch (nen) {                                      \
    HandleDeg1(Template,dim,1,dof)                    \
    HandleDeg1(Template,dim,2,dof)                    \
    HandleDeg1(Template,dim,3,dof)                    \
    HandleDeg1(Template,dim,4,dof)                    \
    HandleDeg1(Template,dim,5,dof)                    \
    default: break; }
#define HandleDeg1(Template,dim,p,dof)                \
  EnableDeg##p(                                       \
  case (p+1): {                                       \
    SelectDof(Template,dim,(p+1),dof)                 \
  } break;)                                          //

#define SelectNen2(Template,dim,nen,dof)              \
  switch (nen) {                                      \
    HandleDeg2(Template,dim,1,1,dof)                  \
    HandleDeg2(Template,dim,1,2,dof)                  \
    HandleDeg2(Template,dim,1,3,dof)                  \
    HandleDeg2(Template,dim,2,2,dof)                  \
    HandleDeg2(Template,dim,1,4,dof)                  \
    HandleDeg2(Template,dim,2,3,dof)                  \
    HandleDeg2(Template,dim,2,4,dof)                  \
    HandleDeg2(Template,dim,3,3,dof)                  \
    HandleDeg2(Template,dim,2,5,dof)                  \
    HandleDeg2(Template,dim,3,4,dof)                  \
    HandleDeg2(Template,dim,3,5,dof)                  \
    HandleDeg2(Template,dim,4,4,dof)                  \
    HandleDeg2(Template,dim,4,5,dof)                  \
    HandleDeg2(Template,dim,5,5,dof)                  \
  default: break; }                                  //
#define HandleDeg2(Template,dim,p,q,dof)              \
  EnableDeg##p(EnableDeg##q(                          \
  case ((p+1)*(q+1)): {                               \
    SelectDof(Template,dim,((p+1)*(q+1)),dof)         \
  } break;))                                         //

#define SelectNen3(Template,dim,nen,dof)              \
  switch (nen) {                                      \
    HandleDeg3(Template,dim,1,1,1,dof)                \
    HandleDeg3(Template,dim,1,1,2,dof)                \
    HandleDeg3(Template,dim,1,1,3,dof)                \
    HandleDeg3(Template,dim,1,2,2,dof)                \
    HandleDeg3(Template,dim,1,1,4,dof)                \
    HandleDeg3(Template,dim,1,2,3,dof)                \
    HandleDeg3(Template,dim,2,2,2,dof)                \
    HandleDeg3(Template,dim,1,2,4,dof)                \
    HandleDeg3(Template,dim,1,3,3,dof)                \
    HandleDeg3(Template,dim,2,2,3,dof)                \
    HandleDeg3(Template,dim,1,3,4,dof)                \
    HandleDeg3(Template,dim,2,2,4,dof)                \
    HandleDeg3(Template,dim,2,3,3,dof)                \
    HandleDeg3(Template,dim,1,4,4,dof)                \
    HandleDeg3(Template,dim,2,2,5,dof)                \
    HandleDeg3(Template,dim,2,3,4,dof)                \
    HandleDeg3(Template,dim,3,3,3,dof)                \
    HandleDeg3(Template,dim,2,3,5,dof)                \
    HandleDeg3(Template,dim,2,4,4,dof)                \
    HandleDeg3(Template,dim,3,3,4,dof)                \
    HandleDeg3(Template,dim,2,4,5,dof)                \
    HandleDeg3(Template,dim,3,3,5,dof)                \
    HandleDeg3(Template,dim,3,4,4,dof)                \
    HandleDeg3(Template,dim,2,5,5,dof)                \
    HandleDeg3(Template,dim,3,4,5,dof)                \
    HandleDeg3(Template,dim,4,4,4,dof)                \
    HandleDeg3(Template,dim,3,5,5,dof)                \
    HandleDeg3(Template,dim,4,4,5,dof)                \
    HandleDeg3(Template,dim,4,5,5,dof)                \
    HandleDeg3(Template,dim,5,5,5,dof)                \
  default: break; }                                  //
#define HandleDeg3(Template,dim,p,q,r,dof)            \
  EnableDeg##p(EnableDeg##q(EnableDeg##r(             \
  case ((p+1)*(q+1)*(r+1)): {                         \
    SelectDof(Template,dim,((p+1)*(q+1)*(r+1)),dof)   \
  } break;)))                                        //

#define SelectDof(Template,dim,nen,dof)               \
  switch (dof) {                                      \
    HandleDof(Template,dim,nen,1)                     \
    HandleDof(Template,dim,nen,2)                     \
    HandleDof(Template,dim,nen,3)                     \
    HandleDof(Template,dim,nen,4)                     \
    HandleDof(Template,dim,nen,5)                     \
    HandleDof(Template,dim,nen,6)                     \
    HandleDof(Template,dim,nen,7)                     \
    HandleDof(Template,dim,nen,8)                     \
    HandleDof(Template,dim,nen,9)                     \
  default: break; }                                  //
#define HandleDof(Template,dim,nen,dof)               \
  EnableDof##dof(                                     \
  case dof: { Template TParamList(dim,nen,dof); }     \
  break;)                                            //
#define TParamList(dim,nen,dof) <dim,nen,dof>


#define LookupTemplate(Template,dim,nen,dof)   \
  SelectDim(Template,dim,nen,dof)

#define LookupTemplateSet(Symbol,q,Template)  \
  LookupTemplate(Symbol = Template,q->dim,q->nen,q->dof)

#define LookupTemplateChk(Symbol,q,Template)                          \
  do{if(PetscUnlikely(!Symbol)){                                      \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,                        \
               #Template " unavailable for dim=%D, nen=%D, dof=%D",   \
               q->dim,q->nen,q->dof);return PETSC_ERR_USER;}}while(0)

#endif
