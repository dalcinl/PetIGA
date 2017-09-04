#ifndef PETIGA_FAD_ADTYPE_HPP
#define PETIGA_FAD_ADTYPE_HPP

#if defined(MINIFAD)
#include <minifad.hpp>
#elif defined(SACADO)
#include <Sacado.hpp>
#elif defined(FADBAD)
#include <fadiff.h>
#else
#include <minifad.hpp>
#endif

namespace PetIGA {

template <typename ScalarT, int nen, int dof>
struct AD {

  static const int N = nen*dof;

#if defined(MINIFAD_HPP)

  typedef typename MiniFAD::Fad<ScalarT, N> Type;
  static inline ScalarT& Def(Type& x, const ScalarT& u, int i)
  { return x.def(u, i, N); }

#elif defined(SACADO_HPP)

  typedef typename Sacado::Fad::SFad<ScalarT, N> Type;
  static inline ScalarT& Def(Type& x, const ScalarT& u, int i)
  { x = u; x.diff(i, N); return x.fastAccessDx(i); }

#elif defined(_FADBAD_H)

  typedef typename fadbad::F<ScalarT, N> Type;
  static inline ScalarT& Def(Type& x, const ScalarT& u, int i)
  { x = u; return x.diff(i); }

#endif

  static inline void Input(Type X[], const ScalarT U[])
  { for (int i=0; i<N; i++) Def(X[i], U[i], i); }

  static inline void Input(Type X[], const ScalarT U[], const ScalarT& s)
  { for (int i=0; i<N; i++) Def(X[i], U[i], i) = s; }

  static inline void Input(Type (&X)[nen][dof], const ScalarT (&U)[nen][dof])
  { Input(&X[0][0], &U[0][0]); }

  static inline void Input(Type (&X)[nen][dof], const ScalarT (&U)[nen][dof], const ScalarT& s)
  { Input(&X[0][0], &U[0][0], s); }

  static inline void Clear(Type R[])
  { for (int i=0; i<N; i++) R[i] = 0.0f; }

  static inline void Output(const Type R[], ScalarT J[])
  {
    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        #ifdef _FADBAD_H
        J[i*N+j] = R[i].deriv(j);
        #else
        J[i*N+j] = R[i].dx(j);
        #endif
  }
  static inline void Output(const Type (&R)[nen][dof], ScalarT (&J)[nen][dof][nen][dof])
  { Output(&R[0][0], &J[0][0][0][0]);}

}; // struct AD

} // namespace PetIGA

#endif
