#include <petiga.h>

typedef struct {
  PetscReal theta;
  PetscReal alpha;
  PetscReal cbar;
} Params;

template<typename Scalar>
static inline void Mobility(const Params *,const Scalar& c,Scalar& M)
{
  M = c*(1-c);
}

template<typename Scalar>
static inline void Mobility(const Params *,const Scalar& c,Scalar& M,Scalar& dM)
{
  M  = c*(1-c);
  dM = 1-2*c;
}

template<typename Scalar>
static inline void Mobility(const Params *,Scalar& c,Scalar& M,Scalar& dM,Scalar& d2M)
{
  M   = c*(1-c);
  dM  = 1-2*c;
  d2M = -2;
}

template<typename Scalar>
static inline void ChemicalPotential(const Params *params,const Scalar& c,Scalar& mu)
{
  PetscReal theta  = params->theta;
  PetscReal alpha  = params->alpha;
  mu = 3*alpha * (0.5/theta*log(c/(1-c)) + 1 - 2*c);
}

template<typename Scalar>
static inline void ChemicalPotential(const Params *params,const Scalar& c,Scalar& mu,Scalar& dmu)
{
  PetscReal theta  = params->theta;
  PetscReal alpha  = params->alpha;
  mu  = 3*alpha * (0.5/theta*log(c/(1-c)) + 1 - 2*c);
  dmu = 3*alpha * (0.5/theta*1/(c*(1-c)) - 2);
}

template<typename Scalar>
static inline void ChemicalPotential(const Params* params,Scalar& c,Scalar& mu,Scalar& dmu,Scalar& d2mu)
{
  PetscReal theta  = params->theta;
  PetscReal alpha  = params->alpha;
  mu   = 3*alpha * (0.5/theta*log(c/(1-c)) + 1 - 2*c);
  dmu  = 3*alpha * (0.5/theta*1/(c*(1-c)) - 2);
  d2mu = 3*alpha * (0.5/theta*(2*c-1)/(c*c*(1-c)*(1-c)));
}
