#!/bin/sh

if [ -z $1 ]
then >&2 echo "usage: $0 /path/to/trilinos-12.0.1-Source [install-prefix]"; exit 1
fi
if [ -z $2 ]
then PREFIX=$PWD/sacado
else PREFIX=$2
fi

TRILINOS_SRCDIR=$1
BUILDDIR=$(mktemp -d /tmp/sacado-build-XXXXXX)

cd $BUILDDIR

cmake \
-D CMAKE_INSTALL_PREFIX:PATH=${PREFIX} \
-D BUILD_SHARED_LIBS=ON \
-D Trilinos_ENABLE_Fortran:BOOL=OFF \
-D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=OFF \
-D Trilinos_ENABLE_CPACK_PACKAGING:BOOL=OFF \
-D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
-D Trilinos_ENABLE_TESTS:BOOL=OFF \
-D Trilinos_ENABLE_Sacado=ON \
-D Sacado_ENABLE_VIEW_SPEC:BOOL=OFF \
$TRILINOS_SRCDIR

make
make install

rm -rf $BUILDDIR
