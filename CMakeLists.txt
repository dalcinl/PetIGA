cmake_minimum_required (VERSION 2.6.2)
project (PetIGA C)

if (0)
  # Use Jed Brown's FindPETSc.cmake
  set (CMAKE_MODULE_PATH ${PetIGA_SOURCE_DIR}/config)
  find_package (PETSc)
else ()
  find_path (PETSC_DIR include/petsc.h HINTS ENV PETSC_DIR PATHS $ENV{HOME}/petsc DOC "PETSc top-level directory")
  set (PETSC_ARCH $ENV{PETSC_ARCH} CACHE STRING "PETSc configuration")
  find_path (PETSC_INCLUDE_DIR  petsc.h HINTS "${PETSC_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
  find_path (PETSC_INCLUDE_CONF petscconf.h HINTS "${PETSC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/include" "include" NO_DEFAULT_PATH)
  mark_as_advanced (PETSC_INCLUDE_DIR PETSC_INCLUDE_CONF)
  set (PETSC_INCLUDES ${PETSC_INCLUDE_CONF} ${PETSC_INCLUDE_DIR} CACHE PATH "PETSc include paths" FORCE)
  find_library (PETSC_LIBRARIES NAMES petsc HINTS "${PETSC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/lib" "lib" NO_DEFAULT_PATH)
  find_path (PETSC_CONF_DIR petscrules HINTS "${PETSC_DIR}/${PETSC_ARCH}" PATH_SUFFIXES "lib/petsc/conf" "conf" NO_DEFAULT_PATH)
  mark_as_advanced (PETSC_CONF_DIR)
  include (${PETSC_CONF_DIR}/PETScBuildInternal.cmake)
  mark_as_advanced (PETSC_CLANGUAGE_Cxx)
endif ()

if (PETSC_CLANGUAGE_Cxx)
  enable_language (CXX)
endif ()
enable_language (Fortran)

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PetIGA_BINARY_DIR}/lib"     CACHE PATH "Output directory for PetIGA archives")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PetIGA_BINARY_DIR}/lib"     CACHE PATH "Output directory for PetIGA libraries")
set (CMAKE_Fortran_MODULE_DIRECTORY "${PetIGA_BINARY_DIR}/include" CACHE PATH "Output directory for Fortran modules")
mark_as_advanced (CMAKE_ARCHIVE_OUTPUT_DIRECTORY CMAKE_LIBRARY_OUTPUT_DIRECTORY CMAKE_Fortran_MODULE_DIRECTORY)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/petiga" CACHE PATH "PetIGA install prefix" FORCE)
endif ()
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# XXX @rpath/libpetiga.dylib
set (CMAKE_MACOSX_RPATH NO)
set (CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

include_directories (${PETSC_INCLUDES} ${PETSC_PACKAGE_INCLUDES}
                    "${PetIGA_SOURCE_DIR}/${PETSC_ARCH}/include"
                    "${PetIGA_SOURCE_DIR}/include")

file (GLOB PetIGA_SOURCES_C RELATIVE ${PetIGA_SOURCE_DIR} ${PetIGA_SOURCE_DIR}/src/*.c)
file (GLOB PetIGA_SOURCES_F RELATIVE ${PetIGA_SOURCE_DIR} ${PetIGA_SOURCE_DIR}/src/*.F90)
set  (PetIGA_SOURCES_ALL ${PetIGA_SOURCES_C} ${PetIGA_SOURCES_F})
if (PETSC_CLANGUAGE_Cxx)
  foreach (file ${PetIGA_SOURCES_C})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)
  endforeach ()
endif ()

set (BUILD_SHARED_LIBS 1)
set (BUILD_STATIC_LIBS 0)

if (BUILD_SHARED_LIBS)
  add_library (petiga ${PetIGA_SOURCES_ALL})
  target_link_libraries (petiga ${PETSC_LIBRARIES} ${PETSC_PACKAGE_LIBS})
  install (TARGETS petiga LIBRARY DESTINATION lib)
endif ()

if (BUILD_STATIC_LIBS)
  add_library (petiga-static STATIC ${PetIGA_SOURCES_ALL})
  set_target_properties (petiga-static PROPERTIES PREFIX "lib" OUTPUT_NAME "petiga")
  install (TARGETS petiga-static ARCHIVE DESTINATION lib)
endif ()

install (DIRECTORY include/               DESTINATION include FILES_MATCHING PATTERN "*.h"    )
install (DIRECTORY ${PETSC_ARCH}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h"    )
install (DIRECTORY ${PETSC_ARCH}/include/ DESTINATION include FILES_MATCHING PATTERN "*.mod"  )
install (DIRECTORY lib/petiga/conf/       DESTINATION lib/petiga/conf)
install (CODE "execute_process(COMMAND sed -i.arch -e s/$\(PETSC_ARCH\).//g \"${CMAKE_INSTALL_PREFIX}/lib/petiga/conf/variables\")")
install (CODE "file(REMOVE \"${CMAKE_INSTALL_PREFIX}/lib/petiga/conf/variables.arch\")")
