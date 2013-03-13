cmake_minimum_required (VERSION 2.6.2)
project (PetIGA C)

if (0)
  # Use Jed Brown's FindPETSc.cmake
  set (CMAKE_MODULE_PATH ${PetIGA_SOURCE_DIR}/conf/cmake)
  find_package (PETSc)
else ()
  find_path (PETSC_DIR include/petsc.h HINTS ENV PETSC_DIR DOC "PETSc Directory")
  set (PETSC_ARCH $ENV{PETSC_ARCH})
  find_path (PETSC_INCLUDE_DIR  petsc.h HINTS "${PETSC_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
  find_path (PETSC_INCLUDE_CONF petscconf.h HINTS "${PETSC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/include" NO_DEFAULT_PATH)
  mark_as_advanced (PETSC_INCLUDE_DIR PETSC_INCLUDE_CONF)
  set (PETSC_INCLUDES ${PETSC_INCLUDE_CONF} ${PETSC_INCLUDE_DIR} CACHE PATH "PETSc include paths" FORCE)
  set (PETSC_DEFINITIONS "-D__INSDIR__=" CACHE STRING "PETSc preprocesor definitions" FORCE)
  find_path (PETSC_LIB_DIR NAMES "" HINTS "${PETSC_DIR}" PATH_SUFFIXES "${PETSC_ARCH}/lib" NO_DEFAULT_PATH)
  find_library (PETSC_LIBRARY NAMES petsc HINTS "${PETSC_LIB_DIR}" NO_DEFAULT_PATH)
  set (PETSC_LIBRARIES ${PETSC_LIBRARY} CACHE FILEPATH "PETSc library" FORCE)
  include (${PETSC_DIR}/${PETSC_ARCH}/conf/PETScConfig.cmake)
endif()

if (PETSC_CLANGUAGE_Cxx)
  enable_language (CXX)
endif ()
enable_language (Fortran)

include_directories (${PETSC_INCLUDES} ${PETSC_PACKAGE_INCLUDES}
                    "${PetIGA_SOURCE_DIR}/${PETSC_ARCH}/include"
                    "${PetIGA_SOURCE_DIR}/include")
add_definitions (${PETSC_DEFINITIONS})

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PetIGA_BINARY_DIR}/lib" CACHE PATH "Output directory for PetIGA archives")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PetIGA_BINARY_DIR}/lib" CACHE PATH "Output directory for PetIGA libraries")
set (CMAKE_Fortran_MODULE_DIRECTORY "${PetIGA_BINARY_DIR}/include" CACHE PATH "Output directory for fortran *.mod files")
mark_as_advanced (CMAKE_ARCHIVE_OUTPUT_DIRECTORY CMAKE_LIBRARY_OUTPUT_DIRECTORY CMAKE_Fortran_MODULE_DIRECTORY)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

file (GLOB PetIGA_SOURCES_C RELATIVE ${PetIGA_SOURCE_DIR} ${PetIGA_SOURCE_DIR}/src/*.c)
file (GLOB PetIGA_SOURCES_F RELATIVE ${PetIGA_SOURCE_DIR} ${PetIGA_SOURCE_DIR}/src/*.[Ff]90)
set  (PetIGA_SOURCES_ALL ${PetIGA_SOURCES_C} ${PetIGA_SOURCES_F})

add_library (petiga ${PetIGA_SOURCES_ALL})
target_link_libraries (petiga ${PETSC_LIBRARIES} ${PETSC_PACKAGE_LIBS})

if (PETSC_CLANGUAGE_Cxx)
  foreach (file ${PetIGA_SOURCES_C})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)
  endforeach ()
endif ()

#set (BUILD_SHARED_LIBS NO)

#if (BUILD_SHARED_LIBS)
#  add_library (petiga-static STATIC ${PetIGA_SOURCES_ALL})
#  set_target_properties (petiga-static PROPERTIES OUTPUT_NAME "petiga")
#  set_target_properties (petiga-static PROPERTIES PREFIX "lib")
#endif()
