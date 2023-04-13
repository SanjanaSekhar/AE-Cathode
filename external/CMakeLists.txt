CPMAddPackage("gh:ericniebler/range-v3#0.12.0")

CPMAddPackage(
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 3.1.1
  OPTIONS "CXXOPTS_BUILD_EXAMPLES NO" "CXXOPTS_BUILD_TESTS NO" "CXXOPTS_ENABLE_INSTALL YES"
)

if(ENABLE_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    CPMAddPackage(
        NAME nanobind
        GITHUB_REPOSITORY wjakob/nanobind
        VERSION 0.1.0
    )
endif()
