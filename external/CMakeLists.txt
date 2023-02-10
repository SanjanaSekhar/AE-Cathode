CPMAddPackage("gh:ericniebler/range-v3#0.12.0")
if(ENABLE_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    CPMAddPackage(
        NAME nanobind
        GITHUB_REPOSITORY wjakob/nanobind
        VERSION 0.1.0
    )
endif()
