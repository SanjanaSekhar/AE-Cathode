add_executable(preprocess main.cc)
target_link_libraries(preprocess PRIVATE project_warnings
    PUBLIC project_options HighFive FastJet::FastJet range-v3)
