add_executable(preprocess main.cc)
target_link_libraries(preprocess PRIVATE project_warnings
    PUBLIC project_options HighFive FastJet::FastJet range-v3)

add_executable(rotate rotate_events.cc)
target_link_libraries(rotate PRIVATE project_warnings
    PUBLIC project_options FastJet::FastJet range-v3 cxxopts)
