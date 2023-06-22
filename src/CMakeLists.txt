add_executable(preprocess main.cc)
target_link_libraries(preprocess PRIVATE project_warnings
    PUBLIC project_options HighFive FastJet::FastJet range-v3)

add_executable(rotate rotate_events.cc)
target_link_libraries(rotate PRIVATE project_warnings
    PUBLIC project_options FastJet::FastJet range-v3 cxxopts)

add_executable(check check.cc)
target_link_libraries(check PRIVATE project_warnings
    PUBLIC project_options FastJet::FastJet range-v3 cxxopts)

add_executable(jet_pad same_length.cc)
target_link_libraries(jet_pad PRIVATE project_warnings
    PUBLIC project_options FastJet::FastJet range-v3 cxxopts)

add_executable(plot_events plot_events.cc)
target_link_libraries(plot_events PRIVATE project_warnings
    PUBLIC project_options FastJet::FastJet range-v3 cxxopts)
