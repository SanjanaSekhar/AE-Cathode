# AE-Cathode
In this project we train an Autoencoder on the LHC Olympics dataset to provide improved input variables to CATHODE

## Preprocessing

The data is preprocessed using FastJet + RecursiveSoftDrop. The data can be preprocessed through the following steps:
1. Ensure you have the needed dependencies ([FastJet + fj-contrib](https://fastjet.fr/) and [HighFive](https://github.com/BlueBrain/HighFive))
2. Run `cmake -S . -B build -DFASTJET_ROOT_DIR=/path/to/fastjet/installation` 
3. Run `cmake --build build`
4. Run the preprocess script from the build directory, *i.e.* `cd build && ./src/preprocess`
