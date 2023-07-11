# AE-Cathode
In this project we train an Autoencoder on the LHC Olympics dataset to provide improved input variables to CATHODE

## Preprocessing

The data is preprocessed using FastJet + RecursiveSoftDrop. The data can be preprocessed through the following steps:
1. Ensure you have the needed dependencies ([FastJet + fj-contrib](https://fastjet.fr/), [HighFive](https://github.com/BlueBrain/HighFive), and [hdf5-blosc](https://github.com/Blosc/hdf5-blosc))

   -- FastJet + fj-contrib: Follow the instructions [here](https://fastjet.fr/quickstart.html)
   
   -- HighFive: Follow the instructions [here](https://github.com/BlueBrain/HighFive#regular-installation-of-highfive)

   -- hdf5-blosc: First clone the repository recursively and then follow the instructions [here](https://github.com/Blosc/hdf5-blosc#compiling)
3. For hdf5-blosc, follow the directions on their README to ensure it is installed for hdf5 to read (this is needed to read the hdf5 files correctly)
4. Run `cmake -S . -B build -DFASTJET_ROOT_DIR=/path/to/fastjet/installation` 
5. Run `cmake --build build`
6. Run the preprocess script from the build directory, *i.e.* `cd build && ./src/preprocess`
