import fastjet
import pandas as pd
import awkward as ak
import numpy as np


def to_mom(array):
    px = array[:, :, 0]*np.cos(array[:, :, 1])
    py = array[:, :, 0]*np.sin(array[:, :, 1])
    pz = array[:, :, 0]*np.sinh(array[:, :, 2])
    e = array[:, :, 0]*np.cosh(array[:, :, 2])
    mom = ak.zip({"E": e, "px": px, "py": py, "pz": pz})
    return mom


def main():
    nevents = 10
    data = pd.read_hdf('events_LHCO2020_BlackBox1.h5',start=0,stop=nevents)
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
    particles = data.to_numpy().reshape(nevents, 700, 3)
    particles = to_mom(particles)
    cluster = fastjet.ClusterSequence(particles, jetdef)
    jets = cluster.inclusive_jets()
    tmpsort = ak.argsort(jets.E, axis=-1, ascending=False)
    jets = jets[tmpsort][:, 0:2]
    constituents = ak.flatten(cluster.constituents()[tmpsort][:, 0:2])
    for constituent in constituents:
        print(constituent)
    # constituents = ak.concatenate(ak.unzip(constituents), axis=-1)
    print(constituents)
    # print(ak.to_dataframe(ak.fill_none(ak.pad_none(ak.flatten(constituents, axis=-1), 142, clip=True, axis=-1), 0, axis=-1)))


if __name__ == '__main__':
    main()
