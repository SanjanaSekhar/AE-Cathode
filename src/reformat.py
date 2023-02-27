import pandas as pd

reader = pd.read_csv("events.csv", header=None, names=list(range(478)),iterator=True, chunksize=1000000)

for i, chunk in enumerate(reader):
    print("Reading chunk %i"%i)
    chunk.fillna(0).to_hdf("events_LHCO2020_BlackBox1_preprocessed.h5", "events", mode='a', complib="bzip2")
