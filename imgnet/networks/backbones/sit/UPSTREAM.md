Source: https://github.com/willisma/SiT
Commit: cbde832a40b153ccc79603412409da9c9b0c568c

model.py is unmodified upstream code. adapter.py disables class dropout during TBSM training while preserving the pretrained null-class embedding, and exposes the repository backbone interface. Native SiT time runs noise to data.
