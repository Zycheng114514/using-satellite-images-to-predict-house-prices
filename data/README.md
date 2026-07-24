# Data

> **The `data/` folder is not included in this repository** (it is large and can be
> regenerated from public sources). Download the prepared bundle and unzip it here:
>
> **📥 https://drive.google.com/drive/folders/158nALbBXwPRBzNIgveQrxfC1c4yMI3eQ?usp=drive_link**

After downloading, this folder should look like:

```
data/
├── raw/                                              # source data (DC Open Data)
│   ├── Common_Ownership_Lots.geojson                 # tax-lot polygon geometries
│   ├── Computer_Assisted_Mass_Appraisal_-_Residential.csv
│   ├── COMMERCIAL_(CAMA).csv
│   └── CONDOMINIUM_(CAMA).csv
├── processed/
│   └── merged_<year>.geojson                         # lots ⋈ qualified sales, per imagery year
├── images/
│   └── <year>/<ssl>.png                              # one aerial ortho chip per lot per year
├── embeddings/
│   ├── image_embeddings_<year>.npy                   # cached frozen-encoder embeddings
│   └── ssl_order_<year>.npy                          # SSL ordering aligned to the embeddings
├── dataset.zip                                        # zipped images/ (fast unzip during training)
└── pretrained_weights/                                # optional
    └── inception_v3.pth                               # local Inception v3 weights (else downloaded)
```

Imagery years used: **2015, 2017, 2019, 2021, 2023, 2025**.

## Regenerating from scratch

The `data/raw/` files come from the [DC Open Data portal](https://opendata.dc.gov/):

- [CAMA – Residential](https://opendata.dc.gov/datasets/DCGIS::computer-assisted-mass-appraisal-residential/about)
- [CAMA – Commercial](https://opendata.dc.gov/datasets/DCGIS::computer-assisted-mass-appraisal-commercial/about)
- [CAMA – Condominium](https://opendata.dc.gov/datasets/DCGIS::computer-assisted-mass-appraisal-condominium/about)
- [Common Ownership Lots](https://opendata.dc.gov/datasets/DCGIS::common-ownership-lots/about)
- [DC From Above (aerial orthophotography)](https://opendata.dc.gov/pages/dc-from-above)

With `data/raw/` in place, running [`src/data_preparation.py`](../src/data_preparation.py)
rebuilds `processed/`, downloads `images/` from the DC ArcGIS ImageServer, and writes `dataset.zip`.
