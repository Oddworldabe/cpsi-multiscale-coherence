# Data Sources for P2 Reproduction

All datasets are publicly available at no cost.

## Tier 1: Pillar Domains

| Domain | Timescale | Source | URL |
|---|---|---|---|
| Cardiology MIT-BIH | ~1 s | PhysioNet | https://physionet.org/content/mitdb/ |
| Cardiology Chapman | ~1 s | PhysioNet | https://physionet.org/content/ecg-arrhythmia/ |
| Sleep EEG | ~30 s | PhysioNet | https://physionet.org/content/sleep-edfx/ |
| Finance S&P 500 | ~1 day | Yahoo Finance | `pip install yfinance` |
| Music GTZAN | ~0.01 s | Marsyas | http://marsyas.info/downloads/datasets.html |
| Seismology | ~1 s | IRIS FDSN | https://service.iris.edu/ |

## Tier 2: Strong Domains

| Domain | Timescale | Source | URL |
|---|---|---|---|
| Agriculture NDVI | ~5 days | Copernicus/Planetary Computer | https://planetarycomputer.microsoft.com/ |
| Genomics GTEx | ~years | GTEx Portal | https://gtexportal.org/ |
| Gravitational waves | ~0.001 s | GWOSC | https://gwosc.org/ |
| Climate reanalysis | ~months | NOAA NCEP | https://psl.noaa.gov/ |
| Gaia stellar | ~Gyr | ESA Gaia | https://gea.esac.esa.int/ |

## Notes

- PhysioNet requires free registration
- Sentinel-2 via Planetary Computer: no account needed
- Financial data: `yfinance` package (Yahoo Finance API, free)
- GWOSC: open data, no registration
