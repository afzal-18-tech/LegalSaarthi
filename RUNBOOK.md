# LegalSaarthi Runbook

## Phase 2 — Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify
python -m pytest tests/ -v
```

## Phase 3 — Metadata Acquisition

```bash
# Download metadata for all target years (2015-2025)
python scripts/acquire_metadata.py

# Download metadata for specific years
python scripts/acquire_metadata.py --year-start 2020 --year-end 2022
```

**Outputs**:
- `data/metadata/parquet/year=YYYY/metadata.parquet` — cached parquet files
- `data/metadata/raw_metadata.csv` — combined metadata

## Phase 4 — PDF Acquisition

### Staged Gates

```bash
# Single document test (reproduce/fix defect)
python scripts/acquire_pdfs.py --limit 1 --year-start 2020 --year-end 2020

# Smoke test: 10 documents
python scripts/acquire_pdfs.py --stage smoke

# Pilot test: 50 documents
python scripts/acquire_pdfs.py --stage pilot

# Prototype: 200 documents
python scripts/acquire_pdfs.py --stage prototype

# Main study: 1000 documents
python scripts/acquire_pdfs.py --stage main
```

### Gate Criteria

| Stage     | Target | Min Success Rate |
|-----------|--------|-----------------|
| Smoke     | 10     | 90%             |
| Pilot     | 50     | 95%             |
| Prototype | 200    | 95%             |
| Main      | 1000   | 95%             |

**Outputs**:
- `data/raw/year=YYYY/*.pdf` — downloaded PDFs
- `data/metadata/manifest_{stage}.csv` — acquisition manifest
- `data/metadata/download_failures_{stage}.csv` — failure log
- `data/metadata/acquisition_stats_{stage}.json` — statistics

## Data Source

- **Bucket**: `s3://indian-supreme-court-judgments`
- **Direct HTTP**: `https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/`
- **No credentials required** — public bucket
- **Rate limit**: 2 req/sec (configurable in `configs/default.yaml`)
- **License**: CC-BY-4.0

## Troubleshooting

### Download failures
Check `data/metadata/download_failures_{stage}.csv` for:
- `case_id` — which judgment failed
- `error` — HTTP status, connection error, etc.
- `attempts` — how many retries were made

### Unicode errors in logs
The logging system auto-configures UTF-8 on Windows. If issues persist,
set the environment variable `PYTHONIOENCODING=utf-8`.

### Changing year range
Edit `configs/default.yaml` → `scope.year_start` and `scope.year_end`,
or use `--year-start` and `--year-end` CLI flags.
