
# DataGuardian4 — four-input engine

**Inputs required**
1. `protocol_config.yaml`
2. `crf_format.yaml`
3. `sample_rules.yaml` (optionally uses `ae_corroboration_map.csv`)
4. Patient datasets dict (DM/AE/VS/EX CSV paths)

**Outputs**
- SDTM-like: `<out>/sdtm/*.csv`
- ADaM: `<out>/adam/ADSL.csv`, `ADAE.csv`, `ADVS.csv`
- Queries: `<out>/queries/queries.jsonl`
- Run report: `<out>/run_report.json`

**Example**
```python
from data_guardian4 import DataGuardian4

guardian = DataGuardian4(
    study_id="CUR-DEM-001",
    protocol_config_path="protocol_config.yaml",
    crf_format_path="crf_format.yaml",
    sample_rules_path="sample_rules.yaml",
    ae_map_path="ae_corroboration_map.csv",
)

meta = guardian.run(
    patient_inputs={
        "DM": "tests/sample_data/DM.csv",
        "AE": "tests/sample_data/AE.csv",
        "VS": "tests/sample_data/VS.csv",
        "EX": "tests/sample_data/EX.csv",
    },
    out_dir="./outputs_demo"
)
print(meta)
```
