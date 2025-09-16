
import os, json, pandas as pd

class DataGuardian:
    def __init__(self, study_id, rules_yaml=None, mapping_csv=None):
        self.study_id = study_id
        self.rules_yaml = rules_yaml
        self.mapping_csv = mapping_csv

    def run_pipeline(self, inputs, out_dir="./outputs"):
        os.makedirs(out_dir, exist_ok=True)
        sdtm_dir = os.path.join(out_dir, "sdtm")
        os.makedirs(sdtm_dir, exist_ok=True)
        for dom, path in inputs.items():
            df = pd.read_csv(path)
            df.to_csv(os.path.join(sdtm_dir, f"{dom}.csv"), index=False)
        report = {"study_id": self.study_id, "domains": list(inputs.keys())}
        with open(os.path.join(out_dir, "run_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return {"sdtm_dir": sdtm_dir, "report_file": os.path.join(out_dir, "run_report.json")}
