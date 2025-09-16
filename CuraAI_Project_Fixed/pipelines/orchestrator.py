
import os, pandas as pd
from agents.data_guardian import DataGuardian
from agents.risk_sentinel import RiskSentinel
from agents.report_maestro import ReportMaestro

def run_end_to_end(study_id, inputs, out_dir="./outputs"):
    dg = DataGuardian(study_id)
    meta = dg.run_pipeline(inputs, out_dir=out_dir)

    sdtm = {}
    for fn in os.listdir(meta["sdtm_dir"]):
        if fn.endswith(".csv"):
            dom = fn.replace(".csv","").upper()
            sdtm[dom] = pd.read_csv(os.path.join(meta["sdtm_dir"], fn))

    rs = RiskSentinel(out_dir=out_dir)
    rs_meta = rs.run(sdtm)

    rm = ReportMaestro(out_dir=out_dir)
    csr_meta = rm.run(rs_meta["adam_dir"])

    return {**meta, **rs_meta, **csr_meta}
