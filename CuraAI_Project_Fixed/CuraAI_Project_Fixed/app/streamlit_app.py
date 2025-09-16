
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
from agents.data_guardian import DataGuardian
from agents.risk_sentinel import RiskSentinel
from agents.report_maestro import ReportMaestro

st.set_page_config(page_title="CuraAI Demo (Fixed)", layout="wide")
st.title("CuraAI — End-to-End Clinical Trial Agents")

with st.sidebar:
    st.header("Inputs")
    study_id = st.text_input("Study ID", "CUR-DEM-001")
    dm = st.file_uploader("DM.csv", type="csv")
    ae = st.file_uploader("AE.csv", type="csv")
    vs = st.file_uploader("VS.csv", type="csv")
    ex = st.file_uploader("EX.csv", type="csv")
    run = st.button("Run Pipeline")

if run:
    out_dir = "./outputs_streamlit"
    os.makedirs(out_dir, exist_ok=True)
    inputs = {}
    for name, fh in [("DM", dm), ("AE", ae), ("VS", vs), ("EX", ex)]:
        if fh is not None:
            path = os.path.join(out_dir, f"{name}.csv")
            with open(path, "wb") as f:
                f.write(fh.read())
            inputs[name] = path

    dg = DataGuardian(study_id)
    meta = dg.run_pipeline(inputs, out_dir=out_dir)
    st.subheader("DataGuardian Outputs"); st.json(meta)

    sdtm = {}
    for dom in ["DM","AE","VS","EX"]:
        fp = os.path.join(out_dir, "sdtm", f"{dom}.csv")
        if os.path.exists(fp): sdtm[dom] = pd.read_csv(fp)

    rs = RiskSentinel(out_dir=out_dir)
    rs_meta = rs.run(sdtm)
    st.subheader("RiskSentinel Outputs"); st.json(rs_meta)

    rm = ReportMaestro(out_dir=out_dir)
    csr_meta = rm.run(rs_meta["adam_dir"])
    st.subheader("ReportMaestro Outputs"); st.json(csr_meta)
else:
    st.info("Upload SDTM-like CSVs and click **Run Pipeline**.")
