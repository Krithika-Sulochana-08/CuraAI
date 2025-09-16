import os
import sys
import json
import pandas as pd
import streamlit as st

# Try to import DataGuardian4 from local file (project root or same dir)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from data_guardian4 import DataGuardian4
except ModuleNotFoundError:
    # also try parent dir
    PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)
    from data_guardian4 import DataGuardian4
    # Make sure Python can see the project root (one level up from /app)

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# Import agents
from agents.risk_sentinel import RiskSentinel
from agents.report_maestro import ReportMaestro

st.set_page_config(page_title="CuraAI — Four-Input Validation", layout="wide")
st.title("CuraAI — Protocol + CRF + Rules + Patient Data → SDTM & ADaM")

# Sidebar Inputs
with st.sidebar:
    st.header("Study Inputs (Configs)")
    study_id = st.text_input("Study ID", "CUR-DEM-001")

    st.markdown("**1) Protocol config (.yaml)**")
    f_protocol = st.file_uploader("protocol_config.yaml", type=["yaml", "yml"])

    st.markdown("**2) CRF format (.yaml)**")
    f_crf = st.file_uploader("crf_format.yaml", type=["yaml", "yml"])

    st.markdown("**3) Sample rules (.yaml)**")
    f_rules = st.file_uploader("sample_rules.yaml", type=["yaml", "yml"])

    st.markdown("**4) AE corroboration map (.csv)**")
    f_aemap = st.file_uploader("ae_corroboration_map.csv", type=["csv"])

    st.divider()
    st.header("Patient datasets (CSV)")
    st.caption("Upload the raw SDTM-like domain files you have (DM, AE, VS, EX at minimum).")
    dm = st.file_uploader("DM.csv", type="csv")
    ae = st.file_uploader("AE.csv", type="csv")
    vs = st.file_uploader("VS.csv", type="csv")
    ex = st.file_uploader("EX.csv", type="csv")

    run = st.button("Run DataGuardian4")

# Directories for saving outputs
OUT_DIR = "./outputs_streamlit"
CONF_DIR = os.path.join(OUT_DIR, "configs")

# Function to save uploaded files
def save_bytes(file, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(file.read())

# Function to read CSV
def read_csv(path):
    import pandas as pd
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

# Run the workflow when the button is clicked
if run:
    # Ensure all four configs are uploaded
    if not (f_protocol and f_crf and f_rules and f_aemap):
        st.error("Please upload all four config files: protocol_config.yaml, crf_format.yaml, sample_rules.yaml, ae_corroboration_map.csv")
        st.stop()

    # Save configs
    os.makedirs(CONF_DIR, exist_ok=True)
    p_protocol = os.path.join(CONF_DIR, "protocol_config.yaml"); save_bytes(f_protocol, p_protocol)
    p_crf = os.path.join(CONF_DIR, "crf_format.yaml"); save_bytes(f_crf, p_crf)
    p_rules = os.path.join(CONF_DIR, "sample_rules.yaml"); save_bytes(f_rules, p_rules)
    p_aemap = os.path.join(CONF_DIR, "ae_corroboration_map.csv"); save_bytes(f_aemap, p_aemap)

    # Save patient CSVs
    inputs = {}
    for (name, fh) in [("DM", dm), ("AE", ae), ("VS", vs), ("EX", ex)]:
        if fh is not None:
            path = os.path.join(OUT_DIR, f"{name}.csv")
            save_bytes(fh, path)
            inputs[name] = path

    if not inputs:
        st.warning("Upload at least one patient dataset CSV.")
        st.stop()

    # Run DataGuardian4
    guardian = DataGuardian4(
        study_id=study_id,
        protocol_config_path=p_protocol,
        crf_format_path=p_crf,
        sample_rules_path=p_rules,
        ae_map_path=p_aemap
    )
    meta = guardian.run(patient_inputs=inputs, out_dir=OUT_DIR)

    # ---- UI output ----
    st.success("DataGuardian4 run completed.")
    st.subheader("Run Metadata")
    st.json(meta)

    # ---- Build SDTM dict from DataGuardian outputs ----
    sdtm = {}
    sd = os.path.join(OUT_DIR, "sdtm")
    for dom in ["DM", "AE", "VS", "EX"]:
        fp = os.path.join(sd, f"{dom}.csv")
        if os.path.exists(fp):
            sdtm[dom] = pd.read_csv(fp)

    # ---- Run RiskSentinel (ADaM) ----
    rs = RiskSentinel(out_dir=OUT_DIR)
    rs_meta = rs.run(sdtm)

    # ---- Display high-risk notifications ----
    st.subheader("Risk Notifications")
    notifications = rs_meta.get("notifications", [])
    if notifications:
        for note in notifications:
            st.warning(note)  # yellow warning box
    else:
        st.info("No notifications generated.")

    # ---- Display table of high-risk patients ----
    high_risk_patients = rs_meta.get("high_dropout_patients", [])
    if high_risk_patients:
        st.subheader("High Dropout Risk Patients")
        df_high = pd.DataFrame(high_risk_patients, columns=["USUBJID"])
        st.dataframe(df_high, use_container_width=True)
    if rs_meta.get("report_pdf_path"):
        st.success(f"PDF report generated: {rs_meta['report_pdf_path']}")
    else:
        st.warning("PDF report was not generated.")
        
    st.subheader("Risk Sentinel Metadata")
    st.json(rs_meta)


    # ---- Generate and Display the Report ----
    report_maestro = ReportMaestro(out_dir=OUT_DIR)
    html_path, report_pdf_path = report_maestro.generate_report(
        sdtm_dir=os.path.join(OUT_DIR, "sdtm"),
        adam_dir=os.path.join(OUT_DIR, "adam"),
        study_title="Clinical Study Report"
    )

    st.subheader("Download Final Report (PDF)")
    with open(report_pdf_path, "rb") as pdf_file:
        st.download_button(
            label="Download CSR Report (PDF)",
            data=pdf_file,
            file_name="CSR_report.pdf",
            mime="application/pdf",
            key="download_csr_pdf" 
        )

    sdtm_tab, adam_tab, queries_tab, report_tab = st.tabs(["SDTM-like Tables", "ADaM Tables", "Queries", "Run Report"])

    with sdtm_tab:
        st.caption("Validated SDTM-like datasets")
        sd = os.path.join(OUT_DIR, "sdtm")
        cols = st.columns(4)
        for i, dom in enumerate(["DM", "AE", "VS", "EX"]):
            fp = os.path.join(sd, f"{dom}.csv")
            if os.path.exists(fp):
                with cols[i % 4]:
                    st.markdown(f"**{dom}.csv**")
                    df = read_csv(fp)
                    st.dataframe(df, use_container_width=True, height=260)
                    st.download_button(f"Download {dom}.csv", open(fp,"rb"), file_name=f"{dom}.csv")

    with adam_tab:
        st.caption("Derived ADaM datasets")
        ad = os.path.join(OUT_DIR, "adam")
        for name in ["ADSL", "ADAE", "ADVS"]:
            fp = os.path.join(ad, f"{name}.csv")
            if os.path.exists(fp):
                st.markdown(f"**{name}.csv**")
                df = read_csv(fp)
                st.dataframe(df, use_container_width=True, height=260)
                st.download_button(f"Download {name}.csv", open(fp,"rb"), file_name=f"{name}.csv")

    with queries_tab:
        st.caption("Issue register (YAML)")
        qyaml = meta.get("queries_yaml")
        if qyaml and os.path.exists(qyaml):
            st.code(open(qyaml, "r", encoding="utf-8").read(), language="yaml")
            st.download_button("Download queries.yaml", open(qyaml, "rb"), file_name="queries.yaml")
        
        else:
            qfp = meta.get("queries_file")
            if qfp and os.path.exists(qfp):
                lines = open(qfp, "r", encoding="utf-8").read().splitlines()
                st.caption("Fallback: JSONL")
                st.code("\n".join(lines[:200]) + ("\n... (truncated)" if len(lines) > 200 else ""), language="json")
                st.download_button("Download queries.jsonl", open(qfp, "rb"), file_name="queries.jsonl")
                
            else:
                st.info("No queries generated.")

    with report_tab:
        st.caption("run_report.json")
        rfp = meta.get("report_file")
        if rfp and os.path.exists(rfp):
            st.code(open(rfp, "r", encoding="utf-8").read(), language="json")
            st.download_button("Download run_report.json", open(rfp, "rb"), file_name="run_report.json")
else:
    st.info("Upload the four configs (protocol, CRF, rules, AE map) and your patient CSVs, then click **Run DataGuardian4**.")
