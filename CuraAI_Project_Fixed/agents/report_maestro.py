import os
import json
import time
import requests
import toml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
from weasyprint import HTML

# ----------------------------- Utilities -----------------------------

def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None

def _fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "0 (0%)"
    return f"{num} ({num/den*100:.1f}%)"

def _maybe(val, default="-"):
    return default if pd.isna(val) else val

# --------------------------- ReportMaestro ---------------------------

class ReportMaestro:
    """
    Build a complete, styled CSR from SDTM & ADaM tables and (optionally) LLM.
    - Reads CSVs from sdtm_dir and adam_dir
    - Computes summaries with pandas
    - Calls LLM per-section with rich, structured prompts (or falls back to template prose)
    - Writes HTML + PDF
    """

    def __init__(
        self,
        out_dir: str = "./outputs",
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        temperature: float = 0.4,
        per_section_tokens: int = 1200,
        request_timeout: int = 120
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.model = model
        self.temperature = temperature
        self.per_section_tokens = per_section_tokens
        self.request_timeout = request_timeout

        # Load OpenRouter API key (Claude/OpenAI via OpenRouter)
        self.api_key = self._load_api_key()

    # ----------------------- Secrets / API key -----------------------

    def _load_api_key(self) -> Optional[str]:
        """
        Load OpenRouter API key from Streamlit secrets first, then .streamlit/secrets.toml.
        Returns None if not found (offline fallback will be used).
        """
        # Streamlit
        try:
            k = st.secrets.get("openrouter", {}).get("api_key") or st.secrets.get("claude_api", {}).get("api_key")
            if k:
                print("Loaded API key from Streamlit secrets.")
                return k
        except Exception:
            pass

        # Local TOML
        for sect in ("openrouter", "claude_api"):
            p = Path("./.streamlit/secrets.toml")
            if p.exists():
                t = toml.load(p)
                container = t.get(sect, {})
                k = container.get("api_key")
                if k:
                    print(f"Loaded API key from local secrets.toml section [{sect}].")
                    return k

        print("No API key found. Will use offline (template) generation.")
        return None

    # ------------------------- Data Loading --------------------------

    def load_sdtm_adam(
        self,
        sdtm_dir: str,
        adam_dir: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Load commonly used SDTM/ADaM tables if present.
        """
        sdtm = Path(sdtm_dir)
        adam = Path(adam_dir)
        tables = {
            # ADaM (preferred for reporting)
            "ADSL": _read_csv_if_exists(adam / "ADSL.csv"),
            "ADAE": _read_csv_if_exists(adam / "ADAE.csv"),
            "ADVS": _read_csv_if_exists(adam / "ADVS.csv"),
            "ADEX": _read_csv_if_exists(adam / "ADEX.csv"),
            "ADLB": _read_csv_if_exists(adam / "ADLB.csv"),
            "ADPR": _read_csv_if_exists(adam / "ADPR.csv"),

            # SDTM (fallbacks/secondary)
            "DM":   _read_csv_if_exists(sdtm / "DM.csv"),
            "AE":   _read_csv_if_exists(sdtm / "AE.csv"),
            "VS":   _read_csv_if_exists(sdtm / "VS.csv"),
            "EX":   _read_csv_if_exists(sdtm / "EX.csv"),
            "LB":   _read_csv_if_exists(sdtm / "LB.csv"),
        }
        return tables

    # ---------------------- Derived Summaries ------------------------

    def derive_core_summaries(self, t: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, dict]:
        """
        Compute high-level summaries used to condition LLM prompts and to render tables.
        Robust to missing tables/columns.
        """
        summaries = {}

        # ---------- Population / Disposition (ADSL preferred) ----------
        adsl = t.get("ADSL")
        pop = {"n_randomized": 0, "n_treated": 0, "n_completed": 0, "n_discontinued": 0,
               "sex": {}, "race": {}, "age_mean": None, "age_sd": None, "arms": {}}

        if adsl is not None and len(adsl) > 0:
            pop["n_randomized"] = len(adsl)
            # typical flags/columns (robust getters)
            trt_flag_cols = [c for c in adsl.columns if c.upper() in ("TRTFL", "SAFFL", "FASFL")]
            if trt_flag_cols:
                treated = adsl[adsl[trt_flag_cols[0]].astype(str).str.upper().isin(["Y", "1", "TRUE"])]
                pop["n_treated"] = len(treated)
            else:
                pop["n_treated"] = len(adsl)

            comp_cols = [c for c in adsl.columns if c.upper().startswith("COMP")]
            if comp_cols:
                completed = adsl[adsl[comp_cols[0]].astype(str).str.upper().isin(["Y", "1", "TRUE"])]
                pop["n_completed"] = len(completed)

            disc_cols = [c for c in adsl.columns if "DISC" in c.upper()]
            if disc_cols:
                discontinued = adsl[adsl[disc_cols[0]].astype(str).str.upper().isin(["Y", "1", "TRUE"])]
                pop["n_discontinued"] = len(discontinued)

            # arms
            for col in ("ARM", "ACTARM", "TRT01A"):
                if col in adsl.columns:
                    pop["arms"] = adsl[col].value_counts(dropna=False).to_dict()
                    break

            # sex/race
            if "SEX" in adsl.columns:
                pop["sex"] = adsl["SEX"].value_counts(dropna=False).to_dict()
            if "RACE" in adsl.columns:
                pop["race"] = adsl["RACE"].value_counts(dropna=False).to_dict()

            # age
            age_col = None
            for c in ("AGE", "AGEYRS", "AGE_Y"):
                if c in adsl.columns:
                    age_col = c; break
            if age_col:
                pop["age_mean"] = round(float(adsl[age_col].dropna().mean()), 1) if adsl[age_col].notna().any() else None
                pop["age_sd"] = round(float(adsl[age_col].dropna().std()), 1) if adsl[age_col].notna().any() else None

        summaries["population"] = pop

        # ---------- Adverse Events (ADAE preferred) ----------
        adae = t.get("ADAE")
        ae_summary = {"n_subjects_with_ae": 0, "n_ae": 0, "n_sev": 0, "n_serious": 0, "top_pt": []}
        if adae is not None and len(adae) > 0:
            ae_summary["n_ae"] = len(adae)
            subj_col = None
            for c in ("USUBJID", "SUBJID", "ID"):
                if c in adae.columns:
                    subj_col = c; break
            if subj_col:
                ae_summary["n_subjects_with_ae"] = adae[subj_col].nunique()

            # severity/severity grading
            sev_col = None
            for c in adae.columns:
                if "SEV" == c.upper() or "AESEV" == c.upper() or "ASEV" == c.upper():
                    sev_col = c; break
            if sev_col:
                ae_summary["n_sev"] = int((adae[sev_col].astype(str).str.upper().isin(["SEVERE", "GRADE 3", "GRADE 4", "GRADE3", "GRADE4"])).sum())

            # seriousness
            ser_col = None
            for c in ("AESER", "SERIOUS", "SER"):
                if c in adae.columns:
                    ser_col = c; break
            if ser_col:
                ae_summary["n_serious"] = int(adae[ser_col].astype(str).str.upper().isin(["Y", "1", "TRUE"]).sum())

            # top preferred terms
            pt_col = None
            for c in ("AETERM", "AEDECOD", "PT", "PREFTERM"):
                if c in adae.columns:
                    pt_col = c; break
            if pt_col:
                ae_summary["top_pt"] = adae[pt_col].value_counts().head(10).to_dict()

        summaries["ae"] = ae_summary

        # ---------- Vital Signs (ADVS preferred) ----------
        advs = t.get("ADVS")
        vs_summary = {}
        if advs is not None and len(advs) > 0:
            # attempt PARAM/PARAMCD with AVAL
            if "PARAM" in advs.columns and "AVAL" in advs.columns:
                vs_summary = {
                    "n_records": len(advs),
                    "by_param_mean": advs.groupby("PARAM", dropna=False)["AVAL"].mean().round(2).to_dict()
                }
        summaries["vitals"] = vs_summary

        # ---------- Exposure ----------
        adex = t.get("ADEX") or t.get("EX")
        exposure = {}
        if adex is not None and len(adex) > 0:
            amt_col = None
            for c in ("ADOSE", "EXDOSE", "DOSE", "AMT"):
                if c in adex.columns:
                    amt_col = c; break
            if amt_col:
                exposure["mean_dose"] = round(float(adex[amt_col].dropna().mean()), 2) if adex[amt_col].notna().any() else None
        summaries["exposure"] = exposure

        return summaries

    # ---------------------------- LLM -------------------------------

    def _llm(self, prompt: str, max_tokens: Optional[int] = None) -> Optional[str]:
        """
        Call OpenRouter. Returns text or None on error; we keep going with fallback prose.
        """
        if not self.api_key:
            return None

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a senior medical writer drafting ICH E3-compliant Clinical Study Reports (CSRs). Use clear, neutral, audit-ready language. Do not hallucinate numbers; use only provided summaries."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens or self.per_section_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            # OpenRouter returns choices[0].message.content
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            # Some providers return list-of-parts; coerce to string
            if isinstance(content, list):
                content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            return content.strip() if content else None
        except Exception as e:
            print("LLM call failed:", repr(e))
            try:
                print("Response:", resp.text[:500])  # best effort
            except Exception:
                pass
            return None

    # ---------------------- Section Generators ----------------------

    def _title_page(self, study_title: str) -> str:
        return f"""
        <section id="title-page">
          <h1>Clinical Study Report</h1>
          <h2>{study_title}</h2>
          <p><strong>Generated on:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        </section>
        """

    def _toc(self) -> str:
        items = [
            ("synopsis", "Synopsis"),
            ("ethics", "Ethics"),
            ("investigators", "Investigators and Study Structure"),
            ("introduction", "Introduction"),
            ("objectives", "Study Objectives"),
            ("plan", "Investigational Plan"),
            ("efficacy-safety", "Efficacy and Safety Evaluation"),
            ("patients", "Study Patients"),
            ("results", "Results"),
            ("safety", "Safety Data"),
            ("discussion", "Discussion and Conclusions"),
        ]
        lis = "\n".join([f'<li><a href="#{i}">{t}</a></li>' for i, t in items])
        return f"""
        <section id="toc">
          <h2>Table of Contents</h2>
          <ol>{lis}</ol>
        </section>
        """

    def _offline_or_llm(self, heading: str, prompt: str, fallback_paragraphs: List[str]) -> str:
        text = self._llm(prompt)
        if not text:
            text = "\n\n".join(fallback_paragraphs)
        return f'<section id="{heading}"><h2>{heading.replace("-", " ").title()}</h2>\n<p>{text}</p></section>'

    # --------------------------- HTML/CSS ---------------------------

    def _wrap_html(self, body_html: str, study_title: str) -> str:
        css = """
        <style>
        body { font-family: 'Times New Roman', serif; line-height: 1.35; margin: 40px; }
        h1 { font-size: 30px; margin-bottom: 0; }
        h2 { margin-top: 28px; font-size: 22px; }
        h3 { font-size: 18px; }
        ol { padding-left: 18px; }
        table { border-collapse: collapse; margin: 12px 0; width: 100%; }
        th, td { border: 1px solid #555; padding: 6px 8px; font-size: 12.5px; }
        th { background: #f0f0f0; }
        .small { font-size: 12px; color: #333; }
        .kpi { display:inline-block; border:1px solid #888; padding:8px 10px; margin:4px 6px; border-radius:6px; }
        a { text-decoration: none; color: #0645ad; }
        #title-page p { margin-top: 4px; }
        #toc ol li { margin: 4px 0; }
        </style>
        """
        return f"<!doctype html><html><head><meta charset='utf-8'><title>{study_title} — CSR</title>{css}</head><body>{body_html}</body></html>"

    # --------------------------- Tables -----------------------------

    def _html_table(self, df: pd.DataFrame, caption: str) -> str:
        if df is None or df.empty:
            return ""
        return f"<h3 class='small'>{caption}</h3>{df.to_html(index=False, border=0)}"

    # ------------------------ Report Generation ---------------------

    def generate_report(
        self,
        sdtm_dir: str,
        adam_dir: str,
        study_title: str,
        save_pdf: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Build and save the full CSR. Returns (html_path, pdf_path or None).
        """
        # 1) Load & summarize data
        tables = self.load_sdtm_adam(sdtm_dir, adam_dir)
        summaries = self.derive_core_summaries(tables)

        # Build small display tables for the PDF (optional, robust to missing)
        adsl = tables.get("ADSL")
        adae = tables.get("ADAE")
        advs = tables.get("ADVS")

        # Example presentation tables
        demo_tbl = None
        if adsl is not None and not adsl.empty:
            cols = [c for c in adsl.columns if c.upper() in ("USUBJID","SUBJID","SEX","AGE","RACE","ARM","ACTARM")]
            demo_tbl = adsl[cols].head(20).copy() if cols else None

        ae_top_tbl = None
        if adae is not None and not adae.empty:
            pt_col = None
            for c in ("AEDECOD","AETERM","PT"):
                if c in adae.columns:
                    pt_col = c; break
            if pt_col:
                top = (adae[pt_col].value_counts().head(10).reset_index())
                top.columns = ["Preferred Term", "Count"]
                ae_top_tbl = top

        vs_mean_tbl = None
        if advs is not None and not advs.empty and "PARAM" in advs.columns and "AVAL" in advs.columns:
            vs_mean_tbl = advs.groupby("PARAM", dropna=False)["AVAL"].mean().round(2).reset_index()
            vs_mean_tbl.columns = ["Parameter", "Mean AVAL"]

        # 2) Compose prompts (grounded with summaries)
        pop = summaries["population"]
        ae = summaries["ae"]
        vit = summaries["vitals"]
        exp = summaries["exposure"]
        
        
        synopsis_prompt = f"""
        
Draft a concise ICH E3-style Synopsis for the Clinical Study Report using ONLY the following derived facts.
Avoid inventing numbers; if a specific value is missing, speak qualitatively.

Population:
- Randomized: {pop['n_randomized']}, Treated: {pop['n_treated']}, Completed: {pop['n_completed']}, Discontinued: {pop['n_discontinued']}
- Arms: {json.dumps(pop['arms'], ensure_ascii=False)}
- Sex: {json.dumps(pop['sex'], ensure_ascii=False)}, Race: {json.dumps(pop['race'], ensure_ascii=False)}
- Age: mean={pop['age_mean']}, sd={pop['age_sd']}

Safety:
- AE records: {ae['n_ae']}, subjects with ≥1 AE: {ae['n_subjects_with_ae']}, severe AEs: {ae['n_sev']}, serious AEs: {ae['n_serious']}
- Top PTs: {json.dumps(ae['top_pt'], ensure_ascii=False)}

Vitals summary:
- {json.dumps(vit, ensure_ascii=False)}

Exposure:
- {json.dumps(exp, ensure_ascii=False)}

Write ~180–250 words.
""".strip()

        ethics_prompt = (
            "Write an Ethics section covering IRB/IEC approval, informed consent, "
            "patient confidentiality, and adherence to GCP/ICH E6(R2). Keep it audit-ready and generic."
        )

        investigators_prompt = (
            "Provide a brief description of Investigators, Study Personnel, Sites, and oversight "
            "(DSMB if applicable). No names—structure only."
        )

        intro_prompt = (
            "Write an Introduction: background disease context, mechanism of intervention, "
            "rationale for study, and brief literature context."
        )

        objectives_prompt = (
            "Detail the Study Objectives: primary, secondary, and exploratory endpoints, clearly enumerated."
        )

        plan_prompt = (
            "Write an Investigational Plan: design (randomized/double-blind etc.), population, "
            "inclusion/exclusion (generic), randomization, blinding, dosing regimen, visit schedule overview, "
            "protocol deviations handling."
        )

        eff_saf_prompt = f"""
Efficacy and Safety Evaluation: Describe analysis populations (ITT/PP/Safety), statistical methods at a high level,
multiplicity (if any), and safety monitoring. Use the derived data: AE subjects={ae['n_subjects_with_ae']},
severe={ae['n_sev']}, serious={ae['n_serious']}. Avoid fabricating p-values or effect sizes.
""".strip()

        patients_prompt = f"""
Study Patients: Summarize disposition using the counts: randomized={pop['n_randomized']},
treated={pop['n_treated']}, completed={pop['n_completed']}, discontinued={pop['n_discontinued']}.
Briefly describe demographics (sex/race breakdown available), and protocol deviations handling generically.
""".strip()

        results_prompt = f"""
Results: Summarize high-level safety and descriptive outcomes ONLY from supplied summaries.
- Adverse events: total records={ae['n_ae']}, subjects with AE={ae['n_subjects_with_ae']}, severe={ae['n_sev']},
  serious={ae['n_serious']}, top PTs={json.dumps(ae['top_pt'], ensure_ascii=False)}.
- Vitals: means by parameter may be available.
- Exposure: mean dose if available = {exp.get('mean_dose')}.
Do NOT invent efficacy results; stay qualitative unless numbers are present. Write ~250–350 words.
""".strip()

        safety_prompt = (
            "Safety Data: Provide a structured narrative of TEAEs, severity, seriousness, relation to study drug, "
            "discontinuations due to AEs, and deaths (if any). Be generic/qualitative if numbers are missing."
        )

        discussion_prompt = (
            "Discussion & Conclusions: Interpret the study at a high level, comment on safety profile, "
            "limitations of the dataset, and recommend next steps. No invented statistics."
        )




        # 3) Build HTML sections (LLM or offline fallback)
        html_sections = [
            self._title_page(study_title),
            self._toc(),
            self._offline_or_llm("synopsis", synopsis_prompt, [
                "This study synopsis summarizes population, conduct, and key outcomes based on derived ADaM/SDTM tables. Randomized and treated participant counts, disposition, and safety findings are described qualitatively. Where numerical values are unavailable, neutral language is used to avoid over-interpretation."
            ]),
            self._offline_or_llm("ethics", ethics_prompt, [
                "The trial adhered to ICH-GCP, with prior approval by an Institutional Review Board/Independent Ethics Committee. Written informed consent was obtained from all subjects before any study procedures. Confidentiality and data protection were maintained per regulatory requirements."
            ]),
            self._offline_or_llm("investigators", investigators_prompt, [
                "The study was conducted at qualified clinical sites by experienced investigators and coordinators. Oversight included routine monitoring and quality control procedures. A data safety monitoring process ensured ongoing review of safety information."
            ]),
            self._offline_or_llm("introduction", intro_prompt, [
                "This Clinical Study Report describes a therapeutic investigation conducted to evaluate the safety and potential efficacy of the investigational product in the target disease area. The rationale is supported by prior nonclinical and early clinical evidence."
            ]),
            self._offline_or_llm("objectives", objectives_prompt, [
                "<strong>Primary:</strong> Evaluate safety and tolerability.<br><strong>Secondary:</strong> Characterize pharmacodynamics and supportive endpoints.<br><strong>Exploratory:</strong> Generate hypotheses for future studies."
            ]),
            self._offline_or_llm("plan", plan_prompt, [
                "This was a randomized, controlled study with prespecified eligibility criteria, a blinded treatment allocation, and scheduled assessments. Protocol deviations were prospectively defined and handled per SOPs."
            ]),
            self._offline_or_llm("efficacy-safety", eff_saf_prompt, [
                "Efficacy and safety evaluations were conducted in predefined analysis sets. Descriptive statistics summarized outcomes; safety monitoring captured TEAEs and serious events. No inferential testing is presented where data are unavailable."
            ]),
            self._offline_or_llm("patients", patients_prompt, [
                f"Disposition: randomized={pop['n_randomized']}, treated={pop['n_treated']}, completed={pop['n_completed']}, discontinued={pop['n_discontinued']}. Demographics and protocol deviations are summarized descriptively."
            ]),
        ]

        # 4) Results section with embedded small tables
        results_html = self._offline_or_llm("results", results_prompt, [
            "Results are summarized qualitatively based on available ADaM/SDTM extracts. Safety findings include counts of adverse events and their characteristics. Vital sign trends and exposure are described descriptively."
        ])
        # Append small tables under Results if present
        tables_html = ""
        if demo_tbl is not None:
            tables_html += self._html_table(demo_tbl, "Table: Sample Demographics (ADSL excerpt)")
        if ae_top_tbl is not None:
            tables_html += self._html_table(ae_top_tbl, "Table: Top 10 Adverse Event Preferred Terms (ADAE)")
        if vs_mean_tbl is not None:
            tables_html += self._html_table(vs_mean_tbl, "Table: Mean Vital Signs by Parameter (ADVS)")
        results_html = results_html.replace("</section>", tables_html + "</section>")
        html_sections.append(results_html)

        # 5) Remaining narrative sections
        html_sections.append(self._offline_or_llm("safety", safety_prompt, [
            "Overall, treatment-emergent adverse events were monitored and summarized by severity and seriousness. No causal inferences are made in the absence of comparative statistics."
        ]))
        html_sections.append(self._offline_or_llm("discussion", discussion_prompt, [
            "In this dataset, the investigational product showed a safety profile consistent with expectations. Limitations include incomplete or descriptive-only outputs. Further controlled studies are recommended."
        ]))

        # 6) Assemble & write
        body_html = "\n".join(html_sections)
        html_str = self._wrap_html(body_html, study_title)

        html_path = os.path.join(self.out_dir, "CSR_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

        pdf_path = None
        if save_pdf:
            pdf_path = os.path.join(self.out_dir, "CSR_report.pdf")
            HTML(string=html_str).write_pdf(pdf_path)

        return html_path, pdf_path
