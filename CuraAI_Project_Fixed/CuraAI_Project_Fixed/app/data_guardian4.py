"""
DataGuardian4 — four-input clinical validation & derivation engine
------------------------------------------------------------------
Inputs (all required):
1) protocol_config.yaml
2) crf_format.yaml
3) sample_rules.yaml            (may reference ae_corroboration_map.csv)
4) patient datasets (dict of domain->CSV paths, e.g., DM/AE/VS/EX)

Outputs:
- SDTM-like tables (CSV) under <out_dir>/sdtm/
- ADaM datasets (CSV)  under <out_dir>/adam/   [ADSL, ADAE, ADVS minimal]
- Queries (YAML)       <out_dir>/queries/queries.yaml   (+ optional JSONL)
- Run report JSON      <out_dir>/run_report.json
"""

from __future__ import annotations
import os, json, yaml, hashlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd


def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class Query:
    study_id: str
    domain: str
    rule_id: str
    severity: str
    message: str
    usubjid: Optional[str] = None
    record_loc: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    created_at: str = ""
    qid: str = ""

    def finalize(self):
        self.created_at = self.created_at or iso_now()
        payload = {k: v for k, v in asdict(self).items() if k not in ["qid"]}
        self.qid = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self


class DataGuardian4:
    def __init__(
        self,
        study_id: str,
        protocol_config_path: str,
        crf_format_path: str,
        sample_rules_path: str,
        ae_map_path: Optional[str] = None,
    ):
        self.study_id = study_id
        self.protocol = yaml.safe_load(open(protocol_config_path, "r", encoding="utf-8"))
        self.crf = yaml.safe_load(open(crf_format_path, "r", encoding="utf-8"))
        self.rules = yaml.safe_load(open(sample_rules_path, "r", encoding="utf-8"))
        self.ae_map = (
            pd.read_csv(ae_map_path) if ae_map_path and os.path.exists(ae_map_path) else pd.DataFrame()
        )

    # -------------------- public API --------------------
    def run(self, patient_inputs: Dict[str, str], out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        sdtm_dir = os.path.join(out_dir, "sdtm")
        queries_dir = os.path.join(out_dir, "queries")
        adam_dir = os.path.join(out_dir, "adam")
        os.makedirs(sdtm_dir, exist_ok=True)
        os.makedirs(queries_dir, exist_ok=True)
        os.makedirs(adam_dir, exist_ok=True)

        # Load patient datasets
        frames: Dict[str, pd.DataFrame] = {}
        for dom, path in patient_inputs.items():
            if path and os.path.exists(path):
                frames[dom.upper()] = pd.read_csv(path)

        # Validate & collect issues
        issues: List[Query] = []
        issues += self._validate_vs(frames.get("VS"))
        issues += self._validate_ae(frames.get("AE"), frames.get("VS"))
        issues += self._validate_temporal(frames.get("EX"), frames.get("VS"))

        # SDTM-like pass-through
        for dom, df in frames.items():
            df.to_csv(os.path.join(sdtm_dir, f"{dom}.csv"), index=False)

        # --- Run report (write FIRST so rpath always exists) ---
        by_sev_counts: Dict[str, int] = {}
        for q in issues:
            by_sev_counts[q.severity] = by_sev_counts.get(q.severity, 0) + 1

        rpath = os.path.join(out_dir, "run_report.json")
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "study_id": self.study_id,
                    "generated_at": iso_now(),
                    "domains": list(frames.keys()),
                    "n_queries": len(issues),
                    "by_severity": by_sev_counts,
                },
                f,
                indent=2,
            )

        # --- Write queries in YAML (human) + optional JSONL (machine) ---
        qjsonl = os.path.join(queries_dir, "queries.jsonl")
        qyaml = os.path.join(queries_dir, "queries.yaml")

        WRITE_JSONL = True  # set False to disable JSONL

        if WRITE_JSONL:
            with open(qjsonl, "w", encoding="utf-8") as f:
                for q in issues:
                    f.write(json.dumps(asdict(q.finalize())) + "\n")

        by_sev = defaultdict(int)
        by_dom = defaultdict(int)
        yaml_issues = []
        for q in issues:
            q.finalize()
            by_sev[q.severity] += 1
            by_dom[q.domain] += 1
            yaml_issues.append(
                {
                    "domain": q.domain,
                    "rule_id": q.rule_id,
                    "severity": q.severity,
                    "message": q.message,
                    "usubjid": q.usubjid,
                    "row": (q.record_loc or {}).get("row"),
                    "column": (q.record_loc or {}).get("column"),
                    "context": q.context,
                    "created_at": q.created_at,
                    "qid": q.qid,
                }
            )

        yaml_payload = {
            "study_id": self.study_id,
            "generated_at": iso_now(),
            "summary": {
                "total": len(issues),
                "by_severity": dict(by_sev),
                "by_domain": dict(by_dom),
            },
            "issues": yaml_issues,
        }
        with open(qyaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_payload, f, sort_keys=False)

        # --- ADaM minimal derivations ---
        adsl = self._build_adsl(frames.get("DM"), frames.get("EX"))
        adae = self._build_adae(frames.get("AE"), frames.get("EX"))
        advs = self._build_advs(frames.get("VS"))

        if adsl is not None and not adsl.empty:
            adsl.to_csv(os.path.join(adam_dir, "ADSL.csv"), index=False)
        if adae is not None and not adae.empty:
            adae.to_csv(os.path.join(adam_dir, "ADAE.csv"), index=False)
        if advs is not None and not advs.empty:
            advs.to_csv(os.path.join(adam_dir, "ADVS.csv"), index=False)

        # Return paths (prefer YAML)
        return {
            "sdtm_dir": sdtm_dir,
            "adam_dir": adam_dir,
            "queries_yaml": qyaml,
            "queries_file": qjsonl if WRITE_JSONL else None,
            "report_file": rpath,
        }

    # -------------------- validators --------------------
    def _validate_vs(self, vs: Optional[pd.DataFrame]) -> List[Query]:
        if vs is None or vs.empty:
            return []
        mech = (self.protocol or {}).get("mechanism_class", "").upper()
        issues: List[Query] = []

        # basic schema
        for col in ["USUBJID", "VSTESTCD", "VSORRES", "VSDTC"]:
            if col not in vs.columns:
                issues.append(
                    Query(self.study_id, "VS", "VS_REQUIRED", "MAJOR", f"Missing required column {col}").finalize()
                )

        # type check
        if "VSDTC" in vs.columns:
            bad = pd.to_datetime(vs["VSDTC"], errors="coerce", utc=True).isna() & vs["VSDTC"].notna()
            for idx in vs[bad].index.tolist():
                r = vs.loc[idx]
                issues.append(
                    Query(
                        self.study_id,
                        "VS",
                        "VS_TYPES",
                        "MAJOR",
                        "VSDTC not a valid datetime",
                        usubjid=r.get("USUBJID"),
                        record_loc={"row": int(idx), "column": "VSDTC"},
                        context={"VSDTC": r.get("VSDTC")},
                    ).finalize()
                )

        # semantic with antihypertensive exception
        for idx, r in vs.iterrows():
            tcd = str(r.get("VSTESTCD", ""))
            try:
                val = float(r.get("VSORRES"))
            except Exception:
                continue

            if tcd == "SYSBP":
                low, high = 80, 200
                is_low, is_high = val < low, val > high
                if is_low and mech == "ANTIHYPERTENSIVE":
                    continue
                if is_low or is_high:
                    issues.append(
                        Query(
                            self.study_id,
                            "VS",
                            "VS_SYS_RANGE",
                            "MAJOR",
                            "SYSBP outside plausible limits",
                            r.get("USUBJID"),
                            {"row": int(idx), "column": "VSORRES"},
                            {"VSTESTCD": "SYSBP", "VSORRES": val},
                        ).finalize()
                    )
            if tcd == "DIABP":
                low, high = 50, 120
                is_low, is_high = val < low, val > high
                if is_low and mech == "ANTIHYPERTENSIVE":
                    continue
                if is_low or is_high:
                    issues.append(
                        Query(
                            self.study_id,
                            "VS",
                            "VS_DIA_RANGE",
                            "MAJOR",
                            "DIABP outside plausible limits",
                            r.get("USUBJID"),
                            {"row": int(idx), "column": "VSORRES"},
                            {"VSTESTCD": "DIABP", "VSORRES": val},
                        ).finalize()
                    )
        return issues

    def _validate_ae(self, ae: Optional[pd.DataFrame], vs: Optional[pd.DataFrame]) -> List[Query]:
        if ae is None or ae.empty:
            return []
        issues: List[Query] = []
        if vs is None or vs.empty or self.ae_map.empty:
            return issues

        vs = vs.copy()
        vs["DTM"] = pd.to_datetime(vs.get("VSDTC"), errors="coerce", utc=True)

        for idx, row in ae.iterrows():
            term = str(row.get("AEDECOD", "") or "").upper()
            aedt = pd.to_datetime(row.get("AESTDTC"), errors="coerce", utc=True)
            if not term or pd.isna(aedt):
                continue
            rules = self.ae_map[self.ae_map["AEDECOD"].str.upper() == term]
            if rules.empty:
                continue

            ok = False
            for _, rr in rules.iterrows():
                if rr.get("MEAS_DOMAIN", "").upper() != "VS":
                    continue  # this minimal demo checks VS corroboration
                test = str(rr.get("MEAS_TESTCD", "")).upper()
                upper = rr.get("UPPER")
                lower = rr.get("LOWER")
                w = str(rr.get("WINDOW", "P3D")).upper()
                # parse PnD → days window
                days = int(w.replace("P", "").replace("D", "")) if "D" in w else 3
                lo, hi = aedt - timedelta(days=days), aedt + timedelta(days=days)
                sub = vs[(vs["USUBJID"] == row.get("USUBJID")) & (vs["DTM"] >= lo) & (vs["DTM"] <= hi)]
                if test:
                    sub = sub[sub["VSTESTCD"].astype(str).str.upper() == test]
                if sub.empty:
                    continue
                vals = pd.to_numeric(sub["VSORRES"], errors="coerce")
                if pd.notna(upper) and (vals >= float(upper)).any():
                    ok = True
                if pd.notna(lower) and (vals <= float(lower)).any():
                    ok = True
            if not ok:
                issues.append(
                    Query(
                        self.study_id,
                        "AE",
                        "AE_GENERIC_CORROBORATION",
                        "MAJOR",
                        "AE lacks corroborating measurement in window",
                        row.get("USUBJID"),
                        {"row": int(idx), "column": "AESTDTC"},
                        {"AEDECOD": row.get("AEDECOD"), "AESTDTC": row.get("AESTDTC")},
                    ).finalize()
                )
        return issues

    def _validate_temporal(self, ex: Optional[pd.DataFrame], vs: Optional[pd.DataFrame]) -> List[Query]:
        if ex is None or ex.empty or vs is None or vs.empty:
            return []
        issues: List[Query] = []
        vs = vs.copy()
        vs["VSDTM"] = pd.to_datetime(vs.get("VSDTC"), errors="coerce", utc=True)
        ex = ex.copy()
        ex["EXDTM"] = pd.to_datetime(ex.get("EXDTC"), errors="coerce", utc=True)

        w = (self.protocol.get("timing_windows") or {}).get(
            "exposure_followup_vs", {"min": "PT48H", "max": "PT72H"}
        )

        def parse_hours(s):
            s = str(s).upper()
            return int(s.replace("PT", "").replace("H", "")) if "H" in s else 48

        hmin, hmax = parse_hours(w.get("min", "PT48H")), parse_hours(w.get("max", "PT72H"))

        for idx, row in ex.iterrows():
            exdtm = row.get("EXDTM")
            if pd.isna(exdtm):
                continue
            same = vs[(vs["USUBJID"] == row.get("USUBJID")) & (vs["VSDTM"].dt.date == exdtm.date())]
            foll = vs[
                (vs["USUBJID"] == row.get("USUBJID"))
                & (vs["VSDTM"] >= exdtm + timedelta(hours=hmin))
                & (vs["VSDTM"] <= exdtm + timedelta(hours=hmax))
            ]
            if same.empty or foll.empty:
                issues.append(
                    Query(
                        self.study_id,
                        "EX",
                        "EX_REQUIRED_VS_WINDOWS",
                        "MINOR",
                        "Missing VS on dose day and/or within +window",
                        row.get("USUBJID"),
                        {"row": int(idx), "column": "EXDTC"},
                        {"EXDTC": row.get("EXDTC")},
                    ).finalize()
                )
        return issues

    # -------------------- ADaM builders (minimal) --------------------
    def _build_adsl(self, dm: Optional[pd.DataFrame], ex: Optional[pd.DataFrame]) -> pd.DataFrame:
        if dm is None or dm.empty:
            return pd.DataFrame()
        adsl = dm.copy()
        if ex is not None and not ex.empty and "EXDTC" in ex.columns:
            t = ex.copy()
            t["EXDTM"] = pd.to_datetime(t["EXDTC"], errors="coerce", utc=True)
            agg = (
                t.groupby("USUBJID")["EXDTM"]
                .agg(["min", "max"])
                .reset_index()
                .rename(columns={"min": "TRTSDTM", "max": "TRTEDTM"})
            )
            adsl = adsl.merge(agg, on="USUBJID", how="left")
        return adsl

    def _build_adae(self, ae: Optional[pd.DataFrame], ex: Optional[pd.DataFrame]) -> pd.DataFrame:
        if ae is None or ae.empty:
            return pd.DataFrame()
        adae = ae.copy()
        adae["AESTDTM"] = pd.to_datetime(adae.get("AESTDTC"), errors="coerce", utc=True)
        if ex is not None and not ex.empty:
            t = ex.copy()
            t["EXDTM"] = pd.to_datetime(t.get("EXDTC"), errors="coerce", utc=True)
            first = t.groupby("USUBJID")["EXDTM"].min().rename("TRTSDTM")
            adae = adae.merge(first, on="USUBJID", how="left")
            adae["TRTEMFL"] = (adae["AESTDTM"] >= adae["TRTSDTM"]).map({True: "Y", False: "N"})
        return adae

    def _build_advs(self, vs: Optional[pd.DataFrame]) -> pd.DataFrame:
        if vs is None or vs.empty:
            return pd.DataFrame()
        advs = vs.copy()
        advs["VSDTM"] = pd.to_datetime(advs.get("VSDTC"), errors="coerce", utc=True)
        base = (
            advs.dropna(subset=["VSORRES"])
            .groupby(["USUBJID", "VSTESTCD"])["VSORRES"]
            .first()
            .reset_index()
            .rename(columns={"VSORRES": "BASE"})
        )
        advs = advs.merge(base, on=["USUBJID", "VSTESTCD"], how="left")
        advs["AVAL"] = pd.to_numeric(advs["VSORRES"], errors="coerce")
        advs["CHG"] = advs["AVAL"] - pd.to_numeric(advs["BASE"], errors="coerce")
        advs.rename(columns={"VSTESTCD": "PARAMCD"}, inplace=True)
        return advs
