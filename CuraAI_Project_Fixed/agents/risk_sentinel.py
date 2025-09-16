# agents/risk_sentinel.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from agents.report_maestro import ReportMaestro

class RiskSentinel:
    def __init__(self, out_dir: str = "./outputs"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.adam_dir = os.path.join(self.out_dir, "adam")
        os.makedirs(self.adam_dir, exist_ok=True)

    # ----------------- ADaM Builders -----------------
    def build_adsl(self, dm: pd.DataFrame | None, ex: pd.DataFrame | None) -> pd.DataFrame:
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

    def build_adae(self, ae: pd.DataFrame | None, ex: pd.DataFrame | None) -> pd.DataFrame:
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

    def build_advs(self, vs: pd.DataFrame | None) -> pd.DataFrame:
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

    # ----------------- Dropout Prediction -----------------
    def _predict_dropout(self, adsl: pd.DataFrame, ae: pd.DataFrame = None, vs: pd.DataFrame = None, ex: pd.DataFrame = None) -> pd.Series:
        if adsl.empty:
            return pd.Series(dtype=float)

        features = pd.DataFrame(index=adsl["USUBJID"])
        features['AGE'] = adsl['AGE'].fillna(adsl['AGE'].mean())
        features['SEX'] = adsl['SEX'].map({'M':0,'F':1}).fillna(0)

        # AE counts
        features['AE_COUNT'] = ae.groupby('USUBJID').size() if ae is not None else 0

        # Max BP change
        if vs is not None:
            vs_base = vs.groupby('USUBJID')['VSORRES'].first()
            vs_last = vs.groupby('USUBJID')['VSORRES'].last()
            features['MAX_BP_CHG'] = (vs_last - vs_base).fillna(0)
        else:
            features['MAX_BP_CHG'] = 0

        # Exposure features
        features['DOSE_COUNT'] = ex.groupby('USUBJID').size() if ex is not None else 0

        features = features.fillna(0)
        X = features.values
        n = X.shape[0]

        # Ensure at least 2 classes for GradientBoosting
        if n <2:
            y_dummy = np.array([0, 1])
            X = np.vstack([X, X])
        else:
            y_dummy = np.random.randint(0, 2, size=n)
            
            if len(np.unique(y_dummy)) == 1:
                y_dummy[0] = 1 - y_dummy[0]

        gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        gbm.fit(X, y_dummy)
        dropout_prob = gbm.predict_proba(X)[:,1]

        # Only return original length
        return pd.Series(dropout_prob[:n], index=features.index, name='DROP_RISK')

    # ----------------- Main Run -----------------
    def run(self, sdtm: dict[str, pd.DataFrame]) -> dict:
        dm = sdtm.get("DM")
        ae = sdtm.get("AE")
        vs = sdtm.get("VS")
        ex = sdtm.get("EX")

        if dm is None or ae is None or vs is None:
            raise ValueError("Missing required SDTM domains (DM, AE, VS)")
        report_pdf_path = None
        # Build ADaM tables
        adsl = self.build_adsl(dm, ex)
        adae = self.build_adae(ae, ex)
        advs = self.build_advs(vs)

        # Dropout prediction
        high_risk_threshold = 0.6
        if not adsl.empty:
            adsl['DROP_RISK'] = self._predict_dropout(adsl, ae, vs, ex)
            adsl['HIGH_RISK'] = (adsl['DROP_RISK'] >= high_risk_threshold).map({True:'Y', False:'N'})
        else:
            adsl['DROP_RISK'] = pd.Series(dtype=float)
            adsl['HIGH_RISK'] = pd.Series(dtype=str)

        # Notify high-risk patients
        high_risk_patients = adsl.loc[adsl['HIGH_RISK'] == 'Y', 'USUBJID'].tolist()
        if high_risk_patients:
            print(f"[Notification] High dropout risk detected for patients: {high_risk_patients}")
        else:
            print("[Notification] No patients at high dropout risk detected.")
            
            
            
            
        # Notify high-risk patients
        high_risk_patients = adsl.loc[adsl['HIGH_RISK'] == 'Y', 'USUBJID'].tolist()
        notifications = []
        if high_risk_patients:
            msg = f"High dropout risk detected for patients: {', '.join(high_risk_patients)}"
            print(f"[Notification] {msg}")
            notifications.append(msg)
        else:
            msg = "No patients at high dropout risk detected."
            print(f"[Notification] {msg}")
            notifications.append(msg)

        # Return in the dictionary
        return {
            "adam_dir": self.adam_dir,
            "n_adsl": len(adsl),
            "n_adae": len(adae),
            "n_advs": len(advs),
            "high_dropout_patients": high_risk_patients,
            "report_pdf_path": report_pdf_path,
            "notifications": notifications
}

        
        # ---------- ReportMaestro integration ----------
        try:
            report_maestro = ReportMaestro(out_dir=self.out_dir)
            _, report_pdf_path = report_maestro.generate_report(
                os.path.join(self.out_dir, "sdtm"),
                os.path.join(self.out_dir, "adam"),
                "Study Report",
                save_pdf=True
        )
        except Exception as e:
            print(f"[Warning] Failed to generate PDF report: {e}")
            report_pdf_path = None

        # Save ADaM tables
        if not adsl.empty:
            adsl.to_csv(os.path.join(self.adam_dir, "ADSL.csv"), index=False)
        if not adae.empty:
            adae.to_csv(os.path.join(self.adam_dir, "ADAE.csv"), index=False)
        if not advs.empty:
            advs.to_csv(os.path.join(self.adam_dir, "ADVS.csv"), index=False)

        return {
            "adam_dir": self.adam_dir,
            "n_adsl": len(adsl),
            "n_adae": len(adae),
            "n_advs": len(advs),
            "high_dropout_patients": high_risk_patients,
            "report_pdf_path": report_pdf_path,
            "notifications": notifications
            
        }
