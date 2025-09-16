
import os, pandas as pd

class RiskSentinel:
    def __init__(self, out_dir="./outputs"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.adam_dir = os.path.join(self.out_dir, "adam")
        os.makedirs(self.adam_dir, exist_ok=True)

    def build_adsl(self, dm): return dm.copy()
    def build_advs(self, vs): return vs.copy()
    def build_adae(self, ae): return ae.copy()

    def run(self, sdtm):
        adsl = self.build_adsl(sdtm.get("DM", pd.DataFrame()))
        advs = self.build_advs(sdtm.get("VS", pd.DataFrame()))
        adae = self.build_adae(sdtm.get("AE", pd.DataFrame()))
        adsl.to_csv(os.path.join(self.adam_dir, "ADSL.csv"), index=False)
        advs.to_csv(os.path.join(self.adam_dir, "ADVS.csv"), index=False)
        adae.to_csv(os.path.join(self.adam_dir, "ADAE.csv"), index=False)
        return {"adam_dir": self.adam_dir}
