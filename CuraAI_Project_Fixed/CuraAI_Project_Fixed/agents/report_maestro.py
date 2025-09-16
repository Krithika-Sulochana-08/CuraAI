
import os, pandas as pd

class ReportMaestro:
    def __init__(self, out_dir="./outputs"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.csr_dir = os.path.join(self.out_dir, "csr")
        os.makedirs(self.csr_dir, exist_ok=True)

    def run(self, adam_dir):
        path = os.path.join(adam_dir, "ADSL.csv")
        n = 0
        if os.path.exists(path):
            df = pd.read_csv(path)
            n = len(df)
        md = "# Clinical Study Report (Auto-draft)\n\n## Population\n- N subjects: {}".format(n)
        out_md = os.path.join(self.csr_dir, "CSR.md")
        with open(out_md, "w") as f:
            f.write(md)
        return {"csr_dir": self.csr_dir, "csr_md": out_md}
