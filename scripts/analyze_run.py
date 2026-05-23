import numpy as np
import pandas as pd
import argparse

def analyze_file(path, protocol):
    print(f"\n=== Analyzing: {path} ===")
    df = pd.read_csv(path)

    t = df["t_s"].values
    QA = df["QA_C_per_m2"].values
    QB = df["QB_C_per_m2"].values
    Qgap = df["Qgap_C_per_m2"].values if "Qgap_C_per_m2" in df.columns else np.zeros_like(QA)
    Qoutside = df["Qoutside_C_per_m2"].values if "Qoutside_C_per_m2" in df.columns else np.zeros_like(QA)
    Qtot = df["Qtot_C_per_m2"].values if "Qtot_C_per_m2" in df.columns else (QA + QB + Qgap + Qoutside)

    # Prefer stored partition residual if available
    if "Qpartition_resid_C_per_m2" in df.columns:
        Qpart_resid = df["Qpartition_resid_C_per_m2"].values
    else:
        Qpart_resid = Qtot - (QA + QB + Qgap + Qoutside)

    # ---- 1. Total charge conservation ----
    Qtot_drift = np.max(Qtot) - np.min(Qtot)
    print(f"Total charge drift (C/m^2): {Qtot_drift:.3e}")

    # ---- 2. Full partition consistency ----
    print(f"Max full-partition residual (C/m^2): {np.max(np.abs(Qpart_resid)):.3e}")

    # ---- 3. Central assembly subtotal, for interpretation only ----
    Qassembly = QA + QB + Qgap
    assembly_span = np.max(Qassembly) - np.min(Qassembly)
    print(f"Assembly subtotal drift QA+QB+Qgap (C/m^2): {assembly_span:.3e}")

    # ---- 4. Phase boundary jumps ----
    n_pre = protocol["n_pre"]
    n_app = protocol["n_approach"]
    n_hold = protocol["n_hold"]

    transitions = {
        "pre→approach": n_pre,
        "approach→hold": n_pre + n_app,
        "hold→separate": n_pre + n_app + n_hold,
    }

    print("\nPhase boundary jumps:")
    for name, idx in transitions.items():
        if 1 <= idx < len(QA):
            dQA = QA[idx] - QA[idx - 1]
            dQB = QB[idx] - QB[idx - 1]
            dQgap = Qgap[idx] - Qgap[idx - 1]
            dQoutside = Qoutside[idx] - Qoutside[idx - 1]
            dQtot = Qtot[idx] - Qtot[idx - 1]
            print(
                f"{name:18s} "
                f"dQA={dQA:.3e}, dQB={dQB:.3e}, dQgap={dQgap:.3e}, "
                f"dQout={dQoutside:.3e}, dQtot={dQtot:.3e}"
            )

    # ---- 5. Summary magnitudes ----
    print("\nSummary magnitudes:")
    print(f"QA range:       {QA.min():.3e} → {QA.max():.3e}")
    print(f"QB range:       {QB.min():.3e} → {QB.max():.3e}")
    print(f"Qgap range:     {Qgap.min():.3e} → {Qgap.max():.3e}")
    print(f"Qoutside range: {Qoutside.min():.3e} → {Qoutside.max():.3e}")

    if "Qwin_Agap_C_per_m2" in df.columns and "Qwin_Bgap_C_per_m2" in df.columns:
        QwinA = df["Qwin_Agap_C_per_m2"].values
        QwinB = df["Qwin_Bgap_C_per_m2"].values
        print(f"Qwin_Agap range: {np.nanmin(QwinA):.3e} → {np.nanmax(QwinA):.3e}")
        print(f"Qwin_Bgap range: {np.nanmin(QwinB):.3e} → {np.nanmax(QwinB):.3e}")

    if "Jcharge_mid_A_per_m2" in df.columns:
        J = df["Jcharge_mid_A_per_m2"].values
        print(f"Max |J_mid|:    {np.max(np.abs(J)):.3e}")


def compare_swap(file1, file2):
    print(f"\n=== Comparing swap symmetry ===")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    QA1 = df1["QA_C_per_m2"].values
    QB1 = df1["QB_C_per_m2"].values
    QA2 = df2["QA_C_per_m2"].values
    QB2 = df2["QB_C_per_m2"].values

    err_A = np.max(np.abs(QA1 - QB2))
    err_B = np.max(np.abs(QB1 - QA2))

    print(f"Max |QA(ref) - QB(swap)|: {err_A:.3e}")
    print(f"Max |QB(ref) - QA(swap)|: {err_B:.3e}")

    if "Qwin_Agap_C_per_m2" in df1.columns and "Qwin_Bgap_C_per_m2" in df2.columns:
        err_win_A = np.nanmax(np.abs(df1["Qwin_Agap_C_per_m2"].values - df2["Qwin_Bgap_C_per_m2"].values))
        err_win_B = np.nanmax(np.abs(df1["Qwin_Bgap_C_per_m2"].values - df2["Qwin_Agap_C_per_m2"].values))
        print(f"Max |Qwin_Agap(ref) - Qwin_Bgap(swap)|: {err_win_A:.3e}")
        print(f"Max |Qwin_Bgap(ref) - Qwin_Agap(swap)|: {err_win_B:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--compare", nargs=2, help="two files to compare swap symmetry")
    parser.add_argument("--n-pre", type=int, required=True)
    parser.add_argument("--n-approach", type=int, required=True)
    parser.add_argument("--n-hold", type=int, required=True)
    args = parser.parse_args()

    protocol = {
        "n_pre": args.n_pre,
        "n_approach": args.n_approach,
        "n_hold": args.n_hold,
    }

    analyze_file(args.input, protocol)

    if args.compare:
        compare_swap(args.compare[0], args.compare[1])
