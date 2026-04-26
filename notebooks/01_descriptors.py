import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from src.plot_utils import save_fig

    return mo, save_fig


@app.cell
def _(mo):
    mo.md(
        """
    # RDKit Descriptor Computation
    """
    )
    return


@app.cell
def _():
    from src.data import load_lipophilicity

    df = load_lipophilicity()
    df.head()
    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"""
    **Dataset:** {len(df):,} molecules &nbsp;|&nbsp; target range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]
    """
    )
    return


@app.cell
def _(df):
    from src.features import smiles_to_descriptors

    desc_df = smiles_to_descriptors(df["Drug"])
    desc_df.head()
    return (desc_df,)


@app.cell
def _(desc_df, mo):
    n_valid = desc_df.notna().all(axis=1).sum()
    n_total = len(desc_df)
    mo.md(
        f"**Descriptors:** {desc_df.shape[1]} features &nbsp;|&nbsp; {n_valid}/{n_total} molecules fully computed"
    )
    return


@app.cell
def _(desc_df, df, save_fig):
    import matplotlib.pyplot as plt

    def _():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(df["Y"], bins=40, edgecolor="white", linewidth=0.4)
        axes[0].set_xlabel("logD")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Target distribution")

        missing_frac = desc_df.isna().mean().sort_values(ascending=False)
        axes[1].bar(
            range(len(missing_frac[missing_frac > 0])),
            missing_frac[missing_frac > 0].values,
        )
        axes[1].set_xlabel("Descriptor index (only those with missing values)")
        axes[1].set_ylabel("Fraction missing")
        axes[1].set_title("Descriptor missingness")

        plt.tight_layout()
        save_fig(fig, "01_target_distribution")
        return fig

    _()
    return


@app.cell
def _(desc_df, mo, save_fig):
    def _():
        import matplotlib.pyplot as plt
        import numpy as np

        variances = desc_df.var(numeric_only=True)
        low_var_threshold = 0.01
        n_low = (variances < low_var_threshold).sum()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            np.log10(variances.clip(lower=1e-10)),
            bins=60,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(
            np.log10(low_var_threshold),
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"threshold (var={low_var_threshold})",
        )
        ax.set_xlabel("log₁₀(variance)")
        ax.set_ylabel("Number of descriptors")
        ax.set_title("Per-descriptor variance distribution")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "01_descriptor_variance")
        return mo.vstack(
            [
                fig,
                mo.md(
                    f"**{n_low}** / {len(variances)} descriptors have variance < {low_var_threshold}"
                ),
            ]
        )

    _()
    return


@app.cell
def _(df, mo, save_fig):
    def _():
        import matplotlib.pyplot as plt

        from src.features import smiles_to_fgs

        fg_df = smiles_to_fgs(df["Drug"])

        counts = fg_df.sum().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.22)))
        ax.barh(counts.index, counts.values)
        ax.set_xlabel("Number of molecules")
        ax.set_title("Functional group prevalence across dataset")
        plt.tight_layout()
        save_fig(fig, "01_functional_groups")
        return mo.vstack(
            [
                fig,
                mo.md(
                    f"**{(counts == 0).sum()}** functional groups absent in all molecules"
                ),
            ]
        )

    _()
    return


@app.cell
def _(desc_df, df, save_fig):
    def _():
        import matplotlib.pyplot as plt

        clogp = desc_df["MolLogP"].dropna()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(clogp, bins=40, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("ClogP (MolLogP)")
        ax.set_ylabel("Count")
        ax.set_title(f"ClogP distribution (n={len(clogp):,} / {len(df):,})")
        plt.tight_layout()
        save_fig(fig, "01_clogp_distribution")
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Split Strategy Comparison

    The original TDC scaffold split assigns scaffold groups to splits without
    balancing the logD distribution.  The stratified scaffold split bins
    scaffold groups by median logD before assigning them, so each split sees
    a representative slice of the target range.
    """
    )
    return


@app.cell
def _(df, mo, save_fig):
    def _():
        import matplotlib.pyplot as plt
        from tdc.single_pred import ADME

        from src.data import get_random_split, get_splits

        tdc_split = ADME(name="Lipophilicity_AstraZeneca").get_split(
            method="scaffold", seed=42
        )
        rand_split = get_random_split(seed=42)
        strat_split = get_splits(seed=42)

        strategies = [
            ("TDC scaffold", tdc_split),
            ("Random", rand_split),
            ("Stratified scaffold", strat_split),
        ]
        split_names = ("train", "valid", "test")
        colors = {"train": "#4e79a7", "valid": "#f28e2b", "test": "#e15759"}

        fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=False)
        bins = 35
        overall_y = df["Y"].values

        for col, split in enumerate(split_names):
            for row, (label, splits_dict) in enumerate(strategies):
                ax = axes[row, col]
                y = splits_dict[split]["Y"].values
                ax.hist(
                    overall_y,
                    bins=bins,
                    color="lightgrey",
                    edgecolor="white",
                    linewidth=0.3,
                    label="full dataset",
                    zorder=1,
                )
                ax.hist(
                    y,
                    bins=bins,
                    color=colors[split],
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.3,
                    label=split,
                    zorder=2,
                )
                ax.set_title(
                    f"{label} — {split}\n"
                    f"n={len(y)}  mean={y.mean():.2f}  std={y.std():.2f}",
                    fontsize=9,
                )
                ax.set_xlabel("logD")
                ax.set_ylabel("Count")
                if col == 0:
                    ax.legend(fontsize=8)

        plt.suptitle(
            "logD distribution per split: TDC scaffold (top) · random (middle) · stratified scaffold (bottom)",
            fontsize=10,
            y=1.01,
        )
        plt.tight_layout()
        save_fig(fig, "01_split_comparison")
        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
