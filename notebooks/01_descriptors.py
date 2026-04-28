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
        fg_df = fg_df[[c for c in fg_df.columns if "." not in c]]

        counts = fg_df.sum().sort_values(ascending=True)
        count_vals = counts.to_numpy(dtype=int)

        fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.22)))
        ax.barh(counts.index, count_vals)
        for val, patch in zip(count_vals, ax.patches):
            ax.text(
                patch.get_width() + count_vals.max() * 0.01,
                patch.get_y() + patch.get_height() / 2,
                str(val),
                va="center",
                ha="left",
                fontsize=8,
            )
        ax.set_xlim(right=count_vals.max() * 1.12)
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
        import numpy as np

        clogp = desc_df["MolLogP"].dropna().to_numpy()
        logd = df["Y"].dropna().to_numpy()

        bins = np.linspace(
            min(clogp.min(), logd.min()) - 0.5,
            max(clogp.max(), logd.max()) + 0.5,
            45,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            clogp,
            bins=bins,
            density=True,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            label=f"ClogP (n={len(clogp):,})",
        )
        ax.hist(
            logd,
            bins=bins,
            density=True,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            label=f"logD (n={len(logd):,})",
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title("ClogP vs. logD distribution")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "01_clogp_distribution")
        return fig

    _()
    return


@app.cell
def _(desc_df, df, save_fig):
    def _():
        import matplotlib.pyplot as plt

        mw = desc_df["MolWt"].dropna()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(mw, bins=40, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Molecular weight (Da)")
        ax.set_ylabel("Count")
        ax.set_title(f"Molecular weight distribution (n={len(mw):,} / {len(df):,})")
        plt.tight_layout()
        save_fig(fig, "01_mw_distribution")
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
def _(df, save_fig):
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


@app.cell
def _(save_fig):
    def _():
        from collections import Counter

        import matplotlib.pyplot as plt
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
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

        def scaffold_cluster_sizes(smiles_list):
            scaffolds = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    scaffolds.append("")
                    continue
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
            counts = Counter(scaffolds)
            return np.array(list(counts.values()), dtype=int)

        fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)

        for col, split in enumerate(split_names):
            for row, (label, splits_dict) in enumerate(strategies):
                ax = axes[row, col]
                smiles = splits_dict[split]["Drug"].tolist()
                sizes = scaffold_cluster_sizes(smiles)
                n_unique = len(sizes)
                pct_singleton = 100 * (sizes == 1).sum() / n_unique

                ax.hist(
                    sizes,
                    bins=range(1, sizes.max() + 2),
                    align="left",
                    color=colors[split],
                    edgecolor="white",
                    linewidth=0.4,
                )
                ax.set_title(
                    f"{label} — {split}  |  {n_unique} scaffolds, {pct_singleton:.0f}% singletons",
                    fontsize=9,
                )
                ax.set_xlabel("Molecules per scaffold")
                ax.set_ylabel("Number of scaffolds")
                ax.set_yscale("log")

        plt.suptitle(
            "Murcko scaffold cluster-size distribution: TDC scaffold (top) · random (middle) · stratified scaffold (bottom)",
            fontsize=10,
            y=1.01,
        )
        plt.tight_layout()
        save_fig(fig, "01_scaffold_diversity")
        return fig

    _()
    return


@app.cell
def _(save_fig):
    def _():
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn3
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
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
        colors = {"train": "#4e79a7", "valid": "#f28e2b", "test": "#e15759"}

        def scaffold_set(smiles_list):
            out = set()
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                out.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
            return out

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (label, splits_dict) in zip(axes, strategies):
            tr = scaffold_set(splits_dict["train"]["Drug"])
            va = scaffold_set(splits_dict["valid"]["Drug"])
            te = scaffold_set(splits_dict["test"]["Drug"])
            v = venn3(
                [tr, va, te],
                set_labels=("train", "valid", "test"),
                set_colors=(colors["train"], colors["valid"], colors["test"]),
                alpha=0.6,
                ax=ax,
            )
            ax.set_title(
                f"{label}  |  {len(tr | va | te)} unique scaffolds", fontsize=10
            )

        fig.suptitle("Murcko scaffold overlap between splits", fontsize=12)
        plt.tight_layout()
        save_fig(fig, "01_scaffold_venn")
        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
