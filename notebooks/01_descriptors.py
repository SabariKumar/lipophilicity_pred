import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
def _(desc_df, df):
    import matplotlib.pyplot as plt

    def fig1():
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
        return fig

    fig1()
    return


@app.cell
def _(desc_df, mo):
    def fig2():
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
        return mo.vstack(
            [
                fig,
                mo.md(
                    f"**{n_low}** / {len(variances)} descriptors have variance < {low_var_threshold}"
                ),
            ]
        )

    fig2()
    return


@app.cell
def _():
    return


@app.cell
def _(df, mo):
    import matplotlib.pyplot as plt

    from src.features import smiles_to_fgs

    fg_df = smiles_to_fgs(df["Drug"])

    counts = fg_df.sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.22)))
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("Number of molecules")
    ax.set_title("Functional group prevalence across dataset")
    plt.tight_layout()

    mo.vstack(
        [
            fig,
            mo.md(
                f"**{(counts == 0).sum()}** functional groups absent in all molecules"
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
