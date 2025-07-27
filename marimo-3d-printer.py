import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import random
    import time
    from scipy.stats import norm
    from sklearn.linear_model import LinearRegression
    import marimo as mo
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import uuid
    from IPython.display import clear_output
    return (
        LinearRegression,
        clear_output,
        mo,
        norm,
        np,
        pd,
        plt,
        random,
        sns,
        uuid,
    )


@app.cell
def _(pd):
    df = pd.read_csv("https://raw.githubusercontent.com/JotaBlanco/AnalyticsExample/refs/heads/main/3D_printer_material_data.csv")
    df
    return (df,)


@app.cell
def _(df):
    df["material"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 01 - First EDA""")
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    for col in ["infill_pattern", "material"]:
      print(col, df[col].nunique())
      print(df[col].value_counts())
      print()
    return


@app.cell
def _(df):
    for col_i in df.columns:
      print(col_i, df[col_i].nunique())
      print(df[col_i].value_counts())
      print()
    return


@app.cell
def _(df, plt, sns):
    # Assuming df is your DataFrame
    sns.set(style="whitegrid")

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Plot each numeric column grouped by material
    for col_j in numeric_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="material", y=col_j)
        plt.title(f'Distribution of {col_j} by Material')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 02 - Data generation""")
    return


@app.cell
def _(LinearRegression, norm, np, pd):
    def generate_elastic_frequency_sine(n_points, base_mu, std, amplitude,
                                        base_freq, freq_mod_amp, freq_mod_freq, phase=0):
        t = np.arange(n_points)

        # Frequency modulation over time (instantaneous frequency)
        inst_freq = base_freq + freq_mod_amp * np.sin(2 * np.pi * freq_mod_freq * t / n_points)

        # Integrate instantaneous frequency to get phase
        inst_phase = 2 * np.pi * np.cumsum(inst_freq) / n_points + phase

        # Sinusoidal mean with elastic frequency
        mu_t = base_mu + amplitude * np.sin(inst_phase)

        # Generate noisy data around the time-varying mean
        data = np.random.normal(mu_t, std)

        return data, mu_t, inst_freq



    def train_and_predict_with_noise_on_new(X_train, y_train, X_new, col_name):
        # Fit linear regressor
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on training set to get residuals
        y_train_pred = model.predict(X_train)
        residuals = y_train - y_train_pred

        # Fit normal distribution to residuals
        mu, std = norm.fit(residuals)
        mu = max(mu, 0.1*y_train_pred.mean())
        std = max(std, 0.001)

        # Predict on new data
        y_pred = model.predict(X_new)

        # Add noise
        y_pred_noisy = y_pred + np.random.normal(mu, std, size=len(y_pred))

        # Soften it up a bit
        if "temperature" in col_name.lower():
          num_points=15
        elif "speed" in col_name.lower():
          num_points=50
        else:
          num_points=5
        y_pred_noisy_softened = np.convolve(y_pred_noisy, np.ones(num_points)/num_points, mode='same')

        return y_pred_noisy_softened


    def expand_df_to_n_rows(df, n):
        """Repeat df until it has exactly n rows."""
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")
        repeats = n // len(df)
        remainder = n % len(df)
        expanded = pd.concat(
            [df] * repeats + [df.iloc[:remainder]],
            ignore_index=True
        )
        return expanded



    def gen_dataset(size, df):
      cat_vars = ["material", "infill_pattern", "layer_height", "wall_thickness", "infill_density"]

      # Generate random single sample
      df_sim = df.sample(1)[cat_vars].merge(df[cat_vars], how="left", on=cat_vars)

      # Parameters
      material_filter = df["material"]==df_sim["material"].iloc[0]
      mean_material = df.loc[material_filter, "nozzle_temperature"].mean()
      std_material = df.loc[material_filter, "nozzle_temperature"].std()
      n = size
      base_mu = norm.rvs(loc=mean_material, scale=std_material*0.25)
      std = std_material/2
      amplitude = base_mu*max(0.01, norm.rvs(loc=0.05, scale=0.025))
      base_freq = norm.rvs(loc=n/50, scale=amplitude*3)
      freq_mod_amp = norm.rvs(loc=n/300, scale=n/30)
      freq_mod_freq = n/300

      # Generate nozzle_temp data
      data, mu_t, inst_freq = generate_elastic_frequency_sine(
          n, base_mu, std, amplitude, base_freq, freq_mod_amp, freq_mod_freq)

      # populate df_sim with nozzle temp
      df_sim = expand_df_to_n_rows(df_sim, n)
      df_sim["nozzle_temperature"] = data

      # populate with rest of columns
      pred_cols = ['nozzle_temperature']
      for col in ['print_speed', 'bed_temperature', 'fan_speed']:
        X = df[pred_cols]
        y = df[col]
        X_new = df_sim[pred_cols]
        df_sim[col] = train_and_predict_with_noise_on_new(X, y, X_new, col)

      # set up limits
      for col in ['print_speed', 'fan_speed']:
        df_sim[col] = df_sim[col].clip(lower=df[col].min(), upper=df[col].max())
      for col in ['nozzle_temperature', 'bed_temperature']:
        df_sim[col] = df_sim[col].clip(lower=df.loc[material_filter,col].min())

      return df_sim
    return (gen_dataset,)


@app.cell
def _(df, gen_dataset, random):
    size = random.randint(500, 5000)
    print(size)
    df_sim = gen_dataset(size, df)
    df_sim
    return (df_sim,)


@app.cell
def _(df_sim, plt):
    # Plot data and mean
    plt.subplot(2,1,1)
    plt.plot(df_sim["nozzle_temperature"], label='nozzle_temperature')
    plt.legend()

    # Plot instantaneous frequency
    plt.subplot(2,1,2)
    plt.hist(df_sim["nozzle_temperature"])
    plt.title("nozzle_temperature")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 03 - Bulk Data generation""")
    return


@app.cell
def _(clear_output, df, gen_dataset, pd, random, uuid):
    df_bulk = pd.DataFrame()

    n_sims = 500
    for i in range(n_sims):
      size_i = random.randint(500, 5000)
      df_sim_i = gen_dataset(size_i, df)
      df_sim_i["piece_id"] = str(uuid.uuid4())
      df_bulk = pd.concat([df_bulk, df_sim_i])
      clear_output(wait=True)
      print(f"i = {i}/{n_sims}")

    df_bulk = df_bulk.reset_index(drop=True)
    df_bulk
    return (df_bulk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 04 - Plots
    These can be used to show analysis of data distribution over several runs (several pieces).
    """
    )
    return


@app.cell
def _(sns):
    # Optional: improve aesthetics
    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

    numerical_cols = [
        'layer_height', 'wall_thickness', 'infill_density',
        'nozzle_temperature', 'bed_temperature', 'print_speed', 'fan_speed'
    ]
    return (numerical_cols,)


@app.cell
def _(df_bulk, numerical_cols, plt, sns):
    def plot_numerical_distributions(df, cols):
        for col in cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=50, color="skyblue")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    plot_numerical_distributions(df_bulk, numerical_cols)
    return


@app.cell
def _(df_bulk, numerical_cols, plt, sns):
    def plot_boxplots(df, cols, color="salmon"):
        for col in cols:
            plt.figure(figsize=(8, 1.5))
            sns.boxplot(x=df[col], color=color)
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.show()
    plot_boxplots(df_bulk, numerical_cols)
    return


@app.cell
def _(df_bulk, numerical_cols, plt, sns):
    def plot_violinplots(df, cols, color="lightgreen"):
        for col in cols:
            plt.figure(figsize=(8, 3))
            sns.violinplot(x=df[col], color=color)
            plt.title(f"Violin Plot of {col}")
            plt.tight_layout()
            plt.show()

    plot_violinplots(df_bulk, numerical_cols)
    return


@app.cell
def _(df_bulk, numerical_cols, plt, sns):
    def plot_pairplot(df, cols, diag_kind="kde", alpha=0.01):
        sns.pairplot(
            df[cols],
            diag_kind=diag_kind,
            plot_kws=dict(alpha=alpha, edgecolor='none')
        )
        plt.show()

    plot_pairplot(df_bulk, numerical_cols)
    return (plot_pairplot,)


@app.cell
def _(df_bulk, numerical_cols, plt, sns):
    def plot_correlation_matrix(df, cols, figsize=(10, 8)):
        corr_matrix = df[cols].corr()
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title("Correlation Matrix of Numerical Features", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    plot_correlation_matrix(df_bulk, numerical_cols)
    return (plot_correlation_matrix,)


@app.cell
def _(df_bulk, numerical_cols, plt, sns):
    def plot_histograms_by_hue(df, cols, hue, bins=50, palette="Set2", alpha=0.8, multiple="stack"):
        for col in cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=df,
                x=col,
                hue=hue,
                kde=True,
                bins=bins,
                multiple=multiple,
                palette=palette,
                alpha=alpha
            )
            plt.title(f"Distribution of {col} by {hue.capitalize()}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    plot_histograms_by_hue(df_bulk, numerical_cols, hue="material")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 05 - Individual piece plots
    These can be used to analyse a single run
    """
    )
    return


@app.cell
def _(df_sim, numerical_cols, plt):
    def plot_time_series(df, cols, time_col='t', figsize=(12, 6)):
        if time_col not in df.columns:
            df[time_col] = range(len(df))
    
        plt.figure(figsize=figsize)
        for col in cols:
            plt.plot(df[time_col], df[col], label=col)
        plt.legend()
        plt.title('Time Series of Process Variables')
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_time_series(df_sim, numerical_cols)
    return


@app.cell
def _(df, numerical_cols, plot_correlation_matrix):
    plot_correlation_matrix(df, numerical_cols)
    return


@app.cell
def _(df_sim, np, plt):
    from scipy.fft import fft, fftfreq

    def plot_fft_spectrum(df, col, sampling_interval=1.0, figsize=(8, 4)):
        N = len(df)
        T = sampling_interval
        signal = df[col].to_numpy()
        yf = fft(signal)
        xf = fftfreq(N, T)[:N // 2]

        plt.figure(figsize=figsize)
        plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        plt.title(f'FFT Spectrum of {col.replace("_", " ").title()}')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_fft_spectrum(df_sim, col='nozzle_temperature')
    return


@app.cell
def _(df_sim, np, plt, sns):
    from scipy.optimize import curve_fit

    def fit_and_plot_sine(df, col, window=15, p0=[10, 0.01, 0, 0], figsize=(10, 5)):
        def sine_func(t, A, f, phi, offset):
            return A * np.sin(2 * np.pi * f * t / len(t) + phi) + offset

        t = np.arange(len(df))
        y = df[col].to_numpy()
        initial_guess = p0 if p0 != [10, 0.01, 0, 0] else [10, 0.01, 0, y.mean()]
        popt, _ = curve_fit(sine_func, t, y, p0=initial_guess)

        # Plot original signal, fit, and smoothed signal
        plt.figure(figsize=figsize)
        plt.plot(t, y, label='Noisy Signal')
        plt.plot(t, sine_func(t, *popt), label='Fitted Sine', color='orange', linewidth=2)
        plt.plot(t, df[col].rolling(window).mean(), label='Smoothed Signal', linestyle='--')
        plt.legend()
        plt.title(f'Sine Fit on {col.replace("_", " ").title()}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot residuals
        residuals = y - sine_func(t, *popt)
        plt.figure(figsize=(8, 4))
        sns.histplot(residuals, kde=True, color="slateblue")
        plt.title('Residuals of Sine Fit')
        plt.xlabel('Residual')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Example call
    fit_and_plot_sine(df_sim, col="nozzle_temperature")
    return


@app.cell
def _(df_sim, numerical_cols, plt):
    def plot_rolling_means(df, cols, time_col='t', window=20, figsize=(10, 4)):
        for col in cols:
            plt.figure(figsize=figsize)
            plt.plot(df[time_col], df[col], label=col, alpha=0.4)
            plt.plot(df[time_col], df[col].rolling(window).mean(), label=f"{col} (Rolling Mean)", linewidth=2)
            plt.title(f"{col} with Rolling Mean")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # Example call
    plot_rolling_means(df_sim, numerical_cols)
    return


@app.cell
def _(df, numerical_cols, plot_pairplot):
    plot_pairplot(df, numerical_cols, alpha=1)
    return


@app.cell
def _(df_sim, plt):
    from statsmodels.graphics.tsaplots import plot_acf

    def plot_autocorrelation(df, col, lags=50, figsize=(10, 4)):
        plt.figure(figsize=figsize)
        plot_acf(df[col], lags=lags)
        plt.title(f"Autocorrelation of {col.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.show()

    # Example call
    plot_autocorrelation(df_sim, col="nozzle_temperature")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 06 - Individual vs bulk
    This is an example of how to study a run vs several historic runs:
    """
    )
    return


@app.cell
def _(df_bulk, df_sim, plt, sns):
    def plot_distribution_comparison(df1, df2, numerical_cols):
        for col in numerical_cols:
            plt.figure(figsize=(8, 4))

            sns.histplot(df1[col], kde=True, bins=30, color='black', label='df1', stat='density', alpha=0.5)
            sns.histplot(df2[col], kde=True, bins=30, color='skyblue', label='df2', stat='density', alpha=0.7)

            plt.title(f'Distribution Comparison: {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Example call
    numerical_cols_i = ['nozzle_temperature', 'bed_temperature', 'print_speed', 'fan_speed']
    plot_distribution_comparison(df_bulk, df_sim, numerical_cols_i)
    return (numerical_cols_i,)


@app.cell
def _(df_bulk, df_sim, numerical_cols_i, pd, plt, sns):
    def plot_pairwise_comparison(df1, df2, numerical_cols=None, sample_size=10000):
        if numerical_cols is None:
            numerical_cols = df1.select_dtypes(include='number').columns.intersection(
                df2.select_dtypes(include='number').columns
            ).tolist()

        df1_sampled = df1[numerical_cols].assign(dataset='bulk').sample(sample_size)
        df2_labeled = df2[numerical_cols].assign(dataset='sim')

        combined = pd.concat([df1_sampled, df2_labeled])

        sns.pairplot(
            combined,
            hue='dataset',
            palette=dict(bulk='black', sim='skyblue'),
            plot_kws=dict(alpha=0.05, edgecolor='none')
        )

        plt.suptitle('Pairplot of Variables', y=1.02)
        plt.show()

    # Example call
    plot_pairwise_comparison(df_bulk, df_sim)

    # Example call
    plot_pairwise_comparison(df_bulk, df_sim, numerical_cols_i)
    return


@app.cell
def _(df_bulk, df_sim, pd, plt, sns):
    def plot_boxplot_comparison(df1, df2, numerical_cols=None):
        if numerical_cols is None:
            numerical_cols = df1.select_dtypes(include='number').columns.intersection(
                df2.select_dtypes(include='number').columns
            ).tolist()

        for col in numerical_cols:
            plt.figure(figsize=(6, 4))
            combined = pd.concat([
                df1[[col]].assign(dataset='bulk'),
                df2[[col]].assign(dataset='sim')
            ])
            sns.boxplot(data=combined, x='dataset', y=col, palette=dict(bulk='black', sim='skyblue'))
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
            plt.show()

    plot_boxplot_comparison(df_bulk, df_sim)
    return


@app.cell
def _(np, plt, sns):
    import matplotlib.gridspec as gridspec
    from scipy.stats import percentileofscore

    def linspace_rounded_to_nearest_5(y, num_ticks=5):
        # Step 1: Find min and max
        y_min, y_max = y.min(), y.max()

        # Step 2: Round min down and max up to nearest multiple of 5
        y_min_5 = np.floor(y_min / 5) * 5
        y_max_5 = np.ceil(y_max / 5) * 5

        # Step 3: Create linspace between those rounded bounds
        return np.linspace(y_min_5, y_max_5, num_ticks)


    def plot_timeseries_with_distribution(df_sim, df_bulk, col):
        t = np.arange(len(df_sim))
        y = df_sim[col].values
        y_bulk = df_bulk[col].values

        # Percentile transformation
        y_scaled = (y - y_bulk.min()) / (y_bulk.max() - y_bulk.min())
        y_bulk_scaled = (y_bulk - y_bulk.min()) / (y_bulk.max() - y_bulk.min())

        # Create figure and gridspec
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 6], height_ratios=[1, 1], hspace=0.1, wspace=0.05)

        # === Top Left: Histogram of df_sim[col] ===
        ax0 = plt.subplot(gs[0, 0])
        ax0.hist(df_sim[col], bins=30, orientation='horizontal', color='skyblue', edgecolor='black')
        ax0.invert_xaxis()
        ax0.set_yticks(linspace_rounded_to_nearest_5(y, num_ticks=6))
        ax0.set_xticks([])
        ax0.set_ylabel(f"{col.capitalize()}\ndistribution", fontsize=12)
        ax0.set_xlabel('')

        # === Top Right: Time Series ===
        ax1 = plt.subplot(gs[0, 1])
        ax1.plot(t, y, color='skyblue', label=col)
        ax1.set_ylabel('')
        ax1.set_yticks(linspace_rounded_to_nearest_5(y, num_ticks=6))
        ax1.tick_params(labelleft=False)  # hide y tick labels
        ax1.set_title(f'{col} Time Series with Distribution')
        ax1.grid(True)

        # === Bottom Left: Vertical KDE Histograms ===
        ax2 = plt.subplot(gs[1, 0], sharey=None)  # do not sharey explicitly here
        sns.kdeplot(y=y_bulk_scaled, ax=ax2, color='black', label='df_bulk', fill=True, alpha=0.5, linewidth=1.5)
        sns.kdeplot(y=y_scaled, ax=ax2, color='skyblue', label='df_sim', fill=True, alpha=0.7, linewidth=1.5)
        ax2.invert_xaxis()
        ax2.set_ylim(0, 1)
        ax2.set_yticks(np.linspace(0, 1, 6))
        ax2.set_xticks([])
        ax2.set_ylabel(f"{col.capitalize()}\nrelative distribution", fontsize=12)
        ax2.set_xlabel('')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

        # === Bottom Right: Percentile Time Series ===
        ax3 = plt.subplot(gs[1, 1], sharey=ax2)  # share Y axis with ax2
        ax3.plot(t, y_scaled, color='skyblue')
        ax3.set_ylabel('')
        ax3.set_yticks(np.linspace(0, 1, 6))
        ax3.tick_params(labelleft=False)  # hide y tick labels
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Time Index')
        ax3.grid(True, axis='y', linestyle='--', alpha=0.4)

        # Legend (placed globally)
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')

        plt.show()
    return (plot_timeseries_with_distribution,)


@app.cell
def _(df_bulk, df_sim, plot_timeseries_with_distribution):
    plot_timeseries_with_distribution(df_sim, df_bulk, 'nozzle_temperature')
    return


@app.cell
def _(df_bulk, df_sim, plot_timeseries_with_distribution):
    plot_timeseries_with_distribution(df_sim, df_bulk, 'bed_temperature')
    return


@app.cell
def _(df_bulk, df_sim, plot_timeseries_with_distribution):
    plot_timeseries_with_distribution(df_sim, df_bulk, 'print_speed')
    return


@app.cell
def _(df_bulk, df_sim, plot_timeseries_with_distribution):
    plot_timeseries_with_distribution(df_sim, df_bulk, 'fan_speed')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
