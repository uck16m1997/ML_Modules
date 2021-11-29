from ml_package import *

def eda_prep():




    # Remove rows with missing labels if any
    df.dropna(axis=0, subset=[label_col], inplace=True)



    return (df.drop(columns=[label_col]), df[label_col])


def plot_histograms(df, groups=None, cont_cols=None):
    # Numeric Histograms
    if groups:
        for k, v in groups.items():
            fig, axs = plt.subplots(len(v), figsize=(18, 18))
            fig.suptitle(k)
            for i in range(len(v)):
                axs[i].hist(df[v[i]])
                axs[i].set_title(v[i])
            plt.show()
    else:
        if not cont_cols:
            col_details = data_prep.get_column_types(df)
            cont_cols = col_details["Continious"]
        for c in cont_cols:
            plt.clf()
            plt.figure(figsize=(18, 18), dpi=80)
            plt.title(c)
            plt.hist(df[c])
            plt.show()


def plot_correlation_heatmaps(df, groups=groups):
    # Correlation Heatmaps
    for k, v in groups.items():
        plt.clf()
        plt.figure(figsize=(18, 18), dpi=80)
        plt.title(k)
        sns.heatmap(df[v].corr())
        plt.show()


def plot_cat_bars(df, obj_columns):
    for c in obj_columns:
        plt.clf()
        plt.figure(figsize=(8, 6), dpi=80)
        df[c].value_counts().plot(kind="bar")
        plt.title(c)
        plt.show()


def plot_scatters(df, x_cols, y_cols=None):
    if not y_cols:
        y_cols = x_cols
    for cx in x_cols:
        for cy in y_cols:
            if cy != cx:
                plt.clf()
                plt.figure(figsize=(8, 6), dpi=80)
                plt.scatter(x=df[cx], y=df[cy])
                plt.xlabel(cx)
                plt.ylabel(cy)
                plt.show()


def plot_categorical_proportions(X, y, cols):
    tmp_X = X.copy()
    tmp_X["Target"] = y
    import matplotlib.patches as mpatches

    for c in cols:
        plt.clf()
        plt.figure(figsize=(16, 10))
        Default = tmp_X.groupby([c])["Target"].mean()
        Non_Default = 1 - Default
        bar_d = sns.barplot(
            x=c,
            y="Target",
            data=(Default + Non_Default).reset_index(),
            color="darkblue",
        )
        bar_nd = sns.barplot(
            x=c,
            y="Target",
            data=Non_Default.reset_index(),
            color="lightblue",
        )
        top_bar = mpatches.Patch(color="darkblue", label="Default = Yes")
        bottom_bar = mpatches.Patch(color="lightblue", label="Default = No")
        plt.legend(
            handles=[top_bar, bottom_bar],
            loc="lower right",
        )

        plt.show()


if __name__ == "__main__":

    X, y = eda_prep()

    # Find inappropriate features
    col_dict = data_prep.find_inapp(X)
    # Remove constant features
    X.drop(columns=col_dict["Constant"], inplace=True)
    # Remove categoric columns with too many unique values
    X.drop(columns=col_dict["Unique"], inplace=True)
    # Remove categoric columns with too many null values
    X.drop(columns=col_dict["Null"], inplace=True)
    # Remove categoric columns with too little variance
    X.drop(columns=col_dict["Low Variance"], inplace=True)

    col_details = data_prep.get_column_types(X)

    ### Univariate Plots
    # Plot bar plots for categorics
    plot_cat_bars(X, col_details["Categoric"])

    # Plot bar plots for discrete
    plot_cat_bars(X, col_details["Discrete"])

    # plot hist for continious
    plot_histograms(X, cont_cols=list(col_details["Continious"]))

    ### Bivariate Plots
    # ### WARNING TAKES TOO LONG !!!
    # plot_scatters(X, x_cols=list(col_details["Continious"]))
    ### Histogram plots
    # plot_scatters(X, x_cols=list(col_details["Discrete"]))

    #### Group Based Plots
    # Plotting histograms
    plot_histograms(X, groups)

    # Plot correlations of groups
    plot_correlation_heatmaps(X, groups)

    ### Class Proportions for categorical variables
    plot_categorical_proportions(X, y, cols=col_details["Categoric"])
    ## Normal Probability Plot

    for k, v in groups.items():
        fig, axs = plt.subplots(len(v), figsize=(18, 18))
        fig.suptitle(k)
        for i in range(len(v)):
            tmp = df[v[i]].dropna().sort_values()
            axs[i].scatter(
                scipy.stats.probplot(tmp)[0][1], scipy.stats.probplot(tmp)[0][0]
            )
            axs[i].set_title(v[i])
        plt.show()

    # tmp = df[v[i]].dropna().sort_values()

    scipy.stats.zscore(tmp)

    # scipy.stats.probplot(tmp)

    # scipy.stats.probplot(tmp)[0][0]

    # scipy.stats.probplot(tmp)[0][1]

    # Numeric Test
    from scipy.stats import normaltest

    for c in num_columns:
        stat, p = normaltest(df[c].values)
        # print("Statistics=%.3f, p=%.20f" % (stat, p))
        # interpret
        alpha = 0.05
        if p > 0:
            print(c)
        # if p > alpha:
        #     print("Sample looks Gaussian (fail to reject H0)")
        # else:
        #     print("Sample does not look Gaussian (reject H0)")
