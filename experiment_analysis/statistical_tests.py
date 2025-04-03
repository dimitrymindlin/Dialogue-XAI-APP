import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu

import matplotlib
import tikzplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def plot_box_and_strip(df, x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_label, y=y_label, data=df)
    sns.stripplot(x=x_label, y=y_label, data=df, color=".25")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def get_groups(df, x_label, y_label):
    group_labels = df[x_label].unique()
    group1 = df[df[x_label] == group_labels[0]][y_label]
    group2 = df[df[x_label] == group_labels[1]][y_label]
    return group1, group2


def perform_ks_test(df, x_label, y_label):
    group1, group2 = get_groups(df, x_label, y_label)
    ident_test = stats.ks_2samp(group1, group2)
    print(f"Kolmogorov-Smirnov test: {ident_test}")
    return group1, group2, ident_test


def perform_t_test(group1, group2):
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False, alternative='greater')
    return t_stat, p_val


def plot_qq_plots(df, x_label, y_label):
    plt.figure(figsize=(10, 5))
    for i, group in enumerate(df[x_label].unique()):
        plt.subplot(1, 2, i + 1)
        stats.probplot(df[df[x_label] == group][y_label], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {group}")


def perform_shapiro_wilk_test(df, x_label, y_label):
    """
    Perform Shapiro-Wilk test for normality
    """
    for group in df[x_label].unique():
        stat, p_value = stats.shapiro(df[df[x_label] == group][y_label])
        print(f"Shapiro-Wilk test for {group}: Statistic={stat}, P-value={p_value}")


def calculate_effect_size(group1, group2):
    """
    Calculate effect size using Cohen's d
    """
    sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    effect_size = (np.mean(group1) - np.mean(group2)) / pooled_sd
    return effect_size, pooled_sd


from scipy.stats import t, norm
import numpy as np

def perform_power_analysis(effect_size=0.5, alpha=0.05, power=0.8, alternative='two-sided'):
    """
    Perform power analysis for an independent two-sample t-test to calculate the required sample size per group.

    Parameters:
    - effect_size: The expected difference between group means divided by the pooled standard deviation (Cohen's d).
    - alpha: Significance level (probability of Type I error).
    - power: Desired power level (1 - probability of Type II error).
    - alternative: Defines the alternative hypothesis ('two-sided', 'greater', or 'less').

    Returns:
    - sample_size: Estimated sample size per group.
    """
    # Determine the z critical value for the given alpha
    if alternative == 'two-sided':
        z_alpha = norm.ppf(1 - alpha / 2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    # Initialize sample size
    sample_size = 2

    # Iteratively compute the sample size needed to achieve the desired power
    while True:
        # Degrees of freedom for two-sample t-test
        df = 2 * (sample_size - 1)
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size / 2)
        # t critical value for the given power
        t_beta = t.ppf(1 - power, df, loc=ncp)
        # Calculate the non-central t distribution's CDF at the critical t value
        beta = t.cdf(t_beta, df, loc=ncp)
        # Check if the achieved power is within a small tolerance of the desired power
        if np.abs(beta - power) < 1e-4:
            break
        sample_size += 1

    print(f"Estimated sample size per group for {power*100}% power: {sample_size}")
    return sample_size


def perform_mann_whitney_u_test(group1, group2):
    U1, p = mannwhitneyu(group1, group2, method="exact", alternative="greater")
    return U1, p


def is_t_test_applicable(df, x_label, y_label):
    """
    Check if the t-test assumptions are met
    """
    group1, group2 = get_groups(df, x_label, y_label)
    # Check for normality
    for group in df[x_label].unique():
        stat, p_value = stats.shapiro(df[df[x_label] == group][y_label])
        if p_value < 0.05:
            #print(f"Shapiro-Wilk test for {group} failed: P-value={p_value}")
            return False
    # Check for homogeneity of variance
    levene_test = stats.levene(group1, group2)
    if levene_test.pvalue < 0.05:
        #print(f"Levene test failed: P-value={levene_test.pvalue}")
        return False

    return True


def plot_box_with_significance_bars(df, x_label, y_label, title, ttest=False, save=True):
    """
    Plot boxplot with significance bars based on the result of a statistical test.
    tttest: If True, perform t-test. Otherwise, perform Mann-Whitney U test.
    """
    group1, group2 = get_groups(df, x_label, y_label)
    if ttest:
        stat, p_value = perform_t_test(group1, group2)
        print(f"Performing t-test for {y_label}, p-value={p_value}")
        title = "T-Test: " + title
    else:
        stat, p_value = perform_mann_whitney_u_test(group1, group2)
        print(f"Performing U test for {y_label}, p-value={p_value}")
        title = "Mann-Whitney U Test: " + title

    # Obtain the sorted unique categories
    categories = sorted(df[x_label].unique())
    # Convert categories to string if not already
    categories = [str(cat) for cat in categories]

    # Begin plotting
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=x_label, y=y_label, data=df, order=categories, width=0.5)
    sns.stripplot(x=x_label, y=y_label, data=df, color=".25", jitter=True, order=categories)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Get the positions of the boxplots
    box_plot_positions = [i for i in range(len(categories))]

    # Determine the highest point on the y-axis for plotting the significance bar
    y = df[y_label].max() + 1
    h = 0.02  # Height of the significance bar
    col = 'k'  # Color of the significance bar and text

    if p_value < 0.05:
        # Draw the significance bar
        plt.plot([box_plot_positions[0], box_plot_positions[1]], [y, y], color=col, lw=1.5)
        # Add the significance level text
        sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
        plt.text((box_plot_positions[0] + box_plot_positions[1]) * 0.5, y + h, sig_text, ha='center', va='bottom',
                 color=col)

    # Set the x-axis tick labels if necessary
    plt.xticks(range(0, len(categories)), categories)

    path = "analysis_plots/" + f"{title}.tex"
    if save:
        tikzplotlib.save(path)
    else:
        plt.show()
