import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set Seaborn style
sns.set(style="whitegrid")

# Load dataset
income_df = pd.read_csv(r"C:\Users\chamb\Downloads\Inc_Exp_Data.csv")

# Streamlit app
st.title("Expenses Dashboard")
st.write(
    "This app contains visual representation of monthly expenses and how many "
    "people are making contributions to earnings."
)

# Function to create and display a plot
def display_plot(title, plot_func):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_func(ax)
    st.pyplot(fig)
    plt.close(fig)

# Plot 1: Qualified Member
def qualified_member(ax):
    income_df["Highest_Qualified_Member"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Highest Qualified Member")

# Plot 2: Interquantile Expenses
def inter_quantile(ax):
    income_df.plot(x="Mthly_HH_Income", y="Mthly_HH_Expense", ax=ax)
    IQR = (
        income_df["Mthly_HH_Expense"].quantile(0.75)
        - income_df["Mthly_HH_Expense"].quantile(0.25)
    )
    ax.set_title(f"Interquantile Expenses (IQR = {IQR:.2f})")

# Plot 3: Earning Members
def earning_member(ax):
    income_df["No_of_Earning_Members"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("No. of Earning Members")

# Display all plots on one page
display_plot("Highest Qualified Member", qualified_member)
display_plot("Interquantile Expenses", inter_quantile)
display_plot("No. of Earning Members", earning_member)
