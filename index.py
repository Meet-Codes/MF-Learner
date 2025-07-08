import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Ensure these are imported if not already
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, \
    OneHotEncoder  # Ensure MinMaxScaler is imported if not already
import plotly.express as px
import plotly.graph_objects as go
import sklearn as sk
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, silhouette_score, mean_squared_error, \
    mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm, percentileofscore, mode
from collections import Counter
import graphviz # For Decision Tree visualization
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.cluster import KMeans
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Experimental Learning Platform",
    page_icon="🧪",
    layout="wide"
)

# --- Custom CSS for better aesthetics ---
st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #264653; /* Dark Teal */
    }
    .st-emotion-cache-zt5igj { /* Target for expander header */
        font-weight: bold;
        font-size: 1.1em;
        color: #2a9d8f; /* Viridian Green */
    }
    .st-emotion-cache-1ftl4i7 p { /* Target for markdown paragraphs */
        font-size: 1.05em;
        line-height: 1.6;
    }
    .stCode {
        background-color: #e6f7ff; /* Lighter blue background for code snippets */
        border-left: 5px solid #2196F3; /* Blue border */
        padding: 10px;
        border-radius: 5px;
        font-family: 'Fira Code', 'Cascadia Code', monospace;
    }
    .stButton>button {
        background-color: #f4a261; /* Orange-Yellow */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e76f51; /* Burnt Sienna */
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Style for Streamlit success/info/warning messages */
    .stAlert {
        border-radius: 5px;
        padding: 10px;
    }
    .stAlert.info {
        background-color: #e0f2f7;
        color: #0056b3;
        border-left: 5px solid #007bff;
    }
    .stAlert.success {
        background-color: #e6ffe6;
        color: #28a745;
        border-left: 5px solid #28a745;
    }
    .stAlert.warning {
        background-color: #fff3e0;
        color: #ffc107;
        border-left: 5px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Function to format equations ---
def latex_equation(eq_string):
    st.markdown(f"$$ {eq_string} $$")

# --- Introduction ---
st.title("🧪 ML Experimental Learning Platform")
st.markdown("""
    Welcome to an interactive platform designed for hands-on learning of Machine Learning and Statistical concepts.
    Explore topics, experiment with data, and understand algorithms through visualizations and practical tasks.
""")

# --- Sidebar for Topic Selection and Data Upload ---
st.sidebar.title("📚 Topics")
topic_options = [
    "Introduction",
    "Mean, Median, Mode",
    "Standard Deviation",
    "Percentiles",
    "Data Distribution",
    "Normal Data Distribution",
    "Scatter Plot",
    "Linear Regression",
    "Polynomial Regression",
    "Multiple Regression", # To be added later
    "Feature Scaling",     # To be added later
    "Train/Test Split",
    "Decision Tree",       # To be added later
    "Confusion Matrix",    # To be added later
    "Hierarchical Clustering", # To be added later
    "Logistic Regression", # To be added later
    "Grid Search",         # To be added later
    # "Categorical Data Encoding", # To be added later
    "K-means Clustering",
    "Bootstrap Aggregation (Bagging)", # To be added later
    "Cross Validation",    # To be added later
    "AUC - ROC Curve",     # To be added later
    "K-nearest Neighbors (KNN)", # To be added later
    "ML Playground (Coming Soon!)" # Placeholder for future expansion
]
selected_topic = st.sidebar.selectbox("Choose a Topic:", topic_options)

st.sidebar.header("📤 Global Data Uploader")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for general use (Optional)", type=["csv"])
global_df = None
if uploaded_file is not None:
    try:
        global_df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV file uploaded successfully!")
        st.sidebar.info(f"Columns available: {', '.join(global_df.columns)}")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# --- Topic Content Functions ---

def show_introduction():
    st.header("Welcome to the ML Experimental Learning Platform!")
    st.markdown("""
    This platform is designed to help you understand core Machine Learning and Statistical concepts through interactive experimentation.
    Here's what you can expect for each topic:

    * **Definitions:** Clear, beginner-friendly explanations with daily-life analogies.
    * **Equations:** Mathematical formulas explained in simple terms.
    * **Interactive Inputs:** Experiment by providing your own data or uploading datasets.
    * **Visualizations:** See concepts come alive with interactive plots.
    * **Python Code:** Understand the implementation with practical snippets.
    * **Practical Tasks:** Apply your knowledge with hands-on exercises and get immediate feedback.
    * **How it Works (Bonus):** Step-by-step visual explanations of algorithms to demystify them.

    Use the sidebar to navigate through various topics. Let's start learning by doing!
    """)
    st.info("Generated By er.Meet Korat and er.Fenil Ramani")


def topic_mean_median_mode():
    st.header("📊 Mean, Median, Mode")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What are Mean, Median, and Mode?")
        st.markdown("""
        These are measures of **central tendency**, which means they tell us about the "center" or "typical" value of a dataset. They help summarize a set of numbers into a single representative value.

        * **Mean (Average):** The sum of all values divided by the number of values. It's like finding the "fair share" if you were distributing something equally among everyone.
        * **Median:** The middle value in a dataset when it's ordered from least to greatest. If there's an even number of values, it's the average of the two middle numbers. The median is especially useful when your data has extreme values (outliers) because it's not affected by them.
        * **Mode:** The value that appears most frequently in a dataset. A dataset can have one mode (unimodal), multiple modes (multimodal), or no mode (if all values appear with the same frequency).
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're tracking the number of hours 5 students spent studying for an exam: 5 hours, 7 hours, 8 hours, 9 hours, and one very diligent student who studied 20 hours.

        * **Mean:** (5 + 7 + 8 + 9 + 20) / 5 = 49 / 5 = **9.8 hours**. Notice how the 20-hour outlier pulls the average up.
        * **Median:** First, sort the data: 5, 7, **8**, 9, 20. The median is **8 hours**, which feels more representative of a "typical" student's study time in this group, as it's not skewed by the high value.
        * **Mode:** If another student also studied 8 hours (data: 5, 7, 8, 8, 9, 20), then **8 hours** would be the mode because it appears most often. If all study times were unique, there would be no distinct mode.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Mathematical Formulas")
        st.markdown("**Mean ($\\bar{x}$):**")
        latex_equation(r'\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}')
        st.markdown("""
        * $x_i$: Represents each individual value in your dataset.
        * $\sum$: This is the Greek letter sigma, and it means "summation" – you add up all the $x_i$ values.
        * $n$: Is the total number of values (data points) in your dataset.
        """)

        st.markdown("**Median:**")
        st.markdown("""
        There isn't a single simple formula like the mean, as its calculation depends on the number of data points ($n$):
        * **If $n$ is odd:** The median is the value at the $((n+1)/2)^{th}$ position after sorting the dataset in ascending order.
        * **If $n$ is even:** The median is the average of the two middle values, specifically the values at the $(n/2)^{th}$ and $((n/2)+1)^{th}$ positions after sorting.
        """)

        st.markdown("**Mode:**")
        st.markdown("""
        The mode is the value(s) that appear with the highest frequency in the dataset. It's determined by counting occurrences, not a mathematical formula.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Experiment with Your Own Data")
        input_method_mmm = st.radio("Choose input method:", ("Enter comma-separated values", "Generate random data"))

        data_mmm = []
        if input_method_mmm == "Enter comma-separated values":
            user_input_mmm = st.text_area("Enter numbers separated by commas (e.g., 1, 2, 2, 3, 4, 5, 10, 100)", "1, 2, 2, 3, 4, 5, 10, 100")
            try:
                data_mmm = [float(x.strip()) for x in user_input_mmm.split(',') if x.strip()]
                if not data_mmm:
                    st.warning("Please enter some numbers for the experiment.")
            except ValueError:
                st.error("Invalid input. Please enter numbers separated by commas.")
        else: # Generate random data
            col1, col2, col3 = st.columns(3)
            num_points_mmm = col1.slider("Number of random data points:", 5, 100, 20, key='mmm_num_points')
            data_range_mmm = col2.slider("Range of random data (min, max):", 0, 200, (0, 100), key='mmm_data_range')
            skew_data_mmm = col3.checkbox("Introduce skewness (e.g., some large outliers)?", key='mmm_skew')

            np.random.seed(42) # For reproducibility
            data_mmm = np.random.uniform(data_range_mmm[0], data_range_mmm[1], num_points_mmm).tolist()
            if skew_data_mmm:
                # Add some outliers to make it skewed
                data_mmm.extend(np.random.uniform(data_range_mmm[1] * 2, data_range_mmm[1] * 5, int(num_points_mmm * 0.1)).tolist())
            data_mmm = sorted(data_mmm) # Sort for median clarity

        if data_mmm:
            df_data_mmm = pd.DataFrame({'Values': data_mmm})

            mean_val_mmm = np.mean(data_mmm)
            median_val_mmm = np.median(data_mmm)

            # Calculate mode(s) using scipy.stats.mode or collections.Counter
            counts_mmm = Counter(data_mmm)
            max_count_mmm = 0
            if counts_mmm:
                max_count_mmm = max(counts_mmm.values())
            mode_vals_mmm = [val for val, count in counts_mmm.items() if count == max_count_mmm]

            # Handle case where all elements are unique (no distinct mode)
            if len(mode_vals_mmm) == len(data_mmm) and len(data_mmm) > 1:
                display_mode = "No distinct mode (all values unique)"
            elif mode_vals_mmm:
                display_mode = f"{', '.join([str(m) for m in mode_vals_mmm])}"
            else:
                display_mode = "N/A (no data)"


            st.write(f"**Input Data:** {', '.join([f'{x:.2f}' for x in data_mmm])}")
            st.write(f"**Calculated Mean:** {mean_val_mmm:.2f}")
            st.write(f"**Calculated Median:** {median_val_mmm:.2f}")
            st.write(f"**Calculated Mode(s):** {display_mode}")

            st.subheader("Data Distribution with Central Tendency Measures")
            # Using Plotly for interactive visualization
            fig_mmm = px.histogram(df_data_mmm, x="Values", nbins=max(1, len(data_mmm)//5),
                                   title="Distribution of Data", opacity=0.7,
                                   color_discrete_sequence=['#457b9d']) # A nice blue shade

            # Add vertical lines for mean, median, mode
            fig_mmm.add_vline(x=mean_val_mmm, line_dash="dash", line_color="#e76f51",
                              annotation_text=f"Mean: {mean_val_mmm:.2f}", annotation_position="top right")
            fig_mmm.add_vline(x=median_val_mmm, line_dash="dot", line_color="#2a9d8f",
                              annotation_text=f"Median: {median_val_mmm:.2f}", annotation_position="top left")
            for m_val in mode_vals_mmm:
                # Add mode lines only if it's a distinct mode
                if display_mode != "No distinct mode (all values unique)":
                    fig_mmm.add_vline(x=m_val, line_dash="dashdot", line_color="#f4a261",
                                      annotation_text=f"Mode: {m_val:.2f}", annotation_position="bottom right")

            st.plotly_chart(fig_mmm, use_container_width=True)
            st.info("💡 **Observation:** Notice how the mean is pulled towards outliers, while the median remains central. The mode shows the most frequent values.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("How to calculate Mean, Median, Mode in Python")
        st.markdown("Here's how you can calculate these measures using `numpy` and `collections.Counter` (or `scipy.stats.mode`).")
        st.code("""
import numpy as np
from collections import Counter
import pandas as pd # For convenience with mode sometimes

# Example data
data = [1, 2, 2, 3, 4, 5, 10, 100] # Try with [1, 1, 2, 2, 3] for multiple modes, or [1, 2, 3] for no distinct mode

# --- Calculate Mean ---
mean_val = np.mean(data)
print(f"Mean: {mean_val:.2f}")

# --- Calculate Median ---
median_val = np.median(data)
print(f"Median: {median_val:.2f}")

# --- Calculate Mode(s) ---
# Using collections.Counter for robustness with multiple modes
counts = Counter(data)
if not counts: # Handle empty data case
    print("Mode: No data provided.")
else:
    max_count = max(counts.values())
    mode_vals = [key for key, value in counts.items() if value == max_count]

    if max_count == 1 and len(data) > 1: # All values unique, no distinct mode
        print("Mode: No distinct mode (all values unique)")
    else:
        print(f"Mode(s): {mode_vals}")

# Alternative for mode (often returns only one mode, or needs special handling for multiple)
# from scipy.stats import mode
# mode_result = mode(data)
# print(f"Mode (scipy): {mode_result.mode} (count: {mode_result.count})")
        """, language="python")
        st.markdown("You can copy and paste this code into a Python environment (like a Jupyter Notebook or a local `.py` file) and run it.")

    # 6. Practical Task
    with st.expander("🎯 Practice Task", expanded=False):
        st.subheader("Calculate Measures of Central Tendency")
        st.markdown("Your task is to calculate the mean, median, and mode for the following set of numbers. Enter your answers below.")
        task_data_mmm = [15, 18, 20, 18, 22, 25, 19]
        st.code(f"Numbers: {task_data_mmm}", language="python")

        col1_task, col2_task, col3_task = st.columns(3)
        user_mean = col1_task.number_input("Your Mean:", format="%.2f", key='task_mean_input')
        user_median = col2_task.number_input("Your Median:", format="%.2f", key='task_median_input')
        user_mode = col3_task.text_input("Your Mode(s) (comma-separated if multiple):", key='task_mode_input')

        if st.button("Check My Answers - Mean, Median, Mode"):
            correct_mean = np.mean(task_data_mmm)
            correct_median = np.median(task_data_mmm)

            counts_task = Counter(task_data_mmm)
            max_count_task = max(counts_task.values())
            correct_mode_vals = [val for val, count in counts_task.items() if count == max_count_task]

            mean_feedback = "Incorrect."
            if abs(user_mean - correct_mean) < 0.01: # Allow for slight floating point inaccuracies
                mean_feedback = "Correct!"
            st.markdown(f"**Mean:** Your answer: `{user_mean:.2f}`, Correct: `{correct_mean:.2f}`. **{mean_feedback}**")

            median_feedback = "Incorrect."
            if abs(user_median - correct_median) < 0.01:
                median_feedback = "Correct!"
            st.markdown(f"**Median:** Your answer: `{user_median:.2f}`, Correct: `{correct_median:.2f}`. **{median_feedback}**")

            # Compare modes (handle multiple and parsing user input)
            user_mode_list = [float(x.strip()) for x in user_mode.split(',') if x.strip()] if user_mode else []
            user_mode_set = set(user_mode_list)
            correct_mode_set = set(correct_mode_vals)

            mode_feedback = "Incorrect."
            if user_mode_set == correct_mode_set:
                mode_feedback = "Correct!"
            elif not user_mode_set and not correct_mode_set: # Both empty (no mode)
                 mode_feedback = "Correct! (No distinct mode)"
            st.markdown(f"**Mode(s):** Your answer: `{user_mode}`, Correct: `{', '.join([str(m) for m in correct_mode_vals])}`. **{mode_feedback}**")

            if mean_feedback == "Correct!" and median_feedback == "Correct!" and mode_feedback.startswith("Correct!"):
                st.balloons()
                st.success("Great job! You've mastered Mean, Median, and Mode!")
            else:
                st.warning("Keep trying! Review the definitions and equations.")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: How They Work (Step-by-Step)", expanded=False):
        st.subheader("Breaking Down the Calculations")
        st.markdown("""
        While these are foundational concepts, understanding their step-by-step computation reinforces the logic:

        #### **Mean Calculation Steps:**
        1.  **Summation:** Add every single number in your dataset together.
            * *Example:* For `[5, 7, 8, 9, 20]`, Sum = `5 + 7 + 8 + 9 + 20 = 49`.
        2.  **Count:** Determine the total number of values in your dataset.
            * *Example:* For `[5, 7, 8, 9, 20]`, Count = `5`.
        3.  **Division:** Divide the sum by the count.
            * *Example:* Mean = `49 / 5 = 9.8`.

        #### **Median Calculation Steps:**
        1.  **Sort Data:** Arrange all numbers in your dataset in ascending order (from smallest to largest).
            * *Example:* `[20, 9, 7, 5, 8]` becomes `[5, 7, 8, 9, 20]`.
        2.  **Find Middle Position:**
            * **If the count of numbers is Odd:** The median is the single number exactly in the middle.
                * *Example (odd count = 5):* The middle position is the 3rd number ( (5+1)/2 ). In `[5, 7, **8**, 9, 20]`, the median is **8**.
            * **If the count of numbers is Even:** The median is the average of the two numbers in the middle.
                * *Example (even count = 6, e.g., `[5, 7, 8, 9, 10, 20]`):* The middle positions are the 3rd and 4th numbers. (6/2 = 3rd, (6/2)+1 = 4th). In `[5, 7, **8**, **9**, 10, 20]`, the median is `(8 + 9) / 2 = 8.5`.

        #### **Mode Calculation Steps:**
        1.  **Count Frequencies:** Go through your dataset and count how many times each unique number appears.
            * *Example:* For `[1, 2, 2, 3, 4, 5, 10, 100]`
                * 1 appears 1 time
                * 2 appears 2 times
                * 3 appears 1 time
                * 4 appears 1 time
                * 5 appears 1 time
                * 10 appears 1 time
                * 100 appears 1 time
        2.  **Identify Highest Frequency:** Find the highest count among all the frequencies.
            * *Example:* The highest count is `2` (for the number `2`).
        3.  **Select Corresponding Values:** The number(s) that correspond to this highest frequency are the mode(s).
            * *Example:* The number `2` is the mode. (If `10` also appeared twice, then `2` and `10` would both be modes).
        """)
        st.info("💡 Understanding these basic steps is crucial for building intuition for more complex algorithms!")


# (Continue from the previous code chunk)

# --- Topic Content Functions (continued) ---

# ... (Previous functions: show_introduction(), topic_mean_median_mode()) ...

def topic_linear_regression():
    st.header("📈 Linear Regression")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Linear Regression?")
        st.markdown("""
        **Linear Regression** is a fundamental supervised machine learning algorithm used for predicting a continuous output variable (often called the **dependent variable** or target, usually denoted as `y`) based on one or more input features (called **independent variables** or predictors, usually `x`).

        The goal is to find the "best-fit" straight line that describes the relationship between the input(s) and the output. Once this line is found, you can use it to predict the output for new, unseen input values.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're trying to predict a student's final exam score based on the number of hours they spent studying.
        * **Input (x):** Hours studied (e.g., 2, 5, 10 hours)
        * **Output (y):** Final exam score (e.g., 60, 75, 90 points)

        Linear regression would try to find a straight line (e.g., Score = 5 * Hours + 50) that best represents this relationship. Once you have this line, if a new student studies 7 hours, you can predict their score.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Mathematical Formula (Simple Linear Regression)")
        st.markdown("""
        For simple linear regression (one independent variable), the equation of the line is:
        """)
        latex_equation(r'y = \beta_0 + \beta_1 x + \epsilon')
        st.markdown("""
        * $y$: The predicted value of the dependent variable.
        * $x$: The independent variable (input feature).
        * $\beta_0$ (Beta-naught): The **y-intercept**. This is the predicted value of `y` when `x` is 0.
        * $\beta_1$ (Beta-one): The **slope** of the regression line. It represents how much `y` is expected to change for every one-unit increase in `x`.
        * $\epsilon$ (Epsilon): The **error term** or residual. It represents the difference between the actual `y` value and the `y` value predicted by the line. In practice, we aim to minimize this error.
        """)
        st.markdown("""
        The process of "finding the best-fit line" typically involves minimizing the **Sum of Squared Errors (SSE)** or **Residual Sum of Squares (RSS)**.
        """)
        latex_equation(r'SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2')
        st.markdown("""
        * $y_i$: The actual value of the dependent variable for the $i^{th}$ data point.
        * $\hat{y}_i$ (y-hat): The predicted value of the dependent variable for the $i^{th}$ data point (i.e., $\beta_0 + \beta_1 x_i$).
        * The goal is to find $\beta_0$ and $\beta_1$ that make this sum as small as possible.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Build Your Own Regression Model")
        data_source_lr = st.radio("Choose data source:", ("Generate synthetic data", "Use uploaded CSV (if available)"))

        X_lr, y_lr = None, None
        df_lr = None

        if data_source_lr == "Generate synthetic data":
            st.markdown("#### Generate Data Points")
            col1, col2 = st.columns(2)
            num_samples_lr = col1.slider("Number of data points:", 20, 200, 50, key='lr_samples')
            noise_level_lr = col2.slider("Noise level (randomness):", 0.0, 100.0, 20.0, key='lr_noise')

            np.random.seed(0)  # for reproducibility
            X_lr = np.random.rand(num_samples_lr, 1) * 100  # X from 0 to 100
            true_slope = 1.5
            true_intercept = 10
            y_lr = true_slope * X_lr + true_intercept + np.random.randn(num_samples_lr, 1) * noise_level_lr

            df_lr = pd.DataFrame({'X': X_lr.flatten(), 'Y': y_lr.flatten()})
            st.info("Synthetic data generated.")

        elif data_source_lr == "Use uploaded CSV (if available)":
            if global_df is not None:
                st.markdown("#### Select Columns from Uploaded CSV")
                numerical_cols = global_df.select_dtypes(include=np.number).columns.tolist()
                if len(numerical_cols) < 2:
                    st.warning("Uploaded CSV must have at least two numerical columns for regression.")
                else:
                    x_col_lr = st.selectbox("Select X (Independent) Column:", numerical_cols, key='lr_x_col')
                    y_col_lr = st.selectbox("Select Y (Dependent) Column:",
                                            [col for col in numerical_cols if col != x_col_lr], key='lr_y_col')

                    if x_col_lr and y_col_lr:
                        X_lr = global_df[[x_col_lr]].values
                        y_lr = global_df[[y_col_lr]].values
                        df_lr = global_df[[x_col_lr, y_col_lr]].copy()
                        df_lr.columns = ['X', 'Y']  # Standardize column names for plotting
                        st.success(f"Using '{x_col_lr}' as X and '{y_col_lr}' as Y from uploaded CSV.")
                    else:
                        st.warning("Please select both X and Y columns.")
            else:
                st.info("No CSV file uploaded. Please upload one in the sidebar or generate synthetic data.")
                X_lr, y_lr = None, None  # Ensure no data is used if CSV not available/selected

        if X_lr is not None and y_lr is not None and len(X_lr) > 1:
            st.subheader("Linear Regression Fit")
            model = LinearRegression()
            model.fit(X_lr, y_lr)
            y_pred_lr = model.predict(X_lr)

            st.write(f"**Learned Intercept (β₀):** {model.intercept_[0]:.2f}")
            st.write(f"**Learned Slope (β₁):** {model.coef_[0][0]:.2f}")

            # Visualization with Plotly
            fig_lr = px.scatter(df_lr, x='X', y='Y', title="Linear Regression: Data Points and Best-Fit Line",
                                labels={'X': 'Independent Variable (X)', 'Y': 'Dependent Variable (Y)'},
                                opacity=0.7, color_discrete_sequence=['#457b9d'])  # Scatter points

            # Add regression line
            fig_lr.add_trace(go.Scatter(x=df_lr['X'], y=y_pred_lr.flatten(), mode='lines', name='Regression Line',
                                        line=dict(color='#e76f51', width=3)))

            st.plotly_chart(fig_lr, use_container_width=True)
            st.info(
                "💡 **Observation:** The line tries to minimize the distance between itself and all data points. High noise makes the fit less perfect.")

            st.subheader("Make a Prediction")
            prediction_input_lr = st.number_input("Enter a new X value to predict Y:", value=float(df_lr['X'].mean()),
                                                  key='lr_predict_input')
            predicted_value_lr = model.predict([[prediction_input_lr]])[0][0]
            st.write(f"Predicted Y for X = `{prediction_input_lr:.2f}`: **`{predicted_value_lr:.2f}`**")

        else:
            st.warning("Please generate or select valid data to run the Linear Regression experiment.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Linear Regression in Python (scikit-learn)")
        st.markdown("Here's how to perform simple linear regression using the `scikit-learn` library.")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt # For plotting

# 1. Prepare your data
# Example: Synthetic data
X = np.array([10, 20, 30, 40, 50, 60, 70, 80]).reshape(-1, 1) # Feature (hours studied)
y = np.array([25, 40, 55, 65, 70, 80, 85, 95]) # Target (exam scores)

# If using a DataFrame:
# df = pd.DataFrame({'Hours_Studied': [10, 20, ...], 'Exam_Score': [25, 40, ...]})
# X = df[['Hours_Studied']].values
# y = df['Exam_Score'].values

# 2. Create a Linear Regression model instance
model = LinearRegression()

# 3. Fit the model to your data (training)
model.fit(X, y)

# 4. Get the learned coefficients
intercept = model.intercept_
slope = model.coef_[0] # For simple linear regression, it's the first element of coef_ array

print(f"Intercept (β₀): {intercept:.2f}")
print(f"Slope (β₁): {slope:.2f}")

# 5. Make predictions
new_X = np.array([[75]]) # Predict score for 75 hours studied
predicted_y = model.predict(new_X)
print(f"Predicted score for 75 hours: {predicted_y[0]:.2f}")

# 6. Visualize the results (optional, but good for understanding)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X (Independent Variable)')
plt.ylabel('Y (Dependent Variable)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
        """, language="python")
        st.markdown(
            "This code first prepares sample data, then uses `LinearRegression` to fit a model, extract its parameters, and make a prediction. The plotting part visualizes the original data and the learned regression line.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Predict Sales Based on Advertising Spend")
        st.markdown("""
        You have historical data showing how much a company spent on advertising (in thousands of dollars) and their corresponding sales (in thousands of units).

        **Data Points:**
        * Ad Spend (X): `[10, 15, 20, 25, 30]`
        * Sales (Y): `[30, 40, 45, 50, 60]`

        Your task is to:
        1.  Mentally or manually estimate the relationship (slope and intercept).
        2.  Use the concept of linear regression (or the code above as reference) to predict sales if the company spends **35** thousand dollars on advertising.
        """)

        task_X_lr = np.array([10, 15, 20, 25, 30]).reshape(-1, 1)
        task_y_lr = np.array([30, 40, 45, 50, 60])

        task_model = LinearRegression()
        task_model.fit(task_X_lr, task_y_lr)
        correct_prediction = task_model.predict([[35]])[0]

        user_prediction_lr = st.number_input(
            "What are the predicted sales (in thousands of units) for an ad spend of 35?", format="%.2f",
            key='lr_task_prediction')

        if st.button("Check Prediction - Linear Regression"):
            if abs(user_prediction_lr - correct_prediction) < 0.5:  # Allow for slight approximation
                st.success(
                    f"Correct! The predicted sales for $35k ad spend are approximately `{correct_prediction:.2f}` thousand units.")
                st.balloons()
            else:
                st.warning(
                    f"Not quite. The correct prediction is around `{correct_prediction:.2f}`. Remember to fit the model and then use it for prediction.")
                st.info("Hint: Use the `model.predict()` method after fitting.")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: How Linear Regression Learns (Conceptually)", expanded=False):
        st.subheader("Minimizing Errors: The Idea Behind Learning the Line")
        st.markdown("""
        How does the algorithm find the "best" line? It's not just drawing any line. Linear Regression tries to find the line that minimizes the sum of the squared distances (errors) from all data points to the line itself. This is often done using a method called **Ordinary Least Squares (OLS)** or, for larger datasets, an iterative optimization algorithm like **Gradient Descent**.

        #### **Simplified Gradient Descent (Conceptual Steps):**
        Imagine you're trying to find the lowest point in a valley (this lowest point represents the best-fit line where errors are minimal).

        1.  **Start Anywhere:** The algorithm starts by picking a random line (random $\beta_0$ and $\beta_1$ values). This is like dropping a ball at a random spot in the valley.
        2.  **Measure the Slope (Gradient):** It then calculates how "steep" the error surface is at that point. This "steepness" (gradient) tells it which way is downhill towards the minimum error.
            * *Analogy:* The ball "feels" the slope of the ground.
        3.  **Take a Step Downhill:** It takes a small step in the direction of the steepest descent (the direction where the error decreases most rapidly).
            * *Analogy:* The ball rolls a little bit downhill.
        4.  **Repeat:** Steps 2 and 3 are repeated many times. With each step, the line gets a little closer to minimizing the total error.
            * *Analogy:* The ball keeps rolling downhill, gradually approaching the very bottom of the valley.
        5.  **Converge:** Eventually, the changes become very small, meaning it has reached (or is very close to) the lowest point where the errors are minimized. This is where the "best-fit" line is found.

        This iterative process allows the model to "learn" the optimal slope and intercept that best describes the relationship in the data.
        """)
        st.image("https://i.ibb.co/3s65W2x/gradient-descent.png",
                 caption="Conceptual illustration of Gradient Descent approaching a minimum.",
                 use_column_width=True)  # Placeholder for GD
        st.info(
            "💡 **Key takeaway:** The algorithm doesn't just guess; it systematically adjusts the line's parameters to reduce the errors until it finds the optimal fit.")


def topic_train_test_split():
    st.header("✂️ Train/Test Split")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Train/Test Split?")
        st.markdown("""
        **Train/Test Split** is a fundamental technique in machine learning used to evaluate the performance of a model on unseen data. You split your dataset into two main parts:

        * **Training Set:** The larger portion of the data used to train (or "teach") your machine learning model. The model learns patterns and relationships from this data.
        * **Test Set:** A smaller, separate portion of the data that the model has **never seen** during training. This set is used to evaluate how well your trained model generalizes to new, real-world data.

        This split helps in identifying if your model is **overfitting** (performing well on training data but poorly on new data) or **underfitting** (performing poorly on both).
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're studying for an important exam.

        * **Training Set:** This is like the practice problems, homework assignments, and textbook examples you work through to learn the material. You study these extensively.
        * **Test Set:** This is the actual exam. It contains new questions you haven't seen before. If you only studied the exact questions from the practice set (and memorized answers), you might do well on those but fail the real exam (overfitting). If you understood the concepts from the practice, you'll do well on the new exam questions (good generalization).

        The Train/Test split ensures your "exam" (test set) is fair and truly measures your model's understanding of the underlying concepts, not just its ability to memorize the training data.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Conceptual Split Proportion")
        st.markdown("""
        While there isn't a complex mathematical formula, the concept revolves around proportions:
        """)
        latex_equation(r'\text{Total Samples} = \text{Training Samples} + \text{Test Samples}')
        latex_equation(r'\text{Test Size Ratio} = \frac{\text{Test Samples}}{\text{Total Samples}}')
        latex_equation(
            r'\text{Train Size Ratio} = \frac{\text{Training Samples}}{\text{Total Samples}} = 1 - \text{Test Size Ratio}')
        st.markdown("""
        * Common split ratios are 70/30, 80/20, or 75/25 (training/testing).
        * **`random_state`**: A crucial parameter. When splitting data, a random shuffle occurs. `random_state` ensures that the shuffle is the same every time you run your code, making your results reproducible. If you don't set it, you'll get a different split (and potentially different evaluation results) each time.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("See How Data Gets Split")

        col1, col2 = st.columns(2)
        num_samples_tts = col1.slider("Number of data points for demonstration:", 50, 500, 100, key='tts_samples')
        test_size_ratio = col2.slider("Test Set Size Ratio (e.g., 0.2 for 20% test, 80% train):", 0.1, 0.5, 0.2, 0.05,
                                      key='tts_test_size')

        # Generate some synthetic data for demonstration
        np.random.seed(42)  # For consistent data generation
        X_tts = np.random.rand(num_samples_tts, 2) * 100  # 2 features for plotting
        y_tts = (X_tts[:, 0] + X_tts[:, 1] + np.random.randn(num_samples_tts) * 10 > 100).astype(
            int)  # Simple classification target

        df_tts = pd.DataFrame({'Feature1': X_tts[:, 0], 'Feature2': X_tts[:, 1], 'Target': y_tts})

        st.markdown(f"**Original Dataset Size:** {num_samples_tts} samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X_tts, y_tts, test_size=test_size_ratio, random_state=42, stratify=y_tts
            # Stratify helps maintain target distribution
        )

        st.markdown(f"**Training Set Size:** {len(X_train)} samples ({len(X_train) / num_samples_tts:.1%})")
        st.markdown(f"**Test Set Size:** {len(X_test)} samples ({len(X_test) / num_samples_tts:.1%})")

        # Prepare data for Plotly visualization
        df_train = pd.DataFrame({'Feature1': X_train[:, 0], 'Feature2': X_train[:, 1], 'Set': 'Train'})
        df_test = pd.DataFrame({'Feature1': X_test[:, 0], 'Feature2': X_test[:, 1], 'Set': 'Test'})
        df_combined = pd.concat([df_train, df_test])

        fig_tts = px.scatter(df_combined, x="Feature1", y="Feature2", color="Set",
                             title=f"Train/Test Split Visualization (Test Size: {test_size_ratio * 100:.0f}%)",
                             color_discrete_map={'Train': '#2a9d8f',
                                                 'Test': '#e76f51'})  # Green for train, red for test

        st.plotly_chart(fig_tts, use_container_width=True)
        st.info(
            "💡 **Observation:** Notice how the data points are randomly assigned to either the 'Train' or 'Test' set based on your chosen ratio.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Train/Test Split in Python (scikit-learn)")
        st.markdown("The `train_test_split` function from `sklearn.model_selection` is the standard way to do this.")
        st.code("""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd # For loading data

# 1. Load your dataset (X = features, y = target)
# Example: Using a dummy dataset
X = np.arange(1, 101).reshape(-1, 1) # Features (e.g., 100 data points with one feature)
y = np.random.randint(0, 2, 100)    # Binary target (0 or 1)

# If you had a DataFrame:
# df = pd.read_csv('your_data.csv')
# X = df[['feature1', 'feature2']].values # Select feature columns
# y = df['target_column'].values        # Select target column

# 2. Perform the split
# test_size: The proportion of the dataset to include in the test split (e.g., 0.2 for 20%)
# random_state: Ensures reproducibility. Use any integer (e.g., 42).
# stratify: (Optional but recommended for classification) Ensures that the proportions of target classes
#           are preserved in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Original X shape: {X.shape}")
print(f"Training X shape: {X_train.shape}")
print(f"Test X shape: {X_test.shape}")
print(f"Original y shape: {y.shape}")
print(f"Training y shape: {y_train.shape}")
print(f"Test y shape: {y_test.shape}")

# You can now use X_train, y_train to train your model,
# and X_test, y_test to evaluate its performance.
        """, language="python")
        st.markdown(
            "This code demonstrates how to split a dataset into training and testing sets, printing their shapes to confirm the split.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Split a Dataset")
        st.markdown("""
        You have a dataset with 50 samples.
        Your goal is to split this dataset into a training set and a test set using a `test_size` of `0.3` (30% for testing).

        Calculate:
        1.  How many samples will be in the training set?
        2.  How many samples will be in the test set?
        """)

        col1_task_tts, col2_task_tts = st.columns(2)
        user_train_count = col1_task_tts.number_input("Number of samples in Training Set:", min_value=0,
                                                      key='tts_task_train')
        user_test_count = col2_task_tts.number_input("Number of samples in Test Set:", min_value=0, key='tts_task_test')

        if st.button("Check My Split - Train/Test"):
            total_samples = 50
            test_ratio = 0.3

            correct_test_count = int(total_samples * test_ratio)
            correct_train_count = total_samples - correct_test_count

            train_feedback = "Incorrect."
            if user_train_count == correct_train_count:
                train_feedback = "Correct!"
            st.markdown(
                f"**Training Samples:** Your answer: `{user_train_count}`, Correct: `{correct_train_count}`. **{train_feedback}**")

            test_feedback = "Incorrect."
            if user_test_count == correct_test_count:
                test_feedback = "Correct!"
            st.markdown(
                f"**Test Samples:** Your answer: `{user_test_count}`, Correct: `{correct_test_count}`. **{test_feedback}**")

            if train_feedback == "Correct!" and test_feedback == "Correct!":
                st.balloons()
                st.success("Excellent! You understand how to split data for model evaluation.")
            else:
                st.warning("Review the `test_size` calculation. Remember it's a proportion of the total.")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: Why and How it Works (Conceptual)", expanded=False):
        st.subheader("The Importance of Unseen Data")
        st.markdown("""
        The core idea behind train/test split is to simulate how your model would perform on **new, unseen data** in the real world.

        #### **Why Split?**
        * **Prevent Overfitting:** If you train and test your model on the *same* data, the model might just memorize the training examples (like memorizing answers to practice questions without understanding the concepts). This model would fail badly on new data. The test set acts as an independent "exam" to catch this.
        * **Generalization:** A good model is one that generalizes well, meaning it performs well on data it hasn't seen before. The test set provides an unbiased estimate of this generalization ability.

        #### **How the Split Works (Conceptually):**
        1.  **Shuffling:** The first step is usually to randomly shuffle the entire dataset. This ensures that the training and test sets are representative of the overall data distribution and prevents any ordering bias.
        2.  **Proportionate Selection:** Based on the `test_size` (e.g., 20%), a random subset of data points is selected for the test set. The remaining data points form the training set.
        3.  **Stratification (for classification):** If you have a classification problem (e.g., predicting 'yes' or 'no'), `stratify=y` ensures that if your original data has, say, 70% 'yes' and 30% 'no', then both your training and test sets will also maintain roughly these proportions. This is crucial for balanced evaluation, especially with imbalanced datasets.

        Essentially, `train_test_split` performs a fair and reproducible random drawing of samples for these two critical roles.
        """)
        st.info("💡 **Remember:** Never train your model on the test set! The test set is only for final evaluation.")


def topic_k_means_clustering():
    st.header("⚪ K-means Clustering")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is K-means Clustering?")
        st.markdown("""
        **K-means Clustering** is a popular **unsupervised machine learning algorithm** used for grouping similar data points into clusters. "Unsupervised" means it learns patterns from data without needing pre-labeled outcomes (targets).

        The goal of K-means is to partition $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype of the cluster.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're a librarian with thousands of books, but they're all just dumped in a big pile. You want to organize them into `k` (e.g., 5) different thematic sections (clusters) without knowing beforehand what each book is about.

        * **K-means** would help you sort them by finding groups of books that are "similar" (e.g., based on keywords, genre, authors). It doesn't tell you "this is a Sci-Fi cluster," but rather "these 200 books are similar to each other, and these 150 books are similar to each other," forming distinct piles. You then interpret what each pile (cluster) represents.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Mathematical Objective (Minimizing Inertia)")
        st.markdown("""
        K-means aims to minimize the **inertia** (also known as "within-cluster sum of squares" or WCSS). This measures how close data points are to the centroid of their assigned cluster. A lower inertia means denser clusters.
        """)
        latex_equation(r'J = \sum_{j=1}^{k} \sum_{i \in S_j} ||x_i - \mu_j||^2')
        st.markdown("""
        * $J$: The inertia, which we want to minimize.
        * $k$: The number of clusters (chosen by you).
        * $S_j$: The set of data points belonging to cluster $j$.
        * $x_i$: A data point.
        * $\mu_j$: The centroid (mean) of cluster $j$.
        * $||x_i - \mu_j||^2$: The squared Euclidean distance between data point $x_i$ and its assigned cluster centroid $\mu_j$.
            * **Euclidean Distance** between two points $(p_1, p_2)$ and $(q_1, q_2)$ in 2D is: $\sqrt{(q_1-p_1)^2 + (q_2-p_2)^2}$. The squared distance removes the square root.
        """)
        st.markdown("""
        **How $k$ (Number of Clusters) is Chosen: The Elbow Method**
        Since K-means requires you to choose $k$ beforehand, a common technique is the **Elbow Method**. You run K-means for a range of $k$ values and plot the inertia for each. The "elbow" point on the plot (where the decrease in inertia starts to slow down significantly) often indicates the optimal $k$.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Experiment with K-means Clustering")

        col1_kmeans, col2_kmeans = st.columns(2)
        num_samples_kmeans = col1_kmeans.slider("Number of data points:", 100, 500, 200, key='kmeans_samples')
        n_clusters_input = col2_kmeans.slider("Number of Clusters (K):", 2, 10, 3, key='kmeans_k')

        # Generate synthetic clusterable data
        X_kmeans, y_true_kmeans = sk.datasets.make_blobs(
            n_samples=num_samples_kmeans,
            n_features=2,
            centers=n_clusters_input + 1,  # Generate a bit more centers than K to make it challenging
            cluster_std=1.0,
            random_state=42
        )
        df_kmeans = pd.DataFrame(X_kmeans, columns=['Feature1', 'Feature2'])

        st.markdown(f"**Running K-means with K = {n_clusters_input}**")

        # Scale data for better clustering performance
        scaler_kmeans = StandardScaler()
        X_scaled_kmeans = scaler_kmeans.fit_transform(X_kmeans)

        kmeans_model = KMeans(n_clusters=n_clusters_input, random_state=42, n_init=10)  # n_init for robustness
        kmeans_model.fit(X_scaled_kmeans)
        cluster_labels = kmeans_model.labels_
        cluster_centers_scaled = kmeans_model.cluster_centers_

        # Inverse transform centroids for plotting on original scale
        cluster_centers = scaler_kmeans.inverse_transform(cluster_centers_scaled)

        df_kmeans['Cluster'] = cluster_labels
        df_kmeans['Cluster'] = df_kmeans['Cluster'].astype(str)  # For categorical coloring in Plotly

        # Plotly Scatter Plot for clusters
        fig_kmeans = px.scatter(df_kmeans, x='Feature1', y='Feature2', color='Cluster',
                                title=f"K-means Clustering (K={n_clusters_input})",
                                hover_data=['Cluster'],
                                color_discrete_sequence=px.colors.qualitative.Pastel)  # Nice color palette

        # Add centroids
        fig_kmeans.add_trace(go.Scatter(x=cluster_centers[:, 0], y=cluster_centers[:, 1],
                                        mode='markers',
                                        marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
                                        name='Centroids',
                                        showlegend=True))
        st.plotly_chart(fig_kmeans, use_container_width=True)
        st.info(
            "💡 **Observation:** Points of similar characteristics are grouped into the same cluster. The 'X' marks the center (centroid) of each cluster.")

        st.subheader("The Elbow Method: Finding Optimal K")
        st.markdown("Run K-means with different K values and observe the inertia (WCSS).")

        max_k_elbow = st.slider("Max K for Elbow Method:", 2, 15, 10, key='elbow_max_k')
        inertia_values = []
        k_range = range(1, max_k_elbow + 1)

        # Progress bar for Elbow method calculation
        progress_bar = st.progress(0)
        status_text = st.empty()

        for k in k_range:
            progress_bar.progress(k / max_k_elbow)
            status_text.text(f"Calculating for K = {k}...")
            try:
                kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_elbow.fit(X_scaled_kmeans)
                inertia_values.append(kmeans_elbow.inertia_)
            except ValueError:  # Handle case where k=1 might not always yield error but results are trivial
                inertia_values.append(0)  # Or a very high value if 1-cluster is treated as invalid for elbow logic

        progress_bar.empty()
        status_text.empty()

        df_elbow = pd.DataFrame({'K': list(k_range), 'Inertia': inertia_values})

        fig_elbow = px.line(df_elbow, x='K', y='Inertia', markers=True,
                            title="Elbow Method for Optimal K",
                            labels={'K': 'Number of Clusters (K)', 'Inertia': 'Within-cluster Sum of Squares (WCSS)'},
                            color_discrete_sequence=['#264653'])  # Darker blue for line
        fig_elbow.update_layout(xaxis_tickmode='linear', xaxis_dtick=1)  # Ensure integer ticks on K axis
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.info(
            "💡 **Observation:** Look for the 'elbow' point where the curve bends significantly. This often suggests the optimal number of clusters.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing K-means Clustering in Python (scikit-learn)")
        st.markdown("Here's how to apply K-means and use the elbow method.")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Prepare your data
# Example: Generate synthetic 2D data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
# X is a NumPy array of data points

# If using a DataFrame:
# df = pd.read_csv('your_unlabeled_data.csv')
# X = df[['feature1', 'feature2']].values # Select numerical features for clustering

# 2. (Optional but Recommended) Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Choose the number of clusters (K)
n_clusters = 4 # Let's say we want 4 clusters

# 4. Create a KMeans model instance
# n_init='auto' or a number: how many times to run k-means algorithm with different centroid seeds.
#                            The final results will be the best output of n_init consecutive runs.
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

# 5. Fit the model to the scaled data
kmeans.fit(X_scaled)

# 6. Get cluster assignments (labels) for each data point
labels = kmeans.labels_
print(f"First 10 cluster labels: {labels[:10]}")

# 7. Get the coordinates of the cluster centroids (on scaled data)
centroids_scaled = kmeans.cluster_centers_
print(f"Cluster centroids (scaled):\\n{centroids_scaled}")

# If you want centroids on original scale:
centroids_original_scale = scaler.inverse_transform(centroids_scaled)
print(f"Cluster centroids (original scale):\\n{centroids_original_scale}")

# 8. Visualize results (using matplotlib for simplicity here)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8, label='Data Points')
plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.title(f'K-Means Clustering with K={n_clusters}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# --- Elbow Method Example ---
print("\\n--- Elbow Method ---")
inertia = []
for i in range(1, 11): # Test K from 1 to 10
    kmeans_elbow = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans_elbow.fit(X_scaled)
    inertia.append(kmeans_elbow.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.show()
        """, language="python")
        st.markdown(
            "This code generates synthetic data, scales it, applies K-means, and then visualizes the clustered data and centroids. It also demonstrates how to implement the Elbow Method to help determine the best K.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Identify the Optimal K for a Dataset")
        st.markdown("""
        You are given a conceptual Elbow Method plot for a dataset. Based on the plot's shape, what would you suggest as the optimal number of clusters (K)?

        *Examine the plot and find the 'elbow' point where the descent significantly slows down.*
        """)

        # Generate a conceptual elbow plot
        k_values = np.arange(1, 11)
        # Create inertia values that clearly show an elbow
        inertia_task = [2500, 1200, 500, 250, 150, 100, 80, 70, 65, 60]  # Elbow around K=3 or 4

        df_task_elbow = pd.DataFrame({'K': k_values, 'Inertia': inertia_task})

        fig_task_elbow = px.line(df_task_elbow, x='K', y='Inertia', markers=True,
                                 title="Task: Conceptual Elbow Method Plot",
                                 labels={'K': 'Number of Clusters (K)', 'Inertia': 'WCSS'})
        fig_task_elbow.update_layout(xaxis_tickmode='linear', xaxis_dtick=1)
        st.plotly_chart(fig_task_elbow, use_container_width=True)

        user_optimal_k = st.number_input("What is your suggested optimal K value (integer)?", min_value=1, max_value=10,
                                         value=1, step=1, key='kmeans_task_k_input')

        if st.button("Check Optimal K"):
            # Based on the inertia_task, 3 or 4 would be reasonable
            if user_optimal_k in [3, 4]:
                st.success(
                    f"Correct! For this plot, K={user_optimal_k} is a very good choice for the 'elbow' point. This is where adding more clusters provides diminishing returns in reducing inertia.")
                st.balloons()
            else:
                st.warning(
                    f"Not quite. The 'elbow' is where the curve's steepness significantly decreases. For this plot, common choices are 3 or 4. Try to visually identify that bend!")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: How K-means Works (Step-by-Step with Example)", expanded=False):
        st.subheader("The Iterative Process of K-means")
        st.markdown("""
        K-means is an iterative algorithm that works in a few repeating steps until clusters no longer change significantly.

        **Let's walk through a simplified example with 6 data points and K=2 clusters.**

        **Data Points (2D):**
        A=(1,1), B=(1.5,2), C=(3,4), D=(5,7), E=(3.5,5), F=(4.5,5)

        ---

        #### **Step 1: Initialization**
        * **Randomly place K centroids.**
            * *Example:* Let's say our two initial centroids are:
                * Centroid 1 ($\mu_1$): (1.5, 2) (same as point B)
                * Centroid 2 ($\mu_2$): (4.5, 5) (same as point F)
        """)

        # Conceptual Plot 1
        points_step1 = pd.DataFrame({
            'x': [1, 1.5, 3, 5, 3.5, 4.5],
            'y': [1, 2, 4, 7, 5, 5],
            'Label': ['A', 'B', 'C', 'D', 'E', 'F']
        })
        centroids_step1 = pd.DataFrame({
            'x': [1.5, 4.5],
            'y': [2, 5],
            'Label': ['Centroid 1', 'Centroid 2']
        })
        fig_step1 = px.scatter(points_step1, x='x', y='y', text='Label', title="Step 1: Initial Centroids")
        fig_step1.add_trace(go.Scatter(x=centroids_step1['x'], y=centroids_step1['y'], mode='markers+text',
                                       marker=dict(symbol='x', size=15, color='red'),
                                       text=centroids_step1['Label'], textposition="top center",
                                       name='Centroids'))
        st.plotly_chart(fig_step1, use_container_width=True)
        st.info("💡 **Visualization 1:** Initial placement of two random centroids.")

        st.markdown("""
        ---
        #### **Step 2: Assignment (E-step - Expectation)**
        * **Assign each data point to its closest centroid.** Calculate the distance from each point to every centroid.
            * *Example:*
                * Point A (1,1) is closer to $\mu_1$(1.5,2).
                * Point B (1.5,2) is its own centroid $\mu_1$.
                * Point C (3,4) is closer to $\mu_1$(1.5,2).
                * Point D (5,7) is closer to $\mu_2$(4.5,5).
                * Point E (3.5,5) is closer to $\mu_2$(4.5,5).
                * Point F (4.5,5) is its own centroid $\mu_2$.
        """)
        # Conceptual Plot 2
        cluster_assignment_step2 = {
            'x': [1, 1.5, 3, 5, 3.5, 4.5],
            'y': [1, 2, 4, 7, 5, 5],
            'Label': ['A', 'B', 'C', 'D', 'E', 'F'],
            'Cluster': ['Cluster 1', 'Cluster 1', 'Cluster 1', 'Cluster 2', 'Cluster 2', 'Cluster 2']
        }
        df_step2 = pd.DataFrame(cluster_assignment_step2)
        fig_step2 = px.scatter(df_step2, x='x', y='y', color='Cluster', text='Label',
                               title="Step 2: Points Assigned to Closest Centroids",
                               color_discrete_map={'Cluster 1': '#a8dadc', 'Cluster 2': '#f77f00'})
        fig_step2.add_trace(go.Scatter(x=centroids_step1['x'], y=centroids_step1['y'], mode='markers+text',
                                       marker=dict(symbol='x', size=15, color='black'),
                                       text=centroids_step1['Label'], textposition="top center",
                                       name='Centroids'))
        st.plotly_chart(fig_step2, use_container_width=True)
        st.info("💡 **Visualization 2:** Data points are colored based on their closest initial centroid.")

        st.markdown("""
        ---
        #### **Step 3: Update (M-step - Maximization)**
        * **Recalculate the new position of each centroid** by taking the mean of all data points assigned to that cluster.
            * *Example:*
                * New $\mu_1$: Mean of (A, B, C) = ((1+1.5+3)/3, (1+2+4)/3) = (1.83, 2.33)
                * New $\mu_2$: Mean of (D, E, F) = ((5+3.5+4.5)/3, (7+5+5)/3) = (4.33, 5.67)
        """)
        # Conceptual Plot 3
        centroids_step3 = pd.DataFrame({
            'x': [1.83, 4.33],
            'y': [2.33, 5.67],
            'Label': ['New Centroid 1', 'New Centroid 2']
        })
        fig_step3 = px.scatter(df_step2, x='x', y='y', color='Cluster', text='Label',
                               title="Step 3: Centroids Move to Cluster Means",
                               color_discrete_map={'Cluster 1': '#a8dadc', 'Cluster 2': '#f77f00'})
        fig_step3.add_trace(go.Scatter(x=centroids_step3['x'], y=centroids_step3['y'], mode='markers+text',
                                       marker=dict(symbol='x', size=15, color='black'),
                                       text=centroids_step3['Label'], textposition="top center",
                                       name='New Centroids'))
        st.plotly_chart(fig_step3, use_container_width=True)
        st.info("💡 **Visualization 3:** The centroids have moved to the center of their assigned points.")

        st.markdown("""
        ---
        #### **Step 4: Repeat (Steps 2 & 3)**
        * Repeat the assignment and update steps until:
            * Centroids no longer move significantly.
            * Cluster assignments no longer change.
            * A maximum number of iterations is reached.

        This iterative refinement ensures that the clusters become more compact and distinct with each step.
        """)
        st.title("Interactive 3D K-Means Clustering Visualization")

        # Sidebar Inputs
        n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, step=50)
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
        random_state = st.sidebar.slider("Random State", 0, 100, 42)

        # Generate synthetic 3D data
        X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=3, random_state=random_state)

        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)

        # Prepare DataFrame
        df = pd.DataFrame(X, columns=['X', 'Y', 'Z'])
        df['Cluster'] = labels.astype(str)

        # Interactive 3D Scatter Plot
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Cluster',
                            title="3D K-Means Clustering",
                            opacity=0.8,width = 800,height=600)
        fig.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig.update_layout(scene=dict(
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
        ))

        st.plotly_chart(fig, use_container_width=False)
        st.info(
            "💡 **Key takeaway:** K-means is an iterative 'guess and refine' process that aims to minimize the distances within each cluster.")







# (Continue from the previous code chunk)

# --- Topic Content Functions (continued) ---

# ... (Previous functions: show_introduction(), topic_mean_median_mode(),
#      topic_linear_regression(), topic_train_test_split(), topic_k_means_clustering()) ...


def topic_standard_deviation():
    st.header("📏 Standard Deviation")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Standard Deviation?")
        st.markdown("""
        **Standard Deviation (SD)** is a measure of the **amount of variation or dispersion** of a set of values. It tells you, on average, how far each data point is from the mean.

        * **Low Standard Deviation:** Indicates that data points tend to be close to the mean (data is tightly clustered).
        * **High Standard Deviation:** Indicates that data points are spread out over a wider range of values (data is more dispersed).

        It's expressed in the same units as the data itself, making it easy to interpret.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine two groups of students took the same exam:

        * **Group A:** Scores: 70, 71, 70, 69, 70. (Mean = 70). These scores are very close to the mean. **Low Standard Deviation.**
        * **Group B:** Scores: 50, 90, 70, 100, 40. (Mean = 70). These scores are widely spread out. **High Standard Deviation.**

        Standard deviation helps you understand the typical "spread" or "risk" around an average. For example, in finance, a high standard deviation for stock returns means higher volatility (risk).
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Mathematical Formula")
        st.markdown("""
        The formula for the **population standard deviation** (σ - sigma) is:
        """)
        latex_equation(r'\sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}')
        st.markdown("""
        And for the **sample standard deviation** (s), which is more commonly used when you have a sample from a larger population:
        """)
        latex_equation(r's = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}}')
        st.markdown("""
        * $x_i$: Each individual data point.
        * $\mu$ (mu): The population mean.
        * $\bar{x}$ (x-bar): The sample mean.
        * $N$: The total number of data points in the population.
        * $n$: The total number of data points in the sample.
        * $\sum$: Summation (add up all the squared differences).
        * $n-1$: This is called **Bessel's correction**. It's used for sample standard deviation to provide an unbiased estimate of the population standard deviation, especially important for smaller samples.
        """)
        st.markdown("""
        **Variance:** The variance ($\sigma^2$ or $s^2$) is simply the standard deviation squared (i.e., the term under the square root in the formulas above). It measures the average of the squared differences from the mean. Standard deviation is often preferred because it's in the original units of the data.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Explore Data Spread")
        input_method_sd = st.radio("Choose input method:",
                                   ("Enter comma-separated values", "Generate random data with custom spread"))

        data_sd = []
        if input_method_sd == "Enter comma-separated values":
            user_input_sd = st.text_area("Enter numbers separated by commas (e.g., 10, 12, 11, 10, 9, 8)",
                                         "10, 12, 11, 10, 9, 8")
            try:
                data_sd = [float(x.strip()) for x in user_input_sd.split(',') if x.strip()]
                if not data_sd:
                    st.warning("Please enter some numbers for the experiment.")
            except ValueError:
                st.error("Invalid input. Please enter numbers separated by commas.")
        else:  # Generate random data with custom spread
            col1, col2, col3 = st.columns(3)
            num_points_sd = col1.slider("Number of random data points:", 10, 200, 50, key='sd_num_points')
            mean_sd = col2.slider("Mean of data:", 0, 100, 50, key='sd_mean')
            std_dev_input = col3.slider("Standard Deviation (spread):", 1, 30, 10, key='sd_std_dev')

            np.random.seed(42)
            data_sd = np.random.normal(mean_sd, std_dev_input, num_points_sd).tolist()

        if data_sd:
            df_sd = pd.DataFrame({'Values': data_sd})

            mean_val_sd = np.mean(data_sd)
            std_val_sd = np.std(data_sd)  # Default is population std (ddof=0)
            sample_std_val_sd = np.std(data_sd, ddof=1)  # Sample std (ddof=1)

            st.write(
                f"**Input Data ({len(data_sd)} points):** {', '.join([f'{x:.2f}' for x in data_sd[:10]])}...")  # Show first few
            st.write(f"**Calculated Mean:** {mean_val_sd:.2f}")
            st.write(f"**Calculated Population Standard Deviation (σ):** {std_val_sd:.2f}")
            st.write(f"**Calculated Sample Standard Deviation (s):** {sample_std_val_sd:.2f}")
            st.info("💡 **Note:** `np.std()` by default calculates population std. Use `ddof=1` for sample std.")

            st.subheader("Data Distribution and Standard Deviation")
            fig_sd = px.histogram(df_sd, x="Values", nbins=max(1, len(data_sd) // 5),
                                  title="Distribution of Data with Standard Deviation",
                                  opacity=0.7, color_discrete_sequence=['#457b9d'])

            # Add lines for mean and +/- 1, 2, 3 standard deviations
            fig_sd.add_vline(x=mean_val_sd, line_dash="dash", line_color="#e76f51",
                             annotation_text=f"Mean: {mean_val_sd:.2f}", annotation_position="top right")

            for i in range(1, 4):
                fig_sd.add_vline(x=mean_val_sd + i * sample_std_val_sd, line_dash="dot", line_color="green",
                                 annotation_text=f"+{i} SD", annotation_position="top left")
                fig_sd.add_vline(x=mean_val_sd - i * sample_std_val_sd, line_dash="dot", line_color="green",
                                 annotation_text=f"-{i} SD", annotation_position="top right")

            st.plotly_chart(fig_sd, use_container_width=True)
            st.info(
                "💡 **Observation:** The green dotted lines show intervals of 1, 2, and 3 standard deviations from the mean. A larger standard deviation means these lines are farther apart, indicating more spread.")
        else:
            st.warning("Please enter or generate valid data to run the Standard Deviation experiment.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("How to calculate Standard Deviation in Python")
        st.code("""
import numpy as np
import pandas as pd # Just for example data container

data = [10, 12, 11, 10, 9, 8, 15, 5, 20, 0]

# Convert to numpy array for easy calculations
np_data = np.array(data)

# Calculate Mean
mean_val = np.mean(np_data)
print(f"Mean: {mean_val:.2f}")

# Calculate Population Standard Deviation (default ddof=0)
# (Assumes your data is the entire population)
pop_std = np.std(np_data)
print(f"Population Standard Deviation (σ): {pop_std:.2f}")

# Calculate Sample Standard Deviation (ddof=1)
# (Assumes your data is a sample from a larger population)
sample_std = np.std(np_data, ddof=1)
print(f"Sample Standard Deviation (s): {sample_std:.2f}")

# Calculate Variance (std dev squared)
pop_variance = np.var(np_data)
sample_variance = np.var(np_data, ddof=1)
print(f"Population Variance (σ²): {pop_variance:.2f}")
print(f"Sample Variance (s²): {sample_variance:.2f}")
        """, language="python")
        st.markdown(
            "Use `np.std(data, ddof=0)` for population standard deviation and `np.std(data, ddof=1)` for sample standard deviation.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Compare Spreads")
        st.markdown("""
        You have two sets of daily temperature readings (in Celsius). Which set has a higher standard deviation (more variation)?

        * **City A Temperatures:** `[20, 21, 19, 20, 22]`
        * **City B Temperatures:** `[15, 25, 20, 10, 30]`

        Calculate the *sample* standard deviation for each city and determine which one is higher.
        """)

        city_a_temps = np.array([20, 21, 19, 20, 22])
        city_b_temps = np.array([15, 25, 20, 10, 30])

        user_choice_sd_task = st.radio("Which city has a higher standard deviation?", ("City A", "City B"),
                                       key='sd_task_choice')

        if st.button("Check My Answer - Standard Deviation"):
            std_a = np.std(city_a_temps, ddof=1)
            std_b = np.std(city_b_temps, ddof=1)

            st.write(f"Sample Standard Deviation for City A: `{std_a:.2f}`")
            st.write(f"Sample Standard Deviation for City B: `{std_b:.2f}`")

            correct_choice = "City B" if std_b > std_a else "City A" if std_a > std_b else "They are equal"

            if user_choice_sd_task == correct_choice:
                st.success(
                    f"Correct! City B has a higher standard deviation (`{std_b:.2f}`) compared to City A (`{std_a:.2f}`), indicating more temperature variation.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. City B has the higher standard deviation. This means its temperatures are more spread out from the average. Review the concept of spread!")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: Standard Deviation Calculation Steps", expanded=False):
        st.subheader("Manual Calculation of Sample Standard Deviation")
        st.markdown("""
        Let's calculate the sample standard deviation for a small dataset step-by-step: `[2, 4, 4, 4, 5]`

        1.  **Calculate the Mean ($\bar{x}$):**
            * Sum = `2 + 4 + 4 + 4 + 5 = 19`
            * Count ($n$) = `5`
            * Mean ($\bar{x}$) = `19 / 5 = 3.8`

        2.  **Calculate the Deviation from the Mean ($x_i - \bar{x}$):**
            * `2 - 3.8 = -1.8`
            * `4 - 3.8 = 0.2`
            * `4 - 3.8 = 0.2`
            * `4 - 3.8 = 0.2`
            * `5 - 3.8 = 1.2`

        3.  **Square Each Deviation ($(x_i - \bar{x})^2$):**
            * `(-1.8)^2 = 3.24`
            * `(0.2)^2 = 0.04`
            * `(0.2)^2 = 0.04`
            * `(0.2)^2 = 0.04`
            * `(1.2)^2 = 1.44`

        4.  **Sum the Squared Deviations ($\sum (x_i - \bar{x})^2$):**
            * Sum of Squared Deviations = `3.24 + 0.04 + 0.04 + 0.04 + 1.44 = 4.8`

        5.  **Divide by (n-1) (Bessel's Correction) to get Variance ($s^2$):**
            * $n-1 = 5 - 1 = 4$
            * Variance ($s^2$) = `4.8 / 4 = 1.2`

        6.  **Take the Square Root to get Standard Deviation (s):**
            * Standard Deviation (s) = `√1.2 ≈ 1.095`

        This step-by-step process shows how standard deviation quantifies the typical distance of data points from the mean.
        """)
        st.info(
            "💡 **Visualization:** Imagine these steps adjusting the green lines on the histogram until they optimally capture the data's spread.")


def topic_percentiles():
    st.header("💯 Percentiles")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What are Percentiles?")
        st.markdown("""
        A **percentile** is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations falls. For example, the 20th percentile is the value below which 20% of the observations may be found.

        * Percentiles divide a dataset into 100 equal parts.
        * The **median** is the 50th percentile.
        * **Quartiles** are special percentiles:
            * **Q1 (First Quartile):** 25th percentile
            * **Q2 (Second Quartile):** 50th percentile (Median)
            * **Q3 (Third Quartile):** 75th percentile
        """)
        st.markdown("""
        **Daily-life Example:**
        Think about your score on a standardized test:

        * If you scored in the **90th percentile**, it means your score was higher than 90% of the people who took the test.
        * If you scored in the **20th percentile**, it means 80% of the test-takers scored higher than you.

        Percentiles help you understand your relative position within a group, not just your absolute score.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Conceptual Calculation of Percentiles")
        st.markdown("""
        While different methods exist (interpolation, nearest rank), the general idea for finding the value at a certain percentile ($P$) is:

        1.  **Order the data:** Arrange all data points in ascending order.
        2.  **Calculate the index:**
            """)
        latex_equation(r'I = \left( \frac{P}{100} \right) \times N')
        st.markdown("""
        * $I$: The calculated index (position) in the ordered list.
        * $P$: The desired percentile (e.g., 25 for 25th percentile).
        * $N$: The total number of data points.

        3.  **Find the value:**
            * If $I$ is an integer, the percentile value is the average of the value at position $I$ and $I+1$.
            * If $I$ is not an integer, round up to the next whole number, and the percentile value is the value at that position.

        **To find the percentile rank of a specific value ($x$):**
        """)
        latex_equation(
            r'\text{Percentile Rank} = \frac{\text{Number of values below } x + 0.5 \times \text{Number of values equal to } x}{\text{Total Number of values}} \times 100')
        st.markdown("""
        This formula gives the percentage of values in the dataset that are less than or equal to $x$.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Calculate Percentiles")

        col1_perc, col2_perc = st.columns(2)
        input_data_perc_str = col1_perc.text_area("Enter numbers (comma-separated):",
                                                  "10, 15, 20, 25, 30, 35, 40, 45, 50, 55", key='perc_input_data')
        desired_percentile = col2_perc.slider("Desired Percentile (P):", 1, 99, 50, key='perc_slider')

        data_perc = []
        try:
            data_perc = sorted([float(x.strip()) for x in input_data_perc_str.split(',') if x.strip()])
            if not data_perc:
                st.warning("Please enter some numbers.")
        except ValueError:
            st.error("Invalid input. Please enter numbers separated by commas.")

        if data_perc:
            df_perc = pd.DataFrame({'Values': data_perc})

            # Calculate percentile value
            percentile_value = np.percentile(data_perc, desired_percentile)
            st.write(f"**Data:** {data_perc}")
            st.write(f"The **{desired_percentile}th percentile** is: **`{percentile_value:.2f}`**")

            st.markdown("---")
            st.subheader("Find Percentile Rank of a Value")
            value_to_find_rank = st.number_input("Enter a value to find its percentile rank:",
                                                 value=float(np.median(data_perc)), key='perc_value_rank')

            # Use scipy's percentileofscore for consistency and common methods
            rank_percentile = percentileofscore(data_perc, value_to_find_rank,
                                                kind='weak')  # 'weak' means (number of scores less than or equal to the given score) / total number of scores

            st.write(f"The value **`{value_to_find_rank:.2f}`** is at the **`{rank_percentile:.2f}th` percentile**.")

            st.subheader("Visualization: Data Points and Percentile Marker")
            fig_perc = px.scatter(df_perc, x=df_perc.index, y='Values', title="Ordered Data Points with Percentile",
                                  labels={'x': 'Rank (Ordered Index)', 'y': 'Value'},
                                  color_discrete_sequence=['#2a9d8f'])

            # Add line for the calculated percentile value
            fig_perc.add_hline(y=percentile_value, line_dash="dash", line_color="#e76f51",
                               annotation_text=f"{desired_percentile}th Percentile: {percentile_value:.2f}",
                               annotation_position="top right")

            # Highlight the point closest to the percentile value (conceptual)
            # This is tricky as np.percentile interpolates. We can find the closest actual data point.
            closest_idx = (np.abs(np.array(data_perc) - percentile_value)).argmin()
            fig_perc.add_trace(go.Scatter(x=[closest_idx], y=[data_perc[closest_idx]], mode='markers',
                                          marker=dict(symbol='star', size=15, color='purple'),
                                          name='Approx. Percentile Point'))

            st.plotly_chart(fig_perc, use_container_width=True)
            st.info(
                "💡 **Observation:** The red dashed line shows the value below which the specified percentage of data points fall. The purple star marks the closest actual data point to that percentile value.")
        else:
            st.warning("Please enter valid data to run the Percentiles experiment.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("How to calculate Percentiles in Python")
        st.code("""
import numpy as np
from scipy.stats import percentileofscore

data = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] # Already sorted for clarity

# 1. Find the value at a specific percentile (e.g., 25th, 50th, 75th)
# np.percentile(array, percentile_value)
p25 = np.percentile(data, 25)
p50_median = np.percentile(data, 50) # This is the median
p75 = np.percentile(data, 75)

print(f"25th percentile (Q1): {p25}")
print(f"50th percentile (Median/Q2): {p50_median}")
print(f"75th percentile (Q3): {p75}")

# 2. Find the percentile rank of a specific value
value_to_rank = 30
# 'kind' parameter specifies method:
# 'rank': (number of values strictly less than score + 1) / total number of values * 100
# 'weak': (number of values less than or equal to score) / total number of values * 100
# 'strict': (number of values strictly less than score) / total number of values * 100
# 'mean': (strict + weak) / 2
rank_perc_weak = percentileofscore(data, value_to_rank, kind='weak')
print(f"Value {value_to_rank} is at the {rank_perc_weak:.2f}th percentile (kind='weak').")

value_to_rank_new = 32 # A value not in the list
rank_perc_new = percentileofscore(data, value_to_rank_new, kind='mean')
print(f"Value {value_to_rank_new} is at the {rank_perc_new:.2f}th percentile (kind='mean').")

# Interquartile Range (IQR) - spread of the middle 50%
iqr = p75 - p25
print(f"Interquartile Range (IQR): {iqr}")
        """, language="python")
        st.markdown(
            "`numpy.percentile` is used to find the value at a given percentile. `scipy.stats.percentileofscore` finds the percentile rank of a value.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Calculate Test Score Percentile")
        st.markdown("""
        Here are the scores of 10 students on a quiz: `[65, 70, 72, 75, 80, 85, 88, 90, 92, 95]`

        If a new student scores **78**, what is their percentile rank (using `kind='mean'` method as in `scipy.stats.percentileofscore`)?
        """)
        task_scores_perc = [65, 70, 72, 75, 80, 85, 88, 90, 92, 95]
        task_new_score = 78

        user_rank_perc_task = st.number_input("Enter the percentile rank (e.g., 50.0):", format="%.2f",
                                              key='perc_task_rank_input')

        if st.button("Check My Answer - Percentile"):
            correct_rank = percentileofscore(task_scores_perc, task_new_score, kind='mean')

            if abs(user_rank_perc_task - correct_rank) < 0.01:
                st.success(f"Correct! A score of {task_new_score} is at the `{correct_rank:.2f}th` percentile.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. The correct percentile rank is `{correct_rank:.2f}`. Make sure to use the `kind='mean'` method for `percentileofscore` and ensure your data is sorted!")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: Deeper Dive into Percentile Calculation", expanded=False):
        st.subheader("Percentile Calculation Methods (Conceptual)")
        st.markdown("""
        There isn't just one way to calculate percentiles, especially when the desired percentile doesn't fall exactly on a data point. Different software (like Excel, R, Python's NumPy/SciPy) might use slightly different interpolation methods. Here's a common one (often called "nearest rank" or similar):

        #### **Example: Finding the 75th Percentile of `[10, 15, 20, 25, 30, 35, 40, 45, 50, 55]` (N=10)**

        1.  **Order the data:** `[10, 15, 20, 25, 30, 35, 40, 45, 50, 55]` (already done).

        2.  **Calculate the index ($I$):**
            * $P = 75$ (for 75th percentile)
            * $N = 10$ (total data points)
            * $I = (75 / 100) \times 10 = 0.75 \times 10 = 7.5$

        3.  **Handle fractional index (Interpolation):**
            * Since $I=7.5$ is not an integer, we often interpolate. NumPy's default (linear interpolation) would look at the values around the 7th and 8th positions (0-indexed, so 8th and 9th actual values).
            * The 7th value (0-indexed) is 45. The 8th value is 50.
            * The 75th percentile lies between 45 and 50.
            * NumPy might calculate it as `45 + (0.5 * (50 - 45)) = 45 + 2.5 = 47.5`.

        **Key Point:** The exact method matters for edge cases and precision, but the core idea remains finding the value that splits the data at the desired percentage.

        #### **Understanding Quartiles and IQR (Interquartile Range):**
        * **Q1 (25th Percentile):** Marks the value below which the lowest 25% of the data falls.
        * **Q2 (50th Percentile / Median):** Marks the value below which the lowest 50% of the data falls.
        * **Q3 (75th Percentile):** Marks the value below which the lowest 75% of the data falls.
        * **Interquartile Range (IQR):** $IQR = Q3 - Q1$. This measures the spread of the middle 50% of the data, making it robust to outliers (unlike standard deviation, which uses all data points relative to the mean).
        """)
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Boxplot_vs_PDF.svg/1200px-Boxplot_vs_PDF.svg.png",
            caption="Box Plot illustrating Quartiles and IQR (Source: Wikipedia)",
            use_column_width=True)
        st.info(
            "💡 **Tip:** Percentiles are widely used in descriptive statistics and for identifying outliers (e.g., values outside 1.5 * IQR from Q1/Q3).")


def topic_data_distribution():
    st.header("📈 Data Distribution")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Data Distribution?")
        st.markdown("""
        **Data Distribution** refers to the way data points are spread or arranged over a range of values. It describes the shape of the data, telling us where values tend to cluster and where they are sparse.

        Understanding data distribution is crucial because:
        * It helps choose appropriate statistical tests or machine learning models.
        * It can reveal underlying patterns, anomalies, or characteristics of the data.
        * It informs about the central tendency, spread, and symmetry of the data.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're tracking the ages of visitors to a children's museum versus a historical museum.

        * **Children's Museum:** You'd expect most visitors to be young children, maybe a few adults. The age distribution would be "skewed right" (many low values, a tail towards higher ages).
        * **Historical Museum:** You might expect a more even spread of adult ages, or perhaps a peak around retirement age. The distribution would look different, possibly more symmetrical or "skewed left" if there's a large elderly population.

        By looking at how ages are distributed, you immediately get a picture of the typical visitor.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Concepts of Shape", expanded=False):
        st.subheader("Key Characteristics of Distribution Shape")
        st.markdown("""
        While there isn't a single equation for "distribution," we describe its shape using concepts like:

        * **Symmetry:** Is the distribution balanced around its center?
            * **Symmetrical:** Mean, Median, Mode are roughly equal (e.g., Normal Distribution).
            * **Skewed:** One tail is longer than the other.
                * **Right-Skewed (Positive Skew):** The tail extends to the right (higher values). Mean > Median > Mode. (e.g., income distribution, house prices).
                * **Left-Skewed (Negative Skew):** The tail extends to the left (lower values). Mean < Median < Mode. (e.g., exam scores when most students do well).
        """)

        st.markdown("""
        * **Modality:** How many peaks (modes) does the distribution have?
            * **Unimodal:** One peak (most common).
            * **Bimodal:** Two distinct peaks.
            * **Multimodal:** More than two peaks.
        """)

        st.markdown("""
        * **Kurtosis:** Describes the "tailedness" of the distribution (how many outliers or extreme values).
            * **Leptokurtic:** Fatter tails, sharper peak (more outliers).
            * **Mesokurtic:** Normal kurtosis (like a normal distribution).
            * **Platykurtic:** Thinner tails, flatter peak (fewer outliers).
        """)
        latex_equation(r'\text{Skewness} = E\left[ \left( \frac{X - \mu}{\sigma} \right)^3 \right]')
        latex_equation(r'\text{Kurtosis} = E\left[ \left( \frac{X - \mu}{\sigma} \right)^4 \right] - 3')
        st.markdown("""
        * These formulas ($E[]$ denotes expected value) quantify skewness and kurtosis. A skewness of 0 indicates perfect symmetry. Kurtosis around 0 (after subtracting 3 for normal distribution) is mesokurtic.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Visualize Different Data Distributions")

        col1_dd, col2_dd = st.columns(2)
        dist_type = col1_dd.selectbox("Choose a Distribution Type:",
                                      ("Normal (Symmetric)", "Right-Skewed", "Left-Skewed", "Uniform", "Bimodal"),
                                      key='dd_dist_type')
        num_samples_dd = col2_dd.slider("Number of data points:", 100, 1000, 300, key='dd_samples')

        data_dd = []
        np.random.seed(0)

        if dist_type == "Normal (Symmetric)":
            data_dd = np.random.normal(loc=50, scale=10, size=num_samples_dd)
            st.info("💡 **Tip:** Normal distribution is often found in natural phenomena and statistical errors.")
        elif dist_type == "Right-Skewed":
            # Example: Exponential distribution
            data_dd = np.random.exponential(scale=10, size=num_samples_dd)
            st.info("💡 **Tip:** Income distribution or reaction times are often right-skewed.")
        elif dist_type == "Left-Skewed":
            # Example: Reflecting exponential distribution
            data_dd = 100 - np.random.exponential(scale=10, size=num_samples_dd)
            st.info("💡 **Tip:** Exam scores for an easy test can be left-skewed.")
        elif dist_type == "Uniform":
            data_dd = np.random.uniform(low=0, high=100, size=num_samples_dd)
            st.info("💡 **Tip:** All values have roughly equal probability of occurring within a range.")
        elif dist_type == "Bimodal":
            # Two normal distributions combined
            data_dd = np.concatenate([
                np.random.normal(loc=30, scale=5, size=num_samples_dd // 2),
                np.random.normal(loc=70, scale=5, size=num_samples_dd // 2)
            ])
            st.info("💡 **Tip:** Can occur when two distinct groups are present in the data.")

        df_dd = pd.DataFrame({'Value': data_dd})

        st.subheader("Histogram and Kernel Density Estimate (KDE)")
        fig_dd = px.histogram(df_dd, x='Value', nbins=max(10, num_samples_dd // 20),
                              title=f"Distribution of {dist_type} Data",
                              opacity=0.7, color_discrete_sequence=['#264653'],
                              histnorm='probability density')  # Normalize to PDF for KDE overlay

        # Overlay KDE for smoother representation
        sns_fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(df_dd['Value'], ax=ax, color='#e76f51', linewidth=3)
        ax.set_title(f"KDE of {dist_type} Data")
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')

        # Convert Matplotlib figure to Plotly compatible image if needed, or simply display
        # For simplicity, let's just display the seaborn plot alongside plotly histogram
        st.pyplot(sns_fig)
        st.plotly_chart(fig_dd, use_container_width=True)
        st.info(
            "💡 **Histogram:** Shows frequency of data in bins. **KDE:** Provides a smoothed representation of the data's probability density function.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Plotting Data Distributions in Python")
        st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Generate different types of data
np.random.seed(42)
data_normal = np.random.normal(loc=0, scale=1, size=1000) # Mean 0, Std Dev 1
data_right_skew = np.random.exponential(scale=2, size=1000)
data_left_skew = 10 - np.random.exponential(scale=2, size=1000) # Reflected exponential
data_uniform = np.random.uniform(low=-5, high=5, size=1000)
data_bimodal = np.concatenate([np.random.normal(-3, 1, 500), np.random.normal(3, 1, 500)])

# Convert to DataFrame for easier plotting with Seaborn/Plotly
df_normal = pd.DataFrame({'Value': data_normal})
df_right_skew = pd.DataFrame({'Value': data_right_skew})
df_bimodal = pd.DataFrame({'Value': data_bimodal})

print("--- Normal Distribution ---")
print(df_normal.describe())
fig_norm = px.histogram(df_normal, x='Value', nbins=30, title='Normal Distribution', histnorm='probability density')
fig_norm.show() # In a real app, use st.plotly_chart(fig_norm)

# Or using Seaborn for histogram with KDE
plt.figure(figsize=(8, 5))
sns.histplot(df_normal['Value'], kde=True, bins=30)
plt.title('Normal Distribution with KDE (Seaborn)')
plt.show() # In a real app, use st.pyplot(plt.gcf())

print("\\n--- Right-Skewed Distribution ---")
print(df_right_skew.describe())
fig_right = px.histogram(df_right_skew, x='Value', nbins=30, title='Right-Skewed Distribution')
fig_right.show()

print("\\n--- Bimodal Distribution ---")
print(df_bimodal.describe())
fig_bimodal = px.histogram(df_bimodal, x='Value', nbins=30, title='Bimodal Distribution')
fig_bimodal.show()
        """, language="python")
        st.markdown(
            "This code shows how to generate different types of data and visualize their distributions using both `plotly.express` (interactive) and `seaborn` (static).")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Identify Distribution Type")
        st.markdown("""
        Look at the histogram below. Which type of distribution does it most closely resemble?
        """)

        # Generate a new skewed dataset for the task
        np.random.seed(10)
        task_data_dd = 100 - np.random.exponential(scale=15, size=200)  # Clearly left-skewed
        df_task_dd = pd.DataFrame({'Values': task_data_dd})

        fig_task_dd = px.histogram(df_task_dd, x='Values', nbins=30, title="Task: Identify This Distribution",
                                   opacity=0.7, color_discrete_sequence=['#f4a261'])
        st.plotly_chart(fig_task_dd, use_container_width=True)

        user_dist_type = st.radio("This distribution appears to be:",
                                  ("Normal (Symmetric)", "Right-Skewed (Positive)", "Left-Skewed (Negative)",
                                   "Uniform"),
                                  key='dd_task_radio')

        if st.button("Check My Answer - Distribution Type"):
            correct_answer = "Left-Skewed (Negative)"
            if user_dist_type == correct_answer:
                st.success(
                    f"Correct! This is a **{correct_answer}** distribution, with a tail extending towards the lower values.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. The tail of the distribution points to the left (lower values), making it left-skewed. Review the characteristics of skewed distributions.")

    # 7. Bonus: How the Algorithm Works (Conceptual)
    with st.expander("✨ Bonus: Why Different Shapes?", expanded=False):
        st.subheader("Factors Shaping Data Distributions")
        st.markdown("""
        The shape of a data distribution isn't random; it reflects the underlying processes generating the data.

        * **Normal Distribution (Bell Curve):** Often results from many independent random factors adding up (Central Limit Theorem). Think of heights, measurement errors, or test scores of a large, diverse group.
        * **Skewed Distributions:** Happen when there are natural limits on one side but not the other.
            * **Right-Skew (Positive):** Very common for naturally bounded quantities at zero but no upper limit. Examples: income (can't be negative, but some very high earners pull the tail right), house prices, waiting times.
            * **Left-Skew (Negative):** Less common in natural phenomena, but can occur with maximum limits. Examples: scores on an easy exam (most people score high, few score low), age of death (most people live long, few die very young).
        * **Uniform Distribution:** Every outcome in a range is equally likely. Example: Rolling a fair die (each face has 1/6 chance), random numbers generated within a range.
        * **Bimodal/Multimodal Distributions:** Indicate that the data might come from two or more distinct subgroups. Example: Heights of people in a room (could show two peaks for male and female heights), exam scores if students either studied a lot or not at all.

        By analyzing the distribution's shape, you gain insights into the nature of the data and its generating process.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*QyV5K_y9d17Q3eF60zB_4A.png",
                 caption="Different types of distribution shapes (Source: Medium)",
                 use_column_width=True)
        st.info("💡 **Remember:** The shape of your data influences what statistical methods are appropriate to use.")


def topic_normal_data_distribution():
    st.header("🔔 Normal Data Distribution")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Normal Distribution (Bell Curve)?")
        st.markdown("""
        The **Normal Distribution**, often called the **"bell curve"** or Gaussian distribution, is one of the most important probability distributions in statistics. It's symmetrical, with a single peak at the center, and its values taper off equally in both directions.

        **Key Characteristics:**
        * **Symmetrical:** The left side is a mirror image of the right side.
        * **Mean = Median = Mode:** All three measures of central tendency are located at the center of the distribution.
        * **Asymptotic to the x-axis:** The tails of the curve approach the x-axis but never quite touch it, implying that extreme values are possible but increasingly unlikely.
        * It is defined by two parameters: its **mean ($\mu$)** and its **standard deviation ($\sigma$)**.
        """)
        st.markdown("""
        **Daily-life Example:**
        Many natural phenomena and measurements tend to follow a normal distribution, especially if they are influenced by many small, random factors:
        * **Heights of adult people in a large population:** Most people are of average height, with fewer very tall or very short individuals.
        * **IQ scores:** Designed to be normally distributed with a mean of 100.
        * **Measurement errors:** Errors in scientific experiments often distribute normally around the true value.

        The normal distribution is a cornerstone of statistical inference and many machine learning algorithms.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Probability Density Function (PDF) and Empirical Rule")
        st.markdown("""
        The mathematical formula for the probability density function (PDF) of a normal distribution is:
        """)
        latex_equation(r'f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}')
        st.markdown("""
        * $f(x)$: The probability density at a given value $x$.
        * $\mu$: The mean of the distribution.
        * $\sigma$: The standard deviation of the distribution.
        * $\pi$: The mathematical constant pi (approximately 3.14159).
        * $e$: Euler's number (the base of the natural logarithm, approximately 2.71828).

        This formula describes the bell shape, where the peak is at $\mu$ and the spread is determined by $\sigma$.

        #### **The Empirical Rule (68-95-99.7 Rule):**
        For a normal distribution, approximately:
        * **68%** of the data falls within **1 standard deviation** ($\mu \pm 1\sigma$) of the mean.
        * **95%** of the data falls within **2 standard deviations** ($\mu \pm 2\sigma$) of the mean.
        * **99.7%** of the data falls within **3 standard deviations** ($\mu \pm 3\sigma$) of the mean.
        """)
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Standard_deviation_diagram.svg/1200px-Standard_deviation_diagram.svg.png",
            caption="The Empirical Rule for Normal Distribution (Source: Wikipedia)",
            use_column_width=True)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Generate and Visualize Normal Distribution")

        col1_nd, col2_nd, col3_nd = st.columns(3)
        num_samples_nd = col1_nd.slider("Number of data points:", 100, 1000, 500, key='nd_samples')
        mean_nd = col2_nd.slider("Mean (μ):", 0, 100, 50, key='nd_mean')
        std_dev_nd = col3_nd.slider("Standard Deviation (σ):", 1, 20, 10, key='nd_std_dev')

        np.random.seed(42)
        data_nd = np.random.normal(loc=mean_nd, scale=std_dev_nd, size=num_samples_nd)
        df_nd = pd.DataFrame({'Value': data_nd})

        st.subheader("Histogram with Normal Distribution Curve")
        fig_nd = px.histogram(df_nd, x='Value', nbins=max(10, num_samples_nd // 20),
                              title=f"Normal Distribution (μ={mean_nd}, σ={std_dev_nd})",
                              opacity=0.6, color_discrete_sequence=['#457b9d'],
                              histnorm='probability density')  # Normalize for PDF overlay

        # Overlay the theoretical PDF curve
        x_range = np.linspace(data_nd.min(), data_nd.max(), 500)
        pdf_curve = norm.pdf(x_range, loc=mean_nd, scale=std_dev_nd)
        fig_nd.add_trace(go.Scatter(x=x_range, y=pdf_curve, mode='lines', name='Normal PDF',
                                    line=dict(color='#e76f51', width=3)))

        # Add empirical rule lines
        for i in range(1, 4):
            fig_nd.add_vline(x=mean_nd + i * std_dev_nd, line_dash="dot", line_color="green",
                             annotation_text=f"+{i}σ", annotation_position="top right")
            fig_nd.add_vline(x=mean_nd - i * std_dev_nd, line_dash="dot", line_color="green",
                             annotation_text=f"-{i}σ", annotation_position="top left")

        st.plotly_chart(fig_nd, use_container_width=True)
        st.info(
            "💡 **Observation:** The red line shows the perfect theoretical bell curve. Your generated data's histogram approximates it, especially with more samples.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Working with Normal Distribution in Python")
        st.code("""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm # To get PDF and CDF

# 1. Generate normally distributed data
mean_val = 60
std_dev_val = 8
num_samples = 1000

data = np.random.normal(loc=mean_val, scale=std_dev_val, size=num_samples)

print(f"Generated data mean: {np.mean(data):.2f}")
print(f"Generated data std dev: {np.std(data):.2f}")

# 2. Plotting the histogram with KDE (approximating the PDF)
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True, stat='density') # stat='density' for PDF-like scale
plt.title(f'Histogram of Normally Distributed Data (Mean={mean_val}, StdDev={std_dev_val})')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# 3. Plotting the theoretical Probability Density Function (PDF)
x_values = np.linspace(mean_val - 4*std_dev_val, mean_val + 4*std_dev_val, 500)
pdf_values = norm.pdf(x_values, loc=mean_val, scale=std_dev_val)

plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.title('Theoretical Normal Distribution PDF')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# 4. Calculating Z-score
# Z-score measures how many standard deviations an element is from the mean.
# Formula: Z = (x - μ) / σ
x_value = 70 # A specific data point
z_score = (x_value - mean_val) / std_dev_val
print(f"Z-score for value {x_value}: {z_score:.2f}")

# You can use Z-scores to find probabilities using a Z-table or norm.cdf()
probability_less_than_x = norm.cdf(x_value, loc=mean_val, scale=std_dev_val)
print(f"Probability of value less than {x_value}: {probability_less_than_x:.2%}")
        """, language="python")
        st.markdown(
            "This code generates normal data, visualizes its histogram, plots the theoretical PDF, and shows how to calculate a Z-score.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Apply the Empirical Rule")
        st.markdown("""
        Suppose the scores on a standardized test are normally distributed with a **mean of 500** and a **standard deviation of 100**.

        Using the Empirical Rule (68-95-99.7), what percentage of students scored between **400 and 600**?
        """)

        user_percentage_nd_task = st.number_input("Enter the percentage (e.g., 68 for 68%):", min_value=0.0,
                                                  max_value=100.0, step=0.1, key='nd_task_perc_input')

        if st.button("Check My Answer - Empirical Rule"):
            # 400 is (500 - 1*100) = mean - 1 std dev
            # 600 is (500 + 1*100) = mean + 1 std dev
            correct_percentage = 68.0

            if abs(user_percentage_nd_task - correct_percentage) < 0.1:
                st.success(
                    f"Correct! Approximately `{correct_percentage:.1f}%` of students scored between 400 and 600 (which is within 1 standard deviation of the mean).")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. Remember the 68-95-99.7 rule. 400 to 600 represents the range within 1 standard deviation from the mean. The correct answer is `{correct_percentage:.1f}%`.")
                st.info("Hint: The mean is 500, std dev is 100. How many standard deviations away are 400 and 600?")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: The Power of Z-Scores", expanded=False):
        st.subheader("Standardizing Data with Z-Scores")
        st.markdown("""
        The **Z-score** (also called standard score) tells you how many standard deviations a data point is from the mean of its distribution.

        **Formula:**
        """)
        latex_equation(r'Z = \frac{x - \mu}{\sigma}')
        st.markdown("""
        * $x$: The individual data point.
        * $\mu$: The mean of the population.
        * $\sigma$: The standard deviation of the population.

        #### **Why are Z-scores useful?**
        1.  **Comparison Across Different Datasets:** They allow you to compare data points from different normal distributions.
            * *Example:* If a student scores 85 on a test with a mean of 70 and std dev of 10 (Z = 1.5), and another student scores 70 on a test with a mean of 60 and std dev of 5 (Z = 2.0), the second student actually performed relatively better, despite having a lower raw score, because their Z-score is higher.
        2.  **Probability Calculation:** Once data is standardized (transformed into Z-scores, forming a "Standard Normal Distribution" with mean=0, std dev=1), you can use Z-tables or statistical software to find the probability of observing a value less than, greater than, or between certain points.
        3.  **Outlier Detection:** Data points with very high or very low Z-scores (e.g., beyond $\pm 2$ or $\pm 3$) are often considered outliers.

        **Conceptual Steps for Z-score:**
        1.  Find the mean of your data ($\mu$).
        2.  Find the standard deviation of your data ($\sigma$).
        3.  For any given data point ($x$):
            * Calculate the difference between the data point and the mean: $(x - \mu)$.
            * Divide this difference by the standard deviation: $(x - \mu) / \sigma$.
        """)
        st.image("https://www.simplypsychology.org/Z-score.jpg",
                 caption="Visualizing Z-scores on a Normal Distribution (Source: Simply Psychology)",
                 use_column_width=True)
        st.info(
            "💡 **Key takeaway:** Z-scores normalize data, making it comparable and easier to analyze relative position and probabilities.")


def topic_scatter_plot():
    st.header("✨ Scatter Plot")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is a Scatter Plot?")
        st.markdown("""
        A **Scatter Plot** is a type of plot or mathematical diagram that uses Cartesian coordinates to display values for typically two variables for a set of data. The data is displayed as a collection of points, where each point has the value of one variable determining its position on the horizontal axis and the value of the other variable determining its position on the vertical axis.

        Scatter plots are used to observe relationships between two numerical variables.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're trying to see if there's a relationship between the amount of time students spend exercising and their academic performance.

        * You'd plot "Hours of Exercise per Week" on the X-axis.
        * You'd plot "GPA" on the Y-axis.

        Each student would be a single dot on the graph. By looking at the pattern of the dots, you can see if:
        * More exercise tends to lead to higher GPA (positive relationship).
        * More exercise tends to lead to lower GPA (negative relationship).
        * There's no clear relationship.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Concepts of Correlation", expanded=False):
        st.subheader("Understanding Relationships: Correlation")
        st.markdown("""
        While a scatter plot doesn't have an "equation" in itself, it visually represents the **correlation** between two variables. Correlation measures the strength and direction of a linear relationship between two quantitative variables.

        The most common measure is the **Pearson Correlation Coefficient (r)**:
        """)
        latex_equation(
            r'r = \frac{n(\sum xy) - (\sum x)(\sum y)}{\sqrt{[n\sum x^2 - (\sum x)^2][n\sum y^2 - (\sum y)^2]}}')
        st.markdown("""
        * $n$: Number of data points.
        * $x, y$: Individual data points for the two variables.
        * $\sum$: Summation.

        **Interpretation of Pearson Correlation (r):**
        * **+1:** Perfect positive linear relationship (as X increases, Y increases perfectly).
        * **-1:** Perfect negative linear relationship (as X increases, Y decreases perfectly).
        * **0:** No linear relationship.
        * Values closer to +1 or -1 indicate stronger relationships.
        """)
        st.image("https://www.statisticshowto.com/wp-content/uploads/2012/11/correlation-coefficient.jpg",
                 caption="Examples of different correlation types (Source: StatisticHowTo.com)",
                 use_column_width=True)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Create Your Own Scatter Plot")

        data_source_sp = st.radio("Choose data source:", ("Generate synthetic data", "Use uploaded CSV (if available)"),
                                  key='sp_data_source')

        X_sp, Y_sp = None, None
        df_sp = None

        if data_source_sp == "Generate synthetic data":
            st.markdown("#### Generate Data Points with Custom Correlation")
            col1, col2 = st.columns(2)
            num_samples_sp = col1.slider("Number of data points:", 50, 500, 100, key='sp_samples')
            correlation_strength = col2.slider("Correlation Strength:", -1.0, 1.0, 0.7, 0.1, key='sp_correlation')

            np.random.seed(1)  # For consistent base
            # Generate correlated data (simple way)
            x_data = np.random.rand(num_samples_sp) * 100
            y_data = correlation_strength * x_data + (1 - abs(correlation_strength)) * np.random.randn(
                num_samples_sp) * 30

            df_sp = pd.DataFrame({'X_Value': x_data, 'Y_Value': y_data})
            st.info("Synthetic data generated.")

        elif data_source_sp == "Use uploaded CSV (if available)":
            if global_df is not None:
                st.markdown("#### Select Two Numerical Columns from Uploaded CSV")
                numerical_cols = global_df.select_dtypes(include=np.number).columns.tolist()
                if len(numerical_cols) < 2:
                    st.warning("Uploaded CSV must have at least two numerical columns for a scatter plot.")
                else:
                    x_col_sp = st.selectbox("Select X-axis Column:", numerical_cols, key='sp_x_col')
                    y_col_sp = st.selectbox("Select Y-axis Column:", [col for col in numerical_cols if col != x_col_sp],
                                            key='sp_y_col')

                    if x_col_sp and y_col_sp:
                        df_sp = global_df[[x_col_sp, y_col_sp]].copy()
                        df_sp.columns = ['X_Value', 'Y_Value']  # Standardize names for plotting
                        st.success(f"Using '{x_col_sp}' as X and '{y_col_sp}' as Y from uploaded CSV.")
                    else:
                        st.warning("Please select both X and Y columns.")
            else:
                st.info("No CSV file uploaded. Please upload one in the sidebar or generate synthetic data.")

        if df_sp is not None and not df_sp.empty:
            # Calculate actual correlation coefficient for display
            actual_correlation = df_sp['X_Value'].corr(df_sp['Y_Value'])
            st.write(f"**Calculated Pearson Correlation Coefficient (r):** `{actual_correlation:.2f}`")

            fig_sp = px.scatter(df_sp, x='X_Value', y='Y_Value',
                                title="Scatter Plot of X vs. Y",
                                labels={'X_Value': 'X-axis Variable', 'Y_Value': 'Y-axis Variable'},
                                opacity=0.7, color_discrete_sequence=['#2a9d8f'])

            # Optional: Add a trendline
            add_trendline = st.checkbox("Add Linear Trendline?", value=True, key='sp_trendline_checkbox')
            if add_trendline:
                fig_sp.update_traces(marker=dict(size=8), selector=dict(mode='markers'))  # Ensure markers are visible
                fig_sp.add_trace(
                    go.Scatter(x=df_sp['X_Value'], y=df_sp['Y_Value'].values, mode='markers', marker_opacity=0.7,
                               showlegend=False))  # Scatter points

                # Fit simple linear regression for trendline
                model_sp = LinearRegression()
                X_fit = df_sp[['X_Value']].values
                y_fit = df_sp['Y_Value'].values
                model_sp.fit(X_fit, y_fit)
                y_pred_sp = model_sp.predict(X_fit)

                fig_sp.add_trace(go.Scatter(x=df_sp['X_Value'], y=y_pred_sp.flatten(), mode='lines',
                                            name='Trendline', line=dict(color='#e76f51', width=2)))

            st.plotly_chart(fig_sp, use_container_width=True)
            st.info(
                "💡 **Observation:** Look for patterns! Do the points tend to go up (positive correlation), down (negative correlation), or form a random cloud (no correlation)?")
        else:
            st.warning("Please generate or select valid numerical data for the scatter plot.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Creating Scatter Plots in Python")
        st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Example 1: Positive Correlation
np.random.seed(42)
x_pos = np.random.rand(50) * 10
y_pos = x_pos * 2 + np.random.randn(50) * 1.5 + 5

df_pos = pd.DataFrame({'X': x_pos, 'Y': y_pos})

# Plotly Scatter Plot (Interactive)
fig_plotly_pos = px.scatter(df_pos, x='X', y='Y', title='Positive Correlation', trendline='ols')
fig_plotly_pos.show() # In Streamlit: st.plotly_chart(fig_plotly_pos)

# Seaborn Scatter Plot (Static, often good for quick plots)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pos, x='X', y='Y')
plt.title('Positive Correlation (Seaborn)')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.grid(True)
plt.show() # In Streamlit: st.pyplot(plt.gcf())

# Example 2: No Correlation
x_no = np.random.rand(50) * 10
y_no = np.random.rand(50) * 10 # No relationship
df_no = pd.DataFrame({'X': x_no, 'Y': y_no})

fig_plotly_no = px.scatter(df_no, x='X', y='Y', title='No Correlation')
fig_plotly_no.show()

# Calculating Pearson Correlation Coefficient
correlation_pos = df_pos['X'].corr(df_pos['Y'])
correlation_no = df_no['X'].corr(df_no['Y'])
print(f"Correlation for Positive example: {correlation_pos:.2f}")
print(f"Correlation for No Correlation example: {correlation_no:.2f}")
        """, language="python")
        st.markdown(
            "This code demonstrates how to create scatter plots using `plotly.express` for interactivity and `seaborn` for static plots, and how to calculate the correlation coefficient.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Identify Correlation Type")
        st.markdown("""
        Examine the scatter plot below. Does it show a positive, negative, or no correlation?
        """)

        # Generate data for a task - slightly negative correlation
        np.random.seed(5)
        task_x_sp = np.random.rand(80) * 100
        task_y_sp = -0.6 * task_x_sp + np.random.randn(80) * 20 + 80  # Negative correlation
        df_task_sp = pd.DataFrame({'X': task_x_sp, 'Y': task_y_sp})

        fig_task_sp = px.scatter(df_task_sp, x='X', y='Y', title="Task: Identify Correlation",
                                 opacity=0.7, color_discrete_sequence=['#f4a261'])
        st.plotly_chart(fig_task_sp, use_container_width=True)

        user_correlation_type = st.radio("This scatter plot shows:",
                                         ("Positive Correlation", "Negative Correlation", "No Correlation"),
                                         key='sp_task_radio')

        if st.button("Check My Answer - Scatter Plot"):
            correct_answer = "Negative Correlation"
            if user_correlation_type == correct_answer:
                st.success(f"Correct! As X increases, Y tends to decrease, indicating a **{correct_answer}**.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. Look at the general trend of the points. As X goes up, does Y tend to go up, down, or stay flat? The correct answer is **{correct_answer}**.")

    # 7. Bonus: How the Algorithm Works (Conceptual)
    with st.expander("✨ Bonus: Correlation vs. Causation", expanded=False):
        st.subheader("Correlation Does Not Imply Causation!")
        st.markdown("""
        This is a critical concept when interpreting scatter plots and correlation coefficients:

        * **Correlation:** Simply means two variables tend to change together. They move in the same direction (positive correlation), opposite directions (negative correlation), or no consistent direction.
            * *Example:* Ice cream sales and drowning incidents might both increase in summer. They are correlated.
        * **Causation:** Means one variable directly causes a change in another.
            * *Example:* Turning on a light switch causes the light to turn on.

        **Why Correlation is NOT Causation:**
        Even if two variables have a strong correlation, it doesn't mean one causes the other. There could be:

        1.  **A Third Variable (Confounding Variable):** An unobserved factor that influences both correlated variables.
            * *Ice Cream & Drowning Example:* The third variable is "summer heat." Hot weather increases both ice cream sales and swimming (and thus, unfortunately, drowning incidents). Ice cream doesn't cause drownings!
        2.  **Reverse Causation:** Y causes X, not X causes Y.
        3.  **Coincidence:** Pure random chance.

        **Always be cautious when interpreting relationships from scatter plots!** Correlation is a hint, not proof of cause.
        """)
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Correlation_and_causation.svg/1024px-Correlation_and_causation.svg.png",
            caption="Correlation vs. Causation (Source: Wikipedia)",
            use_column_width=True)
        st.info(
            "💡 **Key takeaway:** Scatter plots are powerful for *identifying* relationships, but deeper analysis is needed to *prove* causation.")


def topic_polynomial_regression():
    st.header("📈 Polynomial Regression")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Polynomial Regression?")
        st.markdown("""
        **Polynomial Regression** is a form of regression analysis in which the relationship between the independent variable $X$ and the dependent variable $Y$ is modeled as an **nth degree polynomial**. While it models non-linear relationships, it is still considered a form of **linear model** because the model itself (the coefficients) is linear in terms of the weights, despite the curvilinear relationship with the input variable.

        It's used when the relationship between variables isn't a straight line, but rather a curve.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine trying to model the relationship between a car's speed and its fuel efficiency.
        * At very low speeds, efficiency might be low.
        * It might increase to a peak at moderate speeds.
        * Then, it might decrease again at very high speeds due to air resistance.

        A simple straight line (linear regression) won't capture this curve. A polynomial regression could, by adding terms like speed squared ($\text{speed}^2$) or even speed cubed ($\text{speed}^3$) to the model.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("The Polynomial Regression Equation")
        st.markdown("""
        The general equation for a polynomial regression model with one independent variable $X$ and degree $n$ is:
        """)
        latex_equation(r'y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + \dots + \beta_nx^n + \epsilon')
        st.markdown("""
        * $y$: The dependent (target) variable.
        * $x$: The independent (feature) variable.
        * $\beta_0$: The Y-intercept.
        * $\beta_1, \beta_2, \dots, \beta_n$: The coefficients for each polynomial term.
        * $x^2, x^3, \dots, x^n$: The polynomial features (e.g., $x$ squared, $x$ cubed, etc.).
        * $\epsilon$: The error term.

        **Key Idea:** We transform the single input feature $X$ into multiple features ($X, X^2, X^3, \dots, X^n$) and then fit a **linear regression model** to these new, transformed features.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Fit a Polynomial Regression Model")

        col1_pr, col2_pr = st.columns(2)
        num_points_pr = col1_pr.slider("Number of data points:", 50, 500, 100, key='pr_num_points')
        poly_degree = col2_pr.slider("Polynomial Degree (n):", 1, 10, 2, key='pr_poly_degree')
        noise_level_pr = st.slider("Noise Level:", 0.0, 50.0, 10.0, key='pr_noise_level')

        st.info(
            "💡 Try changing the polynomial degree and observe how the curve fits the data. Watch out for overfitting with high degrees!")

        np.random.seed(0)  # for reproducibility
        # Generate some non-linear data
        X_pr = np.linspace(-5, 5, num_points_pr).reshape(-1, 1)
        y_pr = 0.5 * X_pr ** 3 - 3 * X_pr ** 2 + 5 * X_pr + 20 + np.random.normal(0, noise_level_pr,
                                                                                  num_points_pr).reshape(-1, 1)

        # Create polynomial features
        poly_features = PolynomialFeatures(degree=poly_degree)
        X_poly_pr = poly_features.fit_transform(X_pr)

        # Fit linear regression on polynomial features
        model_pr = LinearRegression()
        model_pr.fit(X_poly_pr, y_pr)

        # Predict on a smooth range for plotting the curve
        X_plot = np.linspace(X_pr.min(), X_pr.max(), 500).reshape(-1, 1)
        X_plot_poly = poly_features.transform(X_plot)
        y_pred_pr = model_pr.predict(X_plot_poly)

        # Create Plotly figure
        fig_pr = px.scatter(x=X_pr.flatten(), y=y_pr.flatten(), title="Polynomial Regression Fit",
                            labels={'x': 'X Value', 'y': 'Y Value'},
                            color_discrete_sequence=['#2a9d8f'])

        # Add the polynomial regression line
        fig_pr.add_trace(go.Scatter(x=X_plot.flatten(), y=y_pred_pr.flatten(),
                                    mode='lines', name=f'Degree {poly_degree} Fit',
                                    line=dict(color='#e76f51', width=3)))

        st.plotly_chart(fig_pr, use_container_width=True)

        # Metrics (R-squared)
        y_pred_on_train = model_pr.predict(X_poly_pr)
        r2_pr = r2_score(y_pr, y_pred_on_train)
        st.write(f"**R-squared (on training data):** `{r2_pr:.4f}`")
        st.info(
            "R-squared measures how well the regression line approximates the real data points. 1.0 is a perfect fit.")

        if poly_degree > 5 and r2_pr > 0.95:
            st.warning(
                "⚠️ **Warning: Possible Overfitting!** A very high polynomial degree might perfectly fit the training data (high R-squared) but could perform poorly on new, unseen data, especially with noise.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Polynomial Regression in Python")
        st.code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. Generate some non-linear sample data
np.random.seed(0)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X**3 - 5*X**2 + 2*X + 10 + np.random.normal(0, 5, 100).reshape(-1, 1)

# 2. Choose the polynomial degree
poly_degree = 3 # For a cubic relationship

# 3. Create Polynomial Features
# This transforms X into [1, X, X^2, X^3, ..., X^n]
poly_transformer = PolynomialFeatures(degree=poly_degree)
X_poly = poly_transformer.fit_transform(X)

print(f"Original X shape: {X.shape}")
print(f"Polynomial Features shape (for degree {poly_degree}): {X_poly.shape}")
# print(X_poly[:5]) # Uncomment to see the transformed features

# 4. Fit a Linear Regression model on the polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# 5. Make predictions
# Create a smooth range of X values for plotting the regression curve
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_plot_poly = poly_transformer.transform(X_plot)
y_pred = model.predict(X_plot_poly)

# 6. Evaluate the model (e.g., R-squared)
y_pred_on_train = model.predict(X_poly)
r2 = r2_score(y, y_pred_on_train)
print(f"R-squared on training data: {r2:.4f}")

# 7. Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_plot, y_pred, color='red', label=f'Polynomial Regression (Degree {poly_degree})', linewidth=3)
plt.title(f'Polynomial Regression (Degree {poly_degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
        """, language="python")
        st.markdown(
            "This code demonstrates the steps to implement polynomial regression: transforming features, fitting a linear model, and evaluating its performance.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Choose the Best Degree")
        st.markdown("""
        You have data that looks like a parabola (a U-shape). Which polynomial degree would generally be the *most appropriate* for modeling this relationship without overfitting too much?
        """)

        user_choice_pr_task = st.radio("Select the ideal polynomial degree:", (1, 2, 3, 4), key='pr_task_degree')

        if st.button("Check My Answer - Polynomial Regression"):
            correct_degree = 2
            if user_choice_pr_task == correct_degree:
                st.success(
                    f"Correct! A degree {correct_degree} polynomial (quadratic) is perfect for modeling a parabolic relationship.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. A parabola is a quadratic function, which corresponds to a polynomial of degree 2. Degree 1 is a straight line, while degrees 3 or 4 would be more complex than needed and risk overfitting.")

    # 7. Bonus: Overfitting vs. Underfitting
    with st.expander("✨ Bonus: Overfitting and Underfitting", expanded=False):
        st.subheader("The Bias-Variance Trade-off")
        st.markdown("""
        Choosing the right polynomial degree is a classic example of balancing the **bias-variance trade-off**:

        * **Underfitting (High Bias):**
            * Occurs when the model is too simple to capture the underlying pattern in the data.
            * *Example:* Using a degree 1 (linear) polynomial for parabolic data.
            * **Result:** Poor performance on both training and test data. The model has high "bias" because it makes strong assumptions about the data's shape that are incorrect.
            * **Visual:** The fitted line is consistently far from the data points.

        * **Overfitting (High Variance):**
            * Occurs when the model is too complex and learns the noise and random fluctuations in the training data, rather than just the underlying pattern.
            * *Example:* Using a degree 10 polynomial for data that only needs a degree 2 fit.
            * **Result:** Excellent performance on training data (low training error) but very poor performance on new, unseen test data (high test error). The model has high "variance" because it changes too much with small changes in the training data.
            * **Visual:** The fitted line wiggles through almost every data point, trying to capture every tiny fluctuation.

        The goal is to find a model complexity (e.g., polynomial degree) that generalizes well to new data – a balance between underfitting and overfitting.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*C4FjVf1Hl-yQ6JvC6oN52g.png",
                 caption="Underfitting, Just Right, and Overfitting (Source: Medium)",
                 use_column_width=True)
        st.info(
            "💡 **Tip:** Always evaluate your model on **unseen test data** to detect overfitting. A high R-squared on training data alone can be misleading.")


def topic_multiple_regression():
    st.header("📊 Multiple Regression")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Multiple Regression?")
        st.markdown("""
        **Multiple Regression** is an extension of simple linear regression. Instead of using a single independent variable to predict a dependent variable, multiple regression uses **two or more independent variables** to predict the value of a single dependent variable.

        The goal is to model the linear relationship between the multiple independent variables and the dependent variable. It helps us understand how the dependent variable changes when any one of the independent variables is varied, while the other independent variables are held constant.
        """)
        st.markdown("""
        **Daily-life Example:**
        Predicting the price of a house. A simple linear regression might just use "Size of the house". But a multiple regression model would be much more accurate by considering:
        * **Size of the house** (square footage)
        * **Number of bedrooms**
        * **Number of bathrooms**
        * **Age of the house**
        * **Location/Neighborhood score**

        Each of these factors contributes to the house price, and multiple regression helps quantify their individual impact.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("The Multiple Regression Equation")
        st.markdown("""
        The general equation for a multiple linear regression model is:
        """)
        latex_equation(r'y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon')
        st.markdown("""
        * $y$: The dependent (target) variable.
        * $x_1, x_2, \dots, x_n$: The $n$ independent (feature) variables.
        * $\beta_0$: The Y-intercept (the value of $y$ when all $x_i$ are zero).
        * $\beta_1, \beta_2, \dots, \beta_n$: The coefficients for each independent variable. Each $\beta_i$ represents the change in $y$ for a one-unit increase in $x_i$, holding all other $x$ variables constant.
        * $\epsilon$: The error term.

        The model still assumes a linear relationship between the dependent variable and *each* independent variable, but the overall relationship can be more complex due to the combination of multiple features.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Build a Multiple Regression Model")

        st.info(
            "Let's predict a 'Target' variable using multiple features. We'll generate synthetic data for demonstration.")

        col1_mr, col2_mr = st.columns(2)
        num_samples_mr = col1_mr.slider("Number of data points:", 100, 1000, 300, key='mr_num_points')
        num_features_mr = col2_mr.slider("Number of features (X variables):", 2, 5, 3, key='mr_num_features')
        noise_mr = st.slider("Noise Level:", 0.0, 100.0, 20.0, key='mr_noise_level')

        # Generate synthetic data for multiple regression
        np.random.seed(1)
        X_mr = np.random.rand(num_samples_mr, num_features_mr) * 100  # Features X1, X2, ...

        # Create true coefficients
        true_coeffs = np.random.rand(num_features_mr) * 10 - 5  # Random coeffs between -5 and 5
        true_intercept = 50

        y_mr = np.dot(X_mr, true_coeffs) + true_intercept + np.random.normal(0, noise_mr, num_samples_mr)

        feature_names = [f'Feature_{i + 1}' for i in range(num_features_mr)]
        df_mr = pd.DataFrame(X_mr, columns=feature_names)
        df_mr['Target'] = y_mr

        st.write("---")
        st.subheader("Model Training")

        # Select features and target (using the generated df)
        all_cols_mr = df_mr.columns.tolist()
        target_col_mr = 'Target'  # Always use 'Target' as the dependent variable

        selected_features_mr = st.multiselect("Select Independent Variables (X features):",
                                              [col for col in all_cols_mr if col != target_col_mr],
                                              default=feature_names[:num_features_mr],
                                              # Default to all generated features
                                              key='mr_features_select')

        if not selected_features_mr:
            st.warning("Please select at least one independent variable.")
        else:
            X_train_mr = df_mr[selected_features_mr]
            y_train_mr = df_mr[target_col_mr]

            # Fit the Multiple Linear Regression model
            model_mr = LinearRegression()
            model_mr.fit(X_train_mr, y_train_mr)

            st.write("#### Model Coefficients:")
            st.write(f"**Intercept (β0):** `{model_mr.intercept_:.3f}`")
            coefficients_df = pd.DataFrame({'Feature': selected_features_mr, 'Coefficient (β)': model_mr.coef_})
            st.dataframe(coefficients_df)
            st.info(
                "💡 Each coefficient (β) tells you how much the 'Target' is expected to change for a one-unit increase in that specific 'Feature', assuming all other features are constant.")

            # Evaluate model
            y_pred_mr = model_mr.predict(X_train_mr)
            r2_mr = r2_score(y_train_mr, y_pred_mr)
            mae_mr = mean_absolute_error(y_train_mr, y_pred_mr)
            rmse_mr = np.sqrt(mean_squared_error(y_train_mr, y_pred_mr))

            st.write("#### Model Performance Metrics:")
            st.write(f"**R-squared (R²):** `{r2_mr:.4f}`")
            st.write(f"**Mean Absolute Error (MAE):** `{mae_mr:.3f}`")
            st.write(f"**Root Mean Squared Error (RMSE):** `{rmse_mr:.3f}`")
            st.info(
                "MAE and RMSE measure the average magnitude of errors. Lower values are better. R-squared indicates the proportion of variance in the dependent variable that is predictable from the independent variables.")

            st.subheader("Visualization: Actual vs. Predicted Values")
            plot_df_mr = pd.DataFrame({'Actual': y_train_mr, 'Predicted': y_pred_mr})
            fig_actual_pred = px.scatter(plot_df_mr, x='Actual', y='Predicted',
                                         title="Actual vs. Predicted Target Values",
                                         labels={'Actual': 'Actual Target', 'Predicted': 'Predicted Target'},
                                         color_discrete_sequence=['#264653'], opacity=0.7)
            fig_actual_pred.add_trace(go.Scatter(x=[min(y_train_mr), max(y_train_mr)],
                                                 y=[min(y_train_mr), max(y_train_mr)],
                                                 mode='lines', name='Perfect Fit',
                                                 line=dict(color='#e76f51', dash='dash')))
            st.plotly_chart(fig_actual_pred, use_container_width=True)
            st.info(
                "💡 For a perfect model, all points would lie on the dashed red line (Actual = Predicted). Points far from the line indicate larger errors.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Multiple Regression in Python")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Generate synthetic data with multiple features
np.random.seed(42)
num_samples = 200
X = pd.DataFrame({
    'Feature_1': np.random.rand(num_samples) * 100,
    'Feature_2': np.random.rand(num_samples) * 50,
    'Feature_3': np.random.rand(num_samples) * 20
})

# Define a true relationship with some noise
y = (5 * X['Feature_1'] +
     2.5 * X['Feature_2'] -
     1.8 * X['Feature_3'] +
     100 + # Intercept
     np.random.normal(0, 15, num_samples)) # Add some noise

print("Sample Data Head:")
print(X.head())
print("\\nSample Target Head:")
print(y.head())

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Print the model's coefficients and intercept
print("\\nModel Coefficients:")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model's performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\\nModel Evaluation on Test Set:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

# You can also make predictions for new data points
new_data = pd.DataFrame([[70, 30, 15]], columns=X.columns)
predicted_value = model.predict(new_data)
print(f"\\nPrediction for new data ([70, 30, 15]): {predicted_value[0]:.3f}")
        """, language="python")
        st.markdown(
            "This code demonstrates how to set up, train, and evaluate a multiple linear regression model in Python using `scikit-learn`.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Interpret Coefficients")
        st.markdown("""
        Suppose you trained a multiple regression model to predict `Car Price` based on `Engine Size (liters)` and `Mileage (KM)`.

        The model coefficients are:
        * `Intercept`: \$5000
        * `Engine Size`: \$2500 per liter
        * `Mileage`: -\$0.05 per KM

        If an additional car has an `Engine Size` that is 1 liter larger, while its `Mileage` remains the same, how much would you expect its `Car Price` to change, based on this model?
        """)

        user_price_change = st.number_input("Expected change in Car Price ($):", format="%f", key='mr_task_change')

        if st.button("Check My Answer - Multiple Regression"):
            correct_change = 2500.0  # From the coefficient of Engine Size
            if abs(user_price_change - correct_change) < 0.01:
                st.success(
                    f"Correct! An increase of 1 liter in Engine Size, holding Mileage constant, would lead to an expected increase of **${correct_change:.2f}** in Car Price.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. The coefficient for 'Engine Size' directly indicates the expected change in the target for a one-unit increase in that feature, assuming others are constant. The correct answer is **${correct_change:.2f}**.")

    # 7. Bonus: Assumptions and Challenges
    with st.expander("✨ Bonus: Assumptions of Multiple Linear Regression", expanded=False):
        st.subheader("Key Assumptions for Reliable Multiple Regression")
        st.markdown("""
        Multiple Linear Regression is powerful, but it relies on several key assumptions for its results to be valid and reliable. Violating these assumptions can lead to inaccurate coefficients, p-values, and predictions.

        1.  **Linearity:** The relationship between the independent variables and the dependent variable is linear. (You can address this with polynomial terms or other transformations).
        2.  **Independence of Observations:** Observations are independent of each other. (e.g., one data point doesn't influence another).
        3.  **Homoscedasticity:** The variance of the residuals (errors) is constant across all levels of the independent variables. (i.e., the spread of errors is consistent).
        4.  **Normality of Residuals:** The residuals of the model are approximately normally distributed. (Often checked visually with a Q-Q plot or histogram of residuals).
        5.  **No Multicollinearity:** The independent variables are not highly correlated with each other.
            * **Multicollinearity:** When two or more independent variables are highly correlated, it makes it difficult for the model to estimate the individual impact of each variable. This can lead to unstable coefficients.
            * **Solutions:** Remove one of the highly correlated variables, combine them, or use techniques like PCA.

        Understanding and checking these assumptions is a crucial part of building robust regression models.
        """)
        st.info(
            "💡 **Tip:** Always inspect your data and model residuals to check if these assumptions are met before trusting your regression results too much.")


def topic_feature_scaling():
    st.header("⚖️ Feature Scaling")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Feature Scaling?")
        st.markdown("""
        **Feature Scaling** is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

        **Why is it important?**
        Many machine learning algorithms perform better when numerical input variables are scaled to a standard range or distribution. This is because:
        * **Distance-based algorithms** (like K-Nearest Neighbors, K-Means Clustering, SVMs) are highly sensitive to the magnitude of features. If one feature has a much larger range than others, it will dominate the distance calculations.
        * **Gradient Descent based algorithms** (like Linear Regression, Logistic Regression, Neural Networks) converge much faster when features are scaled. Without scaling, the cost function can have a very elongated shape, making it harder for the optimizer to find the minimum.
        * **Regularization techniques** (L1, L2) assume features are on a similar scale.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're trying to compare apples and oranges... literally!
        * **Apples:** Measured by weight (grams, e.g., 100-300g).
        * **Oranges:** Measured by diameter (cm, e.g., 6-10cm).

        If you put these two features into an algorithm that calculates "distance," the weight (grams) would completely overwhelm the diameter (cm) just because its numerical values are much larger. Scaling them brings them to a comparable range, allowing the algorithm to treat them equally important if that's what's desired.
        """)

    # 2. Types of Scaling & Equations
    with st.expander("➕ Types of Scaling & Equations", expanded=False):
        st.subheader("Common Feature Scaling Methods")
        st.markdown("""
        There are two primary methods for feature scaling:

        #### 1. Standardization (Z-score Normalization)
        * **Goal:** Rescales data to have a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1.
        * **Formula:**
            """)
        latex_equation(r'x_{\text{scaled}} = \frac{x - \mu}{\sigma}')
        st.markdown("""
            * $x$: Original feature value.
            * $\mu$: Mean of the feature.
            * $\sigma$: Standard deviation of the feature.
        * **When to use:** Ideal when features have different scales and the algorithm assumes normally distributed data (though not strictly required). It's less affected by outliers than Min-Max scaling.
        """)

        st.markdown("""
        #### 2. Normalization (Min-Max Scaling)
        * **Goal:** Rescales data to a fixed range, typically between 0 and 1 (or -1 and 1).
        * **Formula:**
            """)
        latex_equation(r'x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}')
        st.markdown("""
            * $x$: Original feature value.
            * $x_{\text{min}}$: Minimum value of the feature.
            * $x_{\text{max}}$: Maximum value of the feature.
        * **When to use:** Useful when features have a limited bounded range (e.g., image processing where pixel values are 0-255). It is sensitive to outliers, as they will determine the min/max values.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Visualize Feature Scaling Effects")

        col1_fs, col2_fs = st.columns(2)
        num_points_fs = col1_fs.slider("Number of data points:", 50, 500, 200, key='fs_num_points')
        scaling_method = col2_fs.radio("Choose Scaling Method:", ("Standardization", "Min-Max Normalization"),
                                       key='fs_method')

        st.markdown("---")
        st.markdown("#### Generate Two Features with Different Scales")
        mean1 = st.slider("Mean Feature 1:", 0, 100, 20, key='fs_mean1')
        std1 = st.slider("Std Dev Feature 1:", 1, 30, 5, key='fs_std1')
        mean2 = st.slider("Mean Feature 2:", 0, 1000, 500, key='fs_mean2')
        std2 = st.slider("Std Dev Feature 2:", 1, 100, 50, key='fs_std2')

        np.random.seed(42)
        feature1 = np.random.normal(mean1, std1, num_points_fs)
        feature2 = np.random.normal(mean2, std2, num_points_fs)

        df_original = pd.DataFrame({'Feature 1': feature1, 'Feature 2': feature2})

        st.subheader("Original Data Distribution")
        fig_orig = px.histogram(df_original, x="Feature 1", opacity=0.6, histnorm='probability density',
                                title="Feature 1 (Original)", color_discrete_sequence=['#457b9d'])
        fig_orig.add_trace(px.histogram(df_original, x="Feature 2", opacity=0.6, histnorm='probability density',
                                        title="Feature 2 (Original)", color_discrete_sequence=['#e76f51']).data[0])
        fig_orig.update_layout(barmode='overlay')
        st.plotly_chart(fig_orig, use_container_width=True)
        st.write(df_original.describe().transpose())

        st.subheader(f"Scaled Data ({scaling_method}) Distribution")
        if scaling_method == "Standardization":
            scaler = StandardScaler()
        else:  # Min-Max Normalization
            scaler = MinMaxScaler()

        df_scaled_array = scaler.fit_transform(df_original)
        df_scaled = pd.DataFrame(df_scaled_array, columns=df_original.columns)

        fig_scaled = px.histogram(df_scaled, x="Feature 1", opacity=0.6, histnorm='probability density',
                                  title="Feature 1 (Scaled)", color_discrete_sequence=['#457b9d'])
        fig_scaled.add_trace(px.histogram(df_scaled, x="Feature 2", opacity=0.6, histnorm='probability density',
                                          title="Feature 2 (Scaled)", color_discrete_sequence=['#e76f51']).data[0])
        fig_scaled.update_layout(barmode='overlay')
        st.plotly_chart(fig_scaled, use_container_width=True)
        st.write(df_scaled.describe().transpose())
        st.info(
            "💡 **Observation:** Notice how the ranges, means, and standard deviations change for the scaled data compared to the original, bringing features to a comparable scale.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Feature Scaling in Python")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a sample DataFrame with features of different scales
data = {
    'Age': np.random.randint(18, 70, 100),
    'Income': np.random.randint(20000, 150000, 100),
    'YearsExperience': np.random.randint(0, 40, 100),
    'NumChildren': np.random.randint(0, 5, 100)
}
df = pd.DataFrame(data)

print("Original Data Description:")
print(df.describe())

# 2. Standardization (Z-score normalization)
scaler_standard = StandardScaler()
# Fit and transform the data
df_scaled_standard = scaler_standard.fit_transform(df)
df_scaled_standard = pd.DataFrame(df_scaled_standard, columns=df.columns)

print("\\nStandardized Data Description:")
print(df_scaled_standard.describe())

# 3. Min-Max Normalization
scaler_minmax = MinMaxScaler()
# Fit and transform the data
df_scaled_minmax = scaler_minmax.fit_transform(df)
df_scaled_minmax = pd.DataFrame(df_scaled_minmax, columns=df.columns)

print("\\nMin-Max Normalized Data Description:")
print(df_scaled_minmax.describe())

# 4. Visualization of scaled data (e.g., using pairplot for multiple features)
# Note: For many features, scatter plots will be dense. Histograms or box plots are better for showing distribution changes.

# Example: Plotting distributions of one feature before and after scaling
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['Income'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Original Income Distribution')

sns.histplot(df_scaled_standard['Income'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Standardized Income Distribution')

sns.histplot(df_scaled_minmax['Income'], kde=True, ax=axes[2], color='red')
axes[2].set_title('Min-Max Normalized Income Distribution')

plt.tight_layout()
plt.show() # In Streamlit, use st.pyplot(fig)
        """, language="python")
        st.markdown(
            "This code demonstrates how to apply `StandardScaler` and `MinMaxScaler` from `scikit-learn` and observe their effects on data distribution.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: When to Scale?")
        st.markdown("""
        You are building a **K-Nearest Neighbors (KNN)** model for classification. Your dataset has features like `Age` (range 20-60) and `Annual Salary` (range \$30,000 - \$200,000).

        Is it generally necessary to apply feature scaling before training a KNN model on this data?
        """)

        user_choice_fs_task = st.radio("Is feature scaling necessary for KNN in this scenario?",
                                       ("Yes, it is generally necessary.", "No, it is not necessary."),
                                       key='fs_task_choice')

        if st.button("Check My Answer - Feature Scaling"):
            correct_answer = "Yes, it is generally necessary."
            if user_choice_fs_task == correct_answer:
                st.success(
                    f"Correct! **{correct_answer}** KNN is a distance-based algorithm, so features with larger magnitudes (like Salary) would unfairly dominate the distance calculations if not scaled.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. KNN relies on distances between data points. Without scaling, features with larger numerical ranges will have a disproportionately large impact on these distances. The correct answer is **{correct_answer}**.")

    # 7. Bonus: Scale `fit_transform` on training, `transform` on test
    with st.expander("✨ Bonus: The `fit_transform` and `transform` Distinction", expanded=False):
        st.subheader("Why Fit on Training and Transform on Test Data?")
        st.markdown("""
        When applying feature scaling (or any preprocessing that learns parameters, like PCA or imputation), it's crucial to distinguish between training and test sets:

        * **`scaler.fit(X_train)`:**
            * Calculates the parameters (e.g., mean and standard deviation for StandardScaler, min and max for MinMaxScaler) **only from the training data**.
            * This prevents **data leakage**, where information from the test set (which is supposed to simulate unseen data) "leaks" into the training process.

        * **`scaler.transform(X_train)`:**
            * Applies the scaling (using the parameters learned from `X_train`) to the training data.

        * **`scaler.transform(X_test)`:**
            * Applies the **same scaling parameters (learned from `X_train`)** to the test data.
            * **NEVER** `fit` the scaler on the test data! This would scale the test data independently, potentially introducing biases and making the test performance an unrealistic estimate of the model's true generalization ability.

        **In summary:**
        ```python
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()

        # Fit ONLY on training data, then transform training data
        X_train_scaled = scaler.fit_transform(X_train)

        # Use the SAME fitted scaler to transform test data
        X_test_scaled = scaler.transform(X_test)

        # Now, X_train_scaled and X_test_scaled are ready for model training/evaluation
        ```
        This practice ensures that your model's performance on the test set is a fair and unbiased estimate of its performance on new, unseen data.
        """)
        st.info(
            "💡 **Best Practice:** `fit_transform()` on training data, `transform()` on test data (and any future new data).")


def topic_k_nearest_neighbors():
    st.header("🤝 K-nearest Neighbors (KNN)")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is K-nearest Neighbors (KNN)?")
        st.markdown("""
        **K-nearest Neighbors (KNN)** is a simple, non-parametric, and lazy learning algorithm used for both **classification** and **regression** tasks.

        * **Non-parametric:** It makes no assumptions about the underlying data distribution.
        * **Lazy learning:** It doesn't build a model explicitly during the training phase. Instead, it memorizes the entire training dataset. Prediction is only made when a query point is given.

        The core idea is that similar things are near to each other. When you want to classify a new data point, KNN looks at its 'K' closest neighbors in the training data and assigns the new point the class that is most common among those K neighbors (for classification) or the average of their values (for regression).
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're trying to decide if a new, unknown fruit is an apple or an orange. You don't have a strict rule, but you look at the fruits closest to it.

        * If the 3 closest fruits (K=3) are 2 apples and 1 orange, you'd guess it's an apple.
        * If the 5 closest fruits (K=5) are 2 apples and 3 oranges, you'd guess it's an orange.

        KNN works similarly: it finds the 'neighbors' that are most like the new data point and lets them "vote" on its classification.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations", expanded=False):
        st.subheader("Distance Metric: Euclidean Distance")
        st.markdown("""
        The "closeness" between data points is typically measured using a distance metric. The most common one is **Euclidean Distance**. For two points $P=(p_1, p_2, \dots, p_n)$ and $Q=(q_1, q_2, \dots, q_n)$ in $n$-dimensional space, the Euclidean distance is:
        """)
        latex_equation(r'd(P, Q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}')
        st.markdown("""
        * $p_i$: The $i^{th}$ coordinate (feature value) of point $P$.
        * $q_i$: The $i^{th}$ coordinate (feature value) of point $Q$.
        * $\sum$: Summation over all dimensions (features).

        **How K is Chosen:**
        * Choosing an optimal 'K' is crucial.
        * A small 'K' (e.g., K=1) makes the model sensitive to noise and outliers (high variance, prone to overfitting).
        * A large 'K' smooths out the decision boundaries but might miss local patterns (high bias, prone to underfitting).
        * Cross-validation (which we'll cover next) is often used to find the best 'K'.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Classify with KNN")

        col1_knn, col2_knn = st.columns(2)
        num_samples_knn = col1_knn.slider("Number of data points:", 50, 500, 150, key='knn_samples')
        n_neighbors_knn = col2_knn.slider("Number of Neighbors (K):", 1, 15, 5, key='knn_k')

        # Generate synthetic classification data
        from sklearn.datasets import make_classification
        X_knn, y_knn = make_classification(n_samples=num_samples_knn, n_features=2, n_redundant=0,
                                           n_clusters_per_class=1, random_state=42, n_classes=2)

        # Scale data (important for distance-based algorithms like KNN)
        scaler_knn = StandardScaler()
        X_scaled_knn = scaler_knn.fit_transform(X_knn)

        # Train KNN model
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
        knn_model.fit(X_scaled_knn, y_knn)

        # Create a meshgrid for plotting decision boundaries
        x_min, x_max = X_scaled_knn[:, 0].min() - 1, X_scaled_knn[:, 0].max() + 1
        y_min, y_max = X_scaled_knn[:, 1].min() - 1, X_scaled_knn[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plotly for interactive visualization
        fig_knn = go.Figure(data=[
            go.Contour(
                z=Z,
                x=xx[0],
                y=yy[:, 0],
                colorscale='RdBu',  # Red-Blue for classification regions
                opacity=0.3,
                showscale=False,
                name='Decision Regions'
            ),
            go.Scatter(
                x=X_scaled_knn[:, 0],
                y=X_scaled_knn[:, 1],
                mode='markers',
                marker=dict(
                    color=y_knn,
                    colorscale='RdBu',
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name='Data Points'
            )
        ])
        fig_knn.update_layout(title=f'KNN Decision Boundary (K={n_neighbors_knn})',
                              xaxis_title='Feature 1 (Scaled)',
                              yaxis_title='Feature 2 (Scaled)')

        st.plotly_chart(fig_knn, use_container_width=True)
        st.info(
            "💡 **Observation:** The colored background represents the decision regions. A new point falling into a red area would be classified as red, and blue as blue.")

        st.subheader("Predict for a New Point")
        col_new_point_1, col_new_point_2 = st.columns(2)
        new_x1 = col_new_point_1.number_input("New Feature 1 value:", value=0.0, key='knn_new_x1')
        new_x2 = col_new_point_2.number_input("New Feature 2 value:", value=0.0, key='knn_new_x2')

        new_point_scaled = scaler_knn.transform([[new_x1, new_x2]])
        predicted_class = knn_model.predict(new_point_scaled)[0]
        predicted_proba = knn_model.predict_proba(new_point_scaled)[0]

        st.write(f"**Predicted Class for ({new_x1:.2f}, {new_x2:.2f}):** `{predicted_class}`")
        st.write(
            f"**Prediction Probabilities:** Class 0: `{predicted_proba[0]:.2f}`, Class 1: `{predicted_proba[1]:.2f}`")

        # Add the new point to the plot
        fig_knn.add_trace(go.Scatter(x=[new_point_scaled[0, 0]], y=[new_point_scaled[0, 1]],
                                     mode='markers',
                                     marker=dict(symbol='star', size=20, color='yellow',
                                                 line=dict(width=2, color='black')),
                                     name='New Point'))
        st.plotly_chart(fig_knn, use_container_width=True)
        st.info(
            "💡 **Try it:** Move the sliders for the new point's features and see how its predicted class changes based on K and its position.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing K-nearest Neighbors in Python")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification # For sample data

# 1. Generate sample data for classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Scale the features (VERY IMPORTANT for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Choose the number of neighbors (K)
n_neighbors = 5

# 5. Create a KNN Classifier model
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# 6. Train the model
knn.fit(X_train_scaled, y_train)

# 7. Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy with K={n_neighbors}: {accuracy:.2f}")

# 9. Predict for a new single data point
new_point = np.array([[0.5, -0.8]]) # Example new point (scaled)
predicted_class = knn.predict(new_point)[0]
print(f"Predicted class for new point {new_point[0]}: {predicted_class}")

# --- Visualization of Decision Boundary (Optional, but good for understanding) ---
# This part is more complex for general code snippet but useful for explanation
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# h = .02  # step size in the mesh
# x_min, x_max = X_scaled_knn[:, 0].min() - 1, X_scaled_knn[:, 0].max() + 1
# y_min, y_max = X_scaled_knn[:, 1].min() - 1, X_scaled_knn[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# plt.scatter(X_scaled_knn[:, 0], X_scaled_knn[:, 1], c=y_knn, cmap=cmap_bold,
#             edgecolor='k', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title(f"KNN Classification (K = {n_neighbors})")
# plt.xlabel("Feature 1 (Scaled)")
# plt.ylabel("Feature 2 (Scaled)")
# plt.show()
        """, language="python")
        st.markdown(
            "This code demonstrates the basic steps of using KNN for classification, including data splitting and feature scaling.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: KNN Prediction")
        st.markdown("""
        Consider the following 2D data points with their classes:
        * (1, 2) -> Class A
        * (2, 3) -> Class A
        * (4, 5) -> Class B
        * (5, 4) -> Class B

        Now, you have a **new point at (3, 3)**.
        If you use **K=1 (1-nearest neighbor)**, what would be the predicted class for this new point?
        """)

        user_knn_class = st.radio("Predicted Class for (3,3) with K=1:", ("Class A", "Class B"), key='knn_task_class')

        if st.button("Check My Answer - KNN"):
            # Calculate distances from (3,3)
            # (1,2): sqrt((3-1)^2 + (3-2)^2) = sqrt(2^2 + 1^2) = sqrt(4+1) = sqrt(5) approx 2.23
            # (2,3): sqrt((3-2)^2 + (3-3)^2) = sqrt(1^2 + 0^2) = sqrt(1) = 1
            # (4,5): sqrt((3-4)^2 + (3-5)^2) = sqrt((-1)^2 + (-2)^2) = sqrt(1+4) = sqrt(5) approx 2.23
            # (5,4): sqrt((3-5)^2 + (3-4)^2) = sqrt((-2)^2 + (-1)^2) = sqrt(4+1) = sqrt(5) approx 2.23

            # The closest point is (2,3) which belongs to Class A.
            correct_class = "Class A"

            if user_knn_class == correct_class:
                st.success(
                    f"Correct! The closest point to (3,3) is (2,3) with a distance of 1, which belongs to **Class A**. So, with K=1, the prediction is Class A.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. Calculate the Euclidean distance from (3,3) to each data point and find the single closest one. The correct answer is **{correct_class}**.")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: KNN Steps with a Visual Example", expanded=False):
        st.subheader("KNN Classification Steps for a New Point")
        st.markdown("""
        Let's classify a new **Green Star** point using K=3 neighbors from existing **Blue Squares** and **Red Circles**.

        ---
        #### **Step 1: Calculate Distances**
        * Measure the distance from the new Green Star point to *every* existing data point (Blue Squares and Red Circles). Euclidean distance is commonly used.
        """)
        # Conceptual Plot 1: Points and New Point
        knn_bonus_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 2.5],
            'y': [1, 2, 1, 4, 5, 3.5],
            'Type': ['Existing', 'Existing', 'Existing', 'Existing', 'Existing', 'New Point'],
            'Class': ['Blue', 'Blue', 'Red', 'Red', 'Red', 'Unknown']
        })
        fig_bonus_knn1 = px.scatter(knn_bonus_data, x='x', y='y', color='Class', symbol='Type',
                                    title="Step 1: New Point and Existing Data",
                                    color_discrete_map={'Blue': 'blue', 'Red': 'red', 'Unknown': 'green'},
                                    symbol_map={'Existing': 'square', 'New Point': 'star'},
                                    size=[10] * 5 + [15])
        st.plotly_chart(fig_bonus_knn1, use_container_width=True)
        st.info("💡 **Visualization 1:** The green star is the point we want to classify.")

        st.markdown("""
        ---
        #### **Step 2: Find K-Nearest Neighbors**
        * Select the 'K' data points that have the smallest distances to the new point.
            * *Example (K=3):* If the 3 closest points are one Blue Square and two Red Circles.
        """)
        # Conceptual Plot 2: Highlight K-neighbors
        fig_bonus_knn2 = px.scatter(knn_bonus_data, x='x', y='y', color='Class', symbol='Type',
                                    title="Step 2: Identify K-Nearest Neighbors (K=3)",
                                    color_discrete_map={'Blue': 'blue', 'Red': 'red', 'Unknown': 'green'},
                                    symbol_map={'Existing': 'square', 'New Point': 'star'},
                                    size=[10] * 5 + [15])
        # Manually add circles around conceptual neighbors
        # These are illustrative and depend on the exact data, here I'll pick some
        # Let's assume (2,2) Blue, (3,1) Red, (4,4) Red are the 3 closest
        closest_neighbors_coords = [(2, 2), (3, 1), (4, 4)]  # Conceptual coordinates
        closest_neighbors_classes = ['Blue', 'Red', 'Red']  # Conceptual classes

        # Add circles around the conceptual closest neighbors
        for i, (nx, ny) in enumerate(closest_neighbors_coords):
            fig_bonus_knn2.add_shape(type="circle", xref="x", yref="y",
                                     x0=nx - 0.3, y0=ny - 0.3, x1=nx + 0.3, y1=ny + 0.3,
                                     line_color="purple", line_width=2, opacity=0.5)
        st.plotly_chart(fig_bonus_knn2, use_container_width=True)
        st.info("💡 **Visualization 2:** The purple circles highlight the 3 closest neighbors to the green star.")

        st.markdown("""
        ---
        #### **Step 3: Vote/Aggregate and Classify**
        * For classification: The new point is assigned the class that is most frequent among its K-nearest neighbors (majority vote).
            * *Example (K=3):* If 1 neighbor is Blue and 2 neighbors are Red, the Green Star is classified as **Red**.
        * For regression: The new point's value is the average of the values of its K-nearest neighbors.
        """)
        # Conceptual Plot 3: Final Classification
        knn_bonus_data.loc[knn_bonus_data['Type'] == 'New Point', 'Class'] = 'Red'  # Assign predicted class
        fig_bonus_knn3 = px.scatter(knn_bonus_data, x='x', y='y', color='Class', symbol='Type',
                                    title="Step 3: New Point Classified by Majority Vote (K=3)",
                                    color_discrete_map={'Blue': 'blue', 'Red': 'red', 'Unknown': 'green'},
                                    symbol_map={'Existing': 'square', 'New Point': 'star'},
                                    size=[10] * 5 + [15])
        st.plotly_chart(fig_bonus_knn3, use_container_width=True)
        st.info(
            "💡 **Visualization 3:** The green star is now classified as 'Red' based on its neighbors' majority vote.")

        st.markdown("""
        ---
        **Key Considerations for KNN:**
        * **Feature Scaling:** Essential, as distance metrics are sensitive to feature scales.
        * **Computational Cost:** Can be slow for very large datasets during prediction, as it needs to calculate distances to many points.
        * **Curse of Dimensionality:** Performance degrades in very high-dimensional spaces.
        """)


def topic_decision_tree():
    st.header("🌳 Decision Tree")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is a Decision Tree?")
        st.markdown("""
        A **Decision Tree** is a supervised machine learning algorithm that can be used for both classification and regression tasks. It works by creating a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

        It's called a "tree" because it literally builds a tree-like structure, where:
        * **Nodes:** Represent a test on a feature (e.g., "Is Age > 30?").
        * **Branches:** Represent the outcome of the test (e.g., "Yes" or "No").
        * **Leaf Nodes:** Represent the final decision or prediction (e.g., "Buy Product" or "Don't Buy").

        The goal is to split the data into increasingly pure subsets based on features, leading to clear decisions at the leaves.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine deciding whether to "Play Outside" or "Stay Inside" based on weather conditions.

        * **Root Node:** "Is it Sunny?"
            * **Branch 1 (Yes, Sunny):** "Is Temperature > 25°C?"
                * **Branch 1a (Yes, Hot):** "Play Outside" (Leaf Node)
                * **Branch 1b (No, Not Hot):** "Stay Inside" (Maybe too cold for outside activities)
            * **Branch 2 (No, Not Sunny):** "Is it Raining?"
                * **Branch 2a (Yes, Raining):** "Stay Inside" (Leaf Node)
                * **Branch 2b (No, Not Raining):** "Play Outside" (Maybe cloudy but fine)

        This structured decision-making process is exactly what a decision tree models.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations: Impurity Measures", expanded=False):
        st.subheader("How Decision Trees Make Splits: Impurity")
        st.markdown("""
        Decision trees decide where to split by looking for the feature and split point that best reduces the "impurity" of the resulting subsets. Common impurity measures for classification are **Gini Impurity** and **Entropy**.

        #### **Gini Impurity**
        * Measures the probability of misclassifying a randomly chosen element in the dataset if it were randomly labeled according to the distribution of labels in the subset. A Gini Impurity of 0 means the subset is perfectly pure (all elements belong to the same class).
        """)
        latex_equation(r'Gini(p) = 1 - \sum_{i=1}^{C} p_i^2')
        st.markdown("""
        * $p_i$: The proportion of observations belonging to class $i$ in the node.
        * $C$: The total number of classes.

        #### **Entropy**
        * Measures the disorder or uncertainty in a node. An entropy of 0 means the node is perfectly pure. Higher entropy means more mixed classes.
        """)
        latex_equation(r'Entropy(p) = - \sum_{i=1}^{C} p_i \log_2(p_i)')
        st.markdown("""
        * $p_i$: The proportion of observations belonging to class $i$ in the node.
        * $\log_2$: Logarithm base 2.

        #### **Information Gain**
        * The goal of a split is to maximize **Information Gain**, which is the reduction in entropy (or Gini impurity) after a dataset is split on an attribute.
        """)
        latex_equation(r'IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)')
        st.markdown("""
        * $IG(S, A)$: Information Gain of splitting dataset $S$ on attribute $A$.
        * $Entropy(S)$: Entropy of the original dataset $S$.
        * $Values(A)$: All possible values of attribute $A$.
        * $S_v$: Subset of $S$ where attribute $A$ has value $v$.
        * $|S_v| / |S|$: Proportion of data points in subset $S_v$.

        The tree algorithm iterates through all possible features and split points to find the one that yields the highest information gain (or lowest impurity) at each step.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Build and Visualize a Decision Tree")

        st.info(
            "Let's generate some simple 2D classification data and see how a Decision Tree learns to separate classes.")

        col1_dt, col2_dt = st.columns(2)
        num_samples_dt = col1_dt.slider("Number of data points:", 50, 500, 150, key='dt_samples')
        max_depth_dt = col2_dt.slider("Max Tree Depth:", 1, 10, 3, key='dt_max_depth')

        criterion_dt = st.radio("Splitting Criterion:", ("gini", "entropy"), key='dt_criterion')

        # Generate synthetic classification data
        from sklearn.datasets import make_classification
        X_dt, y_dt = make_classification(n_samples=num_samples_dt, n_features=2, n_redundant=0,
                                         n_informative=2, n_clusters_per_class=1, random_state=42)

        # Split data for training the tree
        X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
            X_dt, y_dt, test_size=0.2, random_state=42, stratify=y_dt
        )

        # Train Decision Tree Classifier
        dt_classifier = DecisionTreeClassifier(max_depth=max_depth_dt, criterion=criterion_dt, random_state=42)
        dt_classifier.fit(X_train_dt, y_train_dt)

        y_pred_dt = dt_classifier.predict(X_test_dt)
        accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
        st.write(f"**Test Set Accuracy:** `{accuracy_dt:.3f}`")

        st.subheader("Decision Tree Structure")
        st.info("The visualization below shows the rules the tree learned. Each node represents a split.")

        # Visualize the tree using graphviz
        dot_data = export_graphviz(dt_classifier,
                                   feature_names=['Feature 1', 'Feature 2'],
                                   class_names=[str(c) for c in np.unique(y_dt)],
                                   filled=True, rounded=True,
                                   special_characters=True,
                                   out_file=None)
        graph = graphviz.Source(dot_data)
        st.graphviz_chart(graph)
        st.markdown("---")

        st.subheader("Decision Boundary Visualization")
        st.info("The colored regions show how the tree divides the feature space to make predictions.")

        # Create a meshgrid for plotting decision boundary
        x_min, x_max = X_dt[:, 0].min() - 1, X_dt[:, 0].max() + 1
        y_min, y_max = X_dt[:, 1].min() - 1, X_dt[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = dt_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig_dt_boundary = go.Figure(data=[
            go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z,
                       showscale=False, colorscale='RdBu', opacity=0.3, name='Decision Boundary'),
            go.Scatter(x=X_train_dt[:, 0], y=X_train_dt[:, 1], mode='markers',
                       marker=dict(color=y_train_dt, colorscale='RdBu', size=8,
                                   line=dict(width=1, color='DarkSlateGrey')),
                       name='Training Data'),
            go.Scatter(x=X_test_dt[:, 0], y=X_test_dt[:, 1], mode='markers',
                       marker=dict(color=y_test_dt, colorscale='RdBu', size=10, symbol='circle',
                                   line=dict(width=2, color='black')),
                       name='Test Data (Bold Border)')
        ])
        fig_dt_boundary.update_layout(title="Decision Tree Decision Boundary",
                                      xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig_dt_boundary, use_container_width=True)
        st.warning(
            "⚠️ **Note:** For `graphviz` visualization, ensure it's installed on your system (e.g., `brew install graphviz` on macOS, `sudo apt-get install graphviz` on Linux).")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Decision Trees in Python")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import graphviz # For visualizing the tree

# 1. Prepare your data (features X, target y)
# Example: Using a built-in dataset or synthetic data
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2] # Using only first two features for simplicity in plotting
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Create a Decision Tree Classifier instance
# max_depth: Limits the depth of the tree to prevent overfitting
# criterion: 'gini' for Gini impurity, 'entropy' for Information Gain
dt_model = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)

# 3. Train the model
dt_model.fit(X_train, y_train)

# 4. Make predictions
y_pred = dt_model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")

# 6. Visualize the Decision Tree (Method 1: Matplotlib)
plt.figure(figsize=(12, 8))
plot_tree(dt_model, filled=True, feature_names=['sepal length (cm)', 'sepal width (cm)'],
          class_names=iris.target_names, rounded=True)
plt.title("Decision Tree Visualization (Matplotlib)")
plt.show() # In Streamlit: st.pyplot(plt.gcf())

# 7. Visualize the Decision Tree (Method 2: Graphviz - more professional)
# Requires graphviz system installation and 'graphviz' pip package
dot_data = export_graphviz(dt_model,
                           feature_names=['sepal length (cm)', 'sepal width (cm)'],
                           class_names=iris.target_names,
                           filled=True, rounded=True,
                           special_characters=True,
                           out_file=None)
graph = graphviz.Source(dot_data)
# graph.render("iris_decision_tree", view=True) # Saves to file and opens
print("\\nGraphviz visualization generated (check your output or Streamlit app).")
# In Streamlit: st.graphviz_chart(graph)
        """, language="python")
        st.markdown(
            "This code demonstrates how to train a Decision Tree classifier, evaluate it, and visualize its structure using both Matplotlib and Graphviz.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Interpret a Decision Rule")
        st.markdown("""
        Consider a node in a Decision Tree with the following rule:
        `Feature_A <= 5.5`

        If a new data point has `Feature_A = 6.0`, which branch would it follow?
        """)

        user_branch_dt_task = st.radio("The data point would go to the branch where:",
                                       ("Feature_A <= 5.5 is TRUE", "Feature_A <= 5.5 is FALSE"),
                                       key='dt_task_branch')

        if st.button("Check My Answer - Decision Tree"):
            correct_answer = "Feature_A <= 5.5 is FALSE"
            if user_branch_dt_task == correct_answer:
                st.success(
                    f"Correct! Since 6.0 is NOT less than or equal to 5.5, the condition `Feature_A <= 5.5` is FALSE, and the data point follows that branch.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. Remember that `Feature_A <= 5.5` is the condition. If the value is 6.0, this condition is false. The correct answer is **{correct_answer}**.")

    # 7. Bonus: How the Algorithm Works (Step-by-Step)
    with st.expander("✨ Bonus: How Decision Trees Grow (Conceptually)", expanded=False):
        st.subheader("The Recursive Binary Splitting Process")
        st.markdown("""
        Decision Trees are built using a greedy algorithm called **Recursive Binary Splitting**. At each step, the algorithm chooses the best split (feature and threshold) that divides the data into two subsets, and then it recursively applies the same logic to each subset.

        **Let's imagine a simple dataset with two features (X1, X2) and two classes (Red, Blue).**

        #### **Step 1: Start at the Root Node**
        * The entire dataset is at the root. The algorithm looks at all possible features and all possible split points for each feature.
        * It calculates the impurity (e.g., Gini) for each potential split and chooses the one that maximizes information gain (reduces impurity the most).
        * *Example:* The best split might be `X1 <= 0.5`. This divides the data into two regions.
        """)
        # Conceptual plot for step 1
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*L744_H5H605-t0q_j-v-Gg.png",
                 caption="Conceptual Step 1: Initial Split at the Root (Source: Medium)",
                 use_column_width=True)

        st.markdown("""
        ---
        #### **Step 2: Recursive Splitting**
        * For each of the two new subsets created in Step 1, the algorithm repeats the process: it finds the *next best split* within *that subset*.
        * This continues until a stopping condition is met (e.g., maximum depth reached, minimum number of samples in a leaf, or node becomes pure).
        * *Example:* In the `X1 > 0.5` region, the next best split might be `X2 <= 0.7`.
        """)
        # Conceptual plot for step 2
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*C4FjVf1Hl-yQ6JvC6oN52g.png",
                 # Reusing for conceptual split
                 caption="Conceptual Step 2: Further Splits (Source: Medium - adapted)",
                 use_column_width=True)

        st.markdown("""
        ---
        #### **Step 3: Leaf Nodes**
        * When a stopping condition is met, a node becomes a **leaf node**. This node represents a final decision or prediction. The class assigned to this leaf is usually the majority class of the data points within that leaf.

        This greedy, top-down approach constructs the tree. It's "greedy" because it always chooses the best split at the current step without considering if it leads to a globally optimal tree.
        """)
        st.info(
            "💡 **Key takeaway:** The tree grows by repeatedly finding the best single question (split) to ask at each stage to separate the classes as much as possible.")


def topic_confusion_matrix():
    st.header("🔢 Confusion Matrix")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is a Confusion Matrix?")
        st.markdown("""
        A **Confusion Matrix** is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows for the visualization of the performance of an algorithm.

        It summarizes the number of correct and incorrect predictions made by a classifier, broken down by each class. This breakdown is crucial for understanding where your model is succeeding and failing, especially with imbalanced datasets.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine a medical test designed to detect a rare disease.

        |                   | **Actual Positive (Has Disease)** | **Actual Negative (No Disease)** |
        | :---------------- | :-------------------------------- | :------------------------------- |
        | **Predicted Positive** | Test correctly says "Positive" | Test incorrectly says "Positive" |
        | **Predicted Negative** | Test incorrectly says "Negative" | Test correctly says "Negative"   |

        This matrix helps evaluate how often the test correctly identifies sick people (and healthy people) and how often it makes mistakes (false alarms or missed diagnoses).
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Metrics from Confusion Matrix", expanded=False):
        st.subheader("Understanding the Cells and Derived Metrics")
        st.markdown("""
        For a binary classification problem (two classes, e.g., Positive/Negative), the confusion matrix has four key components:

        * **True Positives (TP):** Actual is Positive, Predicted is Positive. (Correctly identified positive cases)
        * **True Negatives (TN):** Actual is Negative, Predicted is Negative. (Correctly identified negative cases)
        * **False Positives (FP):** Actual is Negative, Predicted is Positive. (Type I error - "False Alarm")
        * **False Negatives (FN):** Actual is Positive, Predicted is Negative. (Type II error - "Missed Opportunity")

        From these, several important metrics can be calculated:

        #### **Accuracy**
        * The proportion of total correct predictions (both positive and negative).
        """)
        latex_equation(r'Accuracy = \frac{TP + TN}{TP + TN + FP + FN}')
        st.markdown("""
        * **Caution:** Can be misleading for imbalanced datasets. If 95% of cases are Negative, a model that always predicts Negative will have 95% accuracy.

        #### **Precision (Positive Predictive Value)**
        * Of all predicted positive cases, what proportion were actually positive?
        """)
        latex_equation(r'Precision = \frac{TP}{TP + FP}')
        st.markdown("""
        * High precision means fewer false alarms. Important when the cost of a False Positive is high (e.g., flagging a healthy person with disease).

        #### **Recall (Sensitivity, True Positive Rate)**
        * Of all actual positive cases, what proportion did the model correctly identify?
        """)
        latex_equation(r'Recall = \frac{TP}{TP + FN}')
        st.markdown("""
        * High recall means fewer missed actual positive cases. Important when the cost of a False Negative is high (e.g., missing a disease diagnosis).

        #### **F1-Score**
        * The harmonic mean of Precision and Recall. It provides a single score that balances both. Useful for imbalanced datasets.
        """)
        latex_equation(r'F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}')
        st.markdown("""
        * A high F1-score means the model has good precision and good recall.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Analyze a Confusion Matrix")
        st.info(
            "Manually input values for True Positives, True Negatives, False Positives, and False Negatives to see how metrics change.")

        col1_cm, col2_cm = st.columns(2)
        tp = col1_cm.number_input("True Positives (TP):", min_value=0, value=80, key='cm_tp')
        fn = col2_cm.number_input("False Negatives (FN):", min_value=0, value=20, key='cm_fn')
        fp = col1_cm.number_input("False Positives (FP):", min_value=0, value=10, key='cm_fp')
        tn = col2_cm.number_input("True Negatives (TN):", min_value=0, value=90, key='cm_tn')

        # Create a dummy confusion matrix for visualization
        conf_matrix_data = np.array([[tn, fp], [fn, tp]])  # sklearn format: [[TN, FP], [FN, TP]]

        st.subheader("Confusion Matrix Heatmap")
        fig_cm = px.imshow(conf_matrix_data,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Negative', 'Positive'],
                           y=['Negative', 'Positive'],
                           text_auto=True,
                           color_continuous_scale='Blues',
                           title="Confusion Matrix")
        fig_cm.update_xaxes(side="bottom")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Calculated Metrics")
        if (tp + tn + fp + fn) == 0:
            st.warning("Please enter some values for the confusion matrix.")
        else:
            total_predictions = tp + tn + fp + fn
            accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            st.write(f"**Total Predictions:** `{total_predictions}`")
            st.write(f"**Accuracy:** `{accuracy:.3f}`")
            st.write(f"**Precision:** `{precision:.3f}`")
            st.write(f"**Recall:** `{recall:.3f}`")
            st.write(f"**F1-Score:** `{f1:.3f}`")
            st.info(
                "💡 **Observation:** Change the TP, TN, FP, FN values and see how the metrics (Accuracy, Precision, Recall, F1-Score) respond.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Calculating Confusion Matrix and Metrics in Python")
        st.code("""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Actual vs. Predicted labels
y_true = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1] # Actual labels (0=Negative, 1=Positive)
y_pred = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] # Predicted labels

# 1. Generate the Confusion Matrix
# Rows: Actual labels, Columns: Predicted labels
# For binary: [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract values for clarity
TN, FP, FN, TP = cm.ravel()
print(f"\\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# 2. Calculate various metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred) # By default for positive class (1)
recall = recall_score(y_true, y_pred)     # By default for positive class (1)
f1 = f1_score(y_true, y_pred)             # By default for positive class (1)

print(f"\\nAccuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# 3. Visualize the Confusion Matrix (using Seaborn)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show() # In Streamlit: st.pyplot(plt.gcf())
        """, language="python")
        st.markdown(
            "This code shows how to compute a confusion matrix and derive key classification metrics using `scikit-learn`.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Calculate Metrics for a Spam Filter")
        st.markdown("""
        A spam filter classified 100 emails. Here are the results:
        * **Actual Spam (Positive):** 20 emails
            * Correctly identified as Spam (TP): 18
            * Incorrectly identified as Not Spam (FN): 2
        * **Actual Not Spam (Negative):** 80 emails
            * Incorrectly identified as Spam (FP): 5
            * Correctly identified as Not Spam (TN): 75

        Calculate the **Precision** and **Recall** of this spam filter.
        """)

        col1_task_cm, col2_task_cm = st.columns(2)
        user_precision = col1_task_cm.number_input("Your Precision (e.g., 0.90):", format="%.3f",
                                                   key='cm_task_precision')
        user_recall = col2_task_cm.number_input("Your Recall (e.g., 0.80):", format="%.3f", key='cm_task_recall')

        if st.button("Check My Calculations - Confusion Matrix"):
            # Given values
            TP_task = 18
            FN_task = 2
            FP_task = 5
            TN_task = 75

            correct_precision = TP_task / (TP_task + FP_task)
            correct_recall = TP_task / (TP_task + FN_task)

            precision_feedback = "Incorrect."
            if abs(user_precision - correct_precision) < 0.001:
                precision_feedback = "Correct!"
            st.markdown(
                f"**Precision:** Your answer: `{user_precision:.3f}`, Correct: `{correct_precision:.3f}`. **{precision_feedback}**")

            recall_feedback = "Incorrect."
            if abs(user_recall - correct_recall) < 0.001:
                recall_feedback = "Correct!"
            st.markdown(
                f"**Recall:** Your answer: `{user_recall:.3f}`, Correct: `{correct_recall:.3f}`. **{recall_feedback}**")

            if precision_feedback == "Correct!" and recall_feedback == "Correct!":
                st.balloons()
                st.success("Excellent! You correctly calculated Precision and Recall.")
            else:
                st.warning(
                    "Review the formulas for Precision and Recall. Precision focuses on predicted positives, Recall on actual positives.")

    # 7. Bonus: Trade-offs and Imbalanced Data
    with st.expander("✨ Bonus: Precision-Recall Trade-off & Imbalanced Data", expanded=False):
        st.subheader("Beyond Accuracy: Why Metrics Matter")
        st.markdown("""
        Accuracy can be misleading, especially when dealing with **imbalanced datasets** (where one class is much more frequent than the other).

        * **Example: Fraud Detection**
            * Actual Fraud (Positive): 1% of transactions
            * Actual Non-Fraud (Negative): 99% of transactions
            * A model that *always* predicts "Non-Fraud" would have 99% accuracy! But it would miss all fraud (Recall = 0).

        This is where Precision and Recall become critical:

        * **High Precision, Low Recall:** The model is very cautious about predicting positive. When it does predict positive, it's usually right, but it misses many actual positive cases.
            * *Scenario:* A spam filter that only flags very obvious spam. You get almost no false spam (good precision), but a lot of spam might get through (bad recall).
        * **High Recall, Low Precision:** The model is very good at finding all positive cases, but it also flags many negative cases as positive.
            * *Scenario:* A medical test that is very sensitive to a disease. It catches almost all sick people (good recall), but it also gives many false alarms to healthy people (bad precision).

        The choice between prioritizing Precision or Recall depends on the **cost of errors** in your specific application. F1-score provides a balance, but sometimes you need to optimize for one over the other.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*C4FjVf1Hl-yQ6JvC6oN52g.png",
                 # Reusing for conceptual trade-off
                 caption="Conceptual Precision-Recall Trade-off (Source: Medium - adapted)",
                 use_column_width=True)
        st.info(
            "💡 **Key takeaway:** Always look at Precision, Recall, and F1-Score, especially for imbalanced classification problems, to get a full picture of your model's performance.")


def topic_logistic_regression():
    st.header("📈 Logistic Regression")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Logistic Regression?")
        st.markdown("""
        **Logistic Regression** is a statistical model used for **binary classification** tasks (predicting one of two classes, e.g., Yes/No, Spam/Not Spam, True/False). Despite its name, it's a classification algorithm, not a regression algorithm in the traditional sense, as its output is a probability.

        It works by using a **sigmoid function** to map the output of a linear equation (similar to linear regression) to a probability value between 0 and 1. This probability is then used to classify an observation into one of the two classes.
        """)
        st.markdown("""
        **Daily-life Example:**
        Predicting whether a customer will "Click" on an advertisement (Yes/No) based on how long they spent on a webpage.

        * Linear Regression might predict a "click score" that could be negative or greater than 1, which doesn't make sense for a probability.
        * Logistic Regression takes this "click score" and squashes it using the sigmoid function into a probability (e.g., 0.85 probability of clicking). If this probability is above a certain threshold (e.g., 0.5), it classifies as "Click".

        It's widely used because it's simple, interpretable, and efficient.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations: Sigmoid and Log-Odds", expanded=False):
        st.subheader("The Sigmoid Function and Log-Odds")
        st.markdown("""
        Logistic Regression starts with a linear combination of inputs, similar to linear regression:
        """)
        latex_equation(r'z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n')
        st.markdown("""
        * $z$: The "logit" or "log-odds" score. This can be any real number.

        To convert this $z$ score into a probability between 0 and 1, Logistic Regression uses the **Sigmoid Function** (also called the Logistic Function):
        """)
        latex_equation(r'P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}')
        st.markdown("""
        * $P(y=1|x)$: The probability that the dependent variable $y$ is 1 (the positive class), given the input features $x$.
        * $e$: Euler's number (approximately 2.71828).

        #### **Decision Boundary**
        * Typically, if $P(y=1|x) \ge 0.5$, the observation is classified as 1 (Positive class).
        * If $P(y=1|x) < 0.5$, the observation is classified as 0 (Negative class).
        * The point where $P(y=1|x) = 0.5$ corresponds to $z=0$. This defines the **decision boundary**.

        #### **Cost Function (Loss Function)**
        * Logistic Regression uses a **log-loss** or **binary cross-entropy** cost function, which penalizes incorrect probabilistic predictions. The goal is to minimize this cost function during training.
        """)
        latex_equation(
            r'J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_\theta(x^{(i)})) + (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))]')
        st.markdown("""
        * $m$: Number of training examples.
        * $y^{(i)}$: Actual label (0 or 1) for example $i$.
        * $h_\theta(x^{(i)})$: Predicted probability for example $i$ (output of sigmoid).
        * This function heavily penalizes confident wrong predictions.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Train and Visualize Logistic Regression")

        st.info(
            "Let's generate some 2D binary classification data and see the decision boundary learned by Logistic Regression.")

        col1_lr, col2_lr = st.columns(2)
        num_samples_logr = col1_lr.slider("Number of data points:", 50, 500, 200, key='logr_samples')
        regularization_c = col2_lr.slider("Regularization Strength (C):", 0.01, 10.0, 1.0, 0.01, key='logr_c')
        st.info(
            "💡 **C** is the inverse of regularization strength. Smaller C means stronger regularization (more penalty for complex models, helps prevent overfitting).")

        # Generate synthetic classification data
        from sklearn.datasets import make_blobs
        X_logr, y_logr = make_blobs(n_samples=num_samples_logr, centers=2, cluster_std=1.5, random_state=42)

        # Train/Test Split
        X_train_logr, X_test_logr, y_train_logr, y_test_logr = train_test_split(
            X_logr, y_logr, test_size=0.2, random_state=42, stratify=y_logr
        )

        # Train Logistic Regression
        log_reg_model = LogisticRegression(C=regularization_c, solver='liblinear', random_state=42)
        log_reg_model.fit(X_train_logr, y_train_logr)

        y_pred_logr = log_reg_model.predict(X_test_logr)
        accuracy_logr = accuracy_score(y_test_logr, y_pred_logr)
        st.write(f"**Test Set Accuracy:** `{accuracy_logr:.3f}`")

        st.subheader("Decision Boundary Visualization")
        st.info(
            "The line represents the decision boundary where the model switches its prediction from one class to another.")

        # Create a meshgrid for plotting decision boundary
        x_min, x_max = X_logr[:, 0].min() - 1, X_logr[:, 0].max() + 1
        y_min, y_max = X_logr[:, 1].min() - 1, X_logr[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z_logr = log_reg_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_logr = Z_logr.reshape(xx.shape)

        fig_logr_boundary = go.Figure(data=[
            go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z_logr,
                       showscale=False, colorscale='RdBu', opacity=0.3, name='Decision Boundary'),
            go.Scatter(x=X_train_logr[:, 0], y=X_train_logr[:, 1], mode='markers',
                       marker=dict(color=y_train_logr, colorscale='RdBu', size=8,
                                   line=dict(width=1, color='DarkSlateGrey')),
                       name='Training Data'),
            go.Scatter(x=X_test_logr[:, 0], y=X_test_logr[:, 1], mode='markers',
                       marker=dict(color=y_test_logr, colorscale='RdBu', size=10, symbol='circle',
                                   line=dict(width=2, color='black')),
                       name='Test Data (Bold Border)')
        ])
        fig_logr_boundary.update_layout(title="Logistic Regression Decision Boundary",
                                        xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig_logr_boundary, use_container_width=True)

        st.subheader("Predict Probability for a New Point")
        col_pred_x1, col_pred_x2 = st.columns(2)
        new_x1 = col_pred_x1.number_input("New Feature 1 value:", value=float(X_logr[:, 0].mean()), key='logr_new_x1')
        new_x2 = col_pred_x2.number_input("New Feature 2 value:", value=float(X_logr[:, 1].mean()), key='logr_new_x2')

        new_point = np.array([[new_x1, new_x2]])
        predicted_prob = log_reg_model.predict_proba(new_point)[0, 1]  # Probability of class 1
        predicted_class = log_reg_model.predict(new_point)[0]

        st.write(f"For a point ({new_x1:.2f}, {new_x2:.2f}):")
        st.write(f"**Predicted Probability (Class 1):** `{predicted_prob:.3f}`")
        st.write(f"**Predicted Class:** `{predicted_class}`")
        st.info("💡 The model predicts a probability, and then classifies based on a threshold (default 0.5).")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Logistic Regression in Python")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prepare your data (features X, binary target y)
# Example: Using a synthetic dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Create a Logistic Regression model instance
# C: Inverse of regularization strength. Smaller values specify stronger regularization.
# solver: Algorithm to use in the optimization problem. 'liblinear' is good for small datasets.
log_reg_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)

# 3. Train the model
log_reg_model.fit(X_train, y_train)

# 4. Make predictions
y_pred = log_reg_model.predict(X_test)
y_pred_proba = log_reg_model.predict_proba(X_test) # Get probabilities

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Get model coefficients (interpret feature importance)
print(f"\\nCoefficients: {log_reg_model.coef_}")
print(f"Intercept: {log_reg_model.intercept_}")

# 7. Predict probability for a new single data point
new_point = np.array([[0.5, -0.8]]) # Example new point
predicted_prob_class_0 = log_reg_model.predict_proba(new_point)[0, 0]
predicted_prob_class_1 = log_reg_model.predict_proba(new_point)[0, 1]
predicted_class = log_reg_model.predict(new_point)[0]

print(f"\\nPrediction for new point {new_point[0]}:")
print(f"Probability of Class 0: {predicted_prob_class_0:.3f}")
print(f"Probability of Class 1: {predicted_prob_class_1:.3f}")
print(f"Predicted Class: {predicted_class}")

# 8. Visualize the Sigmoid Function (conceptual)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z_values)

plt.figure(figsize=(8, 5))
plt.plot(z_values, sigmoid_values)
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.axvline(0, color='green', linestyle='--', label='z=0')
plt.title('Sigmoid Function')
plt.xlabel('z (Log-Odds)')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.show() # In Streamlit: st.pyplot(plt.gcf())
        """, language="python")
        st.markdown(
            "This code demonstrates training a Logistic Regression model, evaluating it, inspecting its coefficients, and predicting probabilities for new data. It also includes a conceptual plot of the sigmoid function.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Interpret Probability")
        st.markdown("""
        A Logistic Regression model predicts the probability of a customer clicking on an ad.

        If the model outputs a probability of **0.72** for a new customer, and the classification threshold is **0.5**, what would be the model's final classification for this customer?
        """)

        user_classification_logr = st.radio("The customer would be classified as:",
                                            ("Click (Positive Class)", "Not Click (Negative Class)"),
                                            key='logr_task_class')

        if st.button("Check My Classification - Logistic Regression"):
            predicted_prob_task = 0.72
            threshold_task = 0.5

            correct_answer = "Click (Positive Class)" if predicted_prob_task >= threshold_task else "Not Click (Negative Class)"

            if user_classification_logr == correct_answer:
                st.success(
                    f"Correct! Since `{predicted_prob_task}` is greater than or equal to the threshold of `{threshold_task}`, the customer is classified as **'{correct_answer}'**.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. Remember that if the probability is above the threshold, it's classified as the positive class. The correct answer is **'{correct_answer}'**.")

    # 7. Bonus: Sigmoid Function Visualization
    with st.expander("✨ Bonus: The Sigmoid Function in Action", expanded=False):
        st.subheader("Transforming Linear Output to Probability")
        st.markdown("""
        The Sigmoid function is the heart of Logistic Regression. It takes any real-valued number ($z$, which is the linear combination of features) and squashes it into a value between 0 and 1, making it interpretable as a probability.

        * **If $z$ is a large positive number:** $e^{-z}$ becomes very small, so $1/(1 + \text{small_number})$ approaches 1.
        * **If $z$ is a large negative number:** $e^{-z}$ becomes very large, so $1/(1 + \text{large_number})$ approaches 0.
        * **If $z$ is 0:** $e^{-0} = 1$, so $1/(1 + 1) = 0.5$. This is why $z=0$ is the natural decision boundary when the threshold is 0.5.

        This non-linear transformation allows Logistic Regression to model the probability of a binary outcome.
        """)

        # Plotting the sigmoid function
        z_values_bonus = np.linspace(-7, 7, 100)
        sigmoid_values_bonus = 1 / (1 + np.exp(-z_values_bonus))

        fig_sigmoid_bonus = px.line(x=z_values_bonus, y=sigmoid_values_bonus,
                                    title="Sigmoid Function (Logistic Function)",
                                    labels={'x': 'z (Linear Combination of Features)', 'y': 'Probability P(y=1|x)'},
                                    color_discrete_sequence=['#2a9d8f'])
        fig_sigmoid_bonus.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.5)",
                                    annotation_position="top right")
        fig_sigmoid_bonus.add_vline(x=0, line_dash="dash", line_color="purple", annotation_text="z = 0",
                                    annotation_position="bottom right")
        st.plotly_chart(fig_sigmoid_bonus, use_container_width=True)
        st.info(
            "💡 **Visual Insight:** Notice how the sigmoid curve smoothly transitions from 0 to 1, making it perfect for probability prediction.")


def topic_categorical_data_encoding():
    st.header("🏷️ Categorical Data Encoding")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Categorical Data Encoding?")
        st.markdown("""
        **Categorical Data** refers to data that represents categories or labels (e.g., colors, types of fruit, cities, gender). Many machine learning algorithms, especially those based on mathematical equations (like Linear Regression, SVMs, Neural Networks), cannot directly work with text labels. They require numerical input.

        **Categorical Data Encoding** is the process of converting these categorical (text) labels into numerical representations that machine learning algorithms can understand and process.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're building a model to predict house prices, and one of your features is `Neighborhood`: "Downtown", "Suburb", "Rural".

        * A machine learning model can't directly use "Downtown".
        * You need to convert it to numbers, like `1, 2, 3`. But then, `3` (Rural) might seem "greater" than `1` (Downtown), implying an order that doesn't exist.
        * Encoding techniques solve this problem by converting these labels into a suitable numerical format.
        """)

    # 2. Types of Encoding & Equations/Concepts
    with st.expander("➕ Types of Encoding & Concepts", expanded=False):
        st.subheader("Common Encoding Methods")
        st.markdown("""
        There are several methods, each suitable for different scenarios:

        #### 1. Label Encoding
        * **Concept:** Assigns a unique integer to each category.
            * *Example:* `Red -> 0`, `Green -> 1`, `Blue -> 2`
        * **When to use:**
            * For **ordinal categorical data**, where there's a meaningful order (e.g., "Small", "Medium", "Large").
            * For target variables in classification (e.g., `y` in `fit(X, y)`).
            * When the algorithm can handle the implied order (e.g., Decision Trees, which can split on numerical values).
        * **Caution:** Avoid for nominal data (no inherent order) with algorithms sensitive to magnitude (e.g., Linear Regression, KNN), as it might imply a false numerical relationship.

        #### 2. One-Hot Encoding
        * **Concept:** Creates new binary (0 or 1) columns for each category. If an observation belongs to a category, its corresponding new column gets a 1, and others get 0.
            * *Example:* `Color` feature with categories `Red`, `Green`, `Blue` becomes:
                * `Color_Red`, `Color_Green`, `Color_Blue`
                * `Red` -> `[1, 0, 0]`
                * `Green` -> `[0, 1, 0]`
                * `Blue` -> `[0, 0, 1]`
        * **When to use:**
            * For **nominal categorical data** (no inherent order, like "City", "Gender", "Color").
            * With algorithms sensitive to numerical relationships (e.g., Linear Regression, SVMs, Neural Networks).
        * **Caution:** Can lead to a very high number of features if a categorical variable has many unique categories (high cardinality), potentially causing the "curse of dimensionality." It also introduces multicollinearity if all dummy variables are kept (often one is dropped, "dummy variable trap").
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("See Encoding in Action")

        st.markdown("#### Input Categorical Data")
        user_categories_str = st.text_area("Enter categories (comma-separated, e.g., Red, Green, Blue, Red, Yellow)",
                                           "Apple, Orange, Banana, Apple, Orange, Grape", key='encoding_input_data')

        categories_list = [x.strip() for x in user_categories_str.split(',') if x.strip()]
        if not categories_list:
            st.warning("Please enter some categorical values.")
            return

        df_encoding = pd.DataFrame({'Original_Category': categories_list})
        st.write("Original Data:")
        st.dataframe(df_encoding)

        st.subheader("Label Encoding Result")
        le = LabelEncoder()
        try:
            df_encoding['Label_Encoded'] = le.fit_transform(df_encoding['Original_Category'])
            st.dataframe(df_encoding[['Original_Category', 'Label_Encoded']])
            st.info(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        except Exception as e:
            st.error(f"Error during Label Encoding: {e}")

        st.subheader("One-Hot Encoding Result")
        # Handle potential for single category input for OneHotEncoder
        if len(df_encoding['Original_Category'].unique()) > 1:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # sparse_output=False for dense array
            try:
                ohe_array = ohe.fit_transform(df_encoding[['Original_Category']])
                ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(['Original_Category']))
                st.dataframe(ohe_df)
                st.info("Each category becomes a new column with 0s and 1s.")
            except Exception as e:
                st.error(f"Error during One-Hot Encoding: {e}")
        else:
            st.info("One-Hot Encoding requires at least two unique categories to demonstrate effectively.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Categorical Encoding in Python")
        st.code("""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample Data
data = {'Fruit': ['Apple', 'Orange', 'Banana', 'Apple', 'Grape'],
        'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
        'Color': ['Red', 'Orange', 'Yellow', 'Red', 'Green']}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# --- 1. Label Encoding ---
print("\\n--- Label Encoding ---")
le = LabelEncoder()
df['Fruit_LabelEncoded'] = le.fit_transform(df['Fruit'])
print("Fruit Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(df[['Fruit', 'Fruit_LabelEncoded']])

# --- 2. One-Hot Encoding ---
print("\\n--- One-Hot Encoding ---")
# For a single column:
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for dense array
fruit_encoded = ohe.fit_transform(df[['Fruit']])
fruit_encoded_df = pd.DataFrame(fruit_encoded, columns=ohe.get_feature_names_out(['Fruit']))
print("One-Hot Encoded Fruit:")
print(fruit_encoded_df)

# Concatenate with original DataFrame (optional)
df_encoded = pd.concat([df, fruit_encoded_df], axis=1)
print("\\nDataFrame with One-Hot Encoded Fruit:")
print(df_encoded)

# One-Hot Encoding multiple columns (e.g., 'Color')
# pd.get_dummies is a convenient way for multiple columns
df_with_dummies = pd.get_dummies(df, columns=['Color'], prefix='Color', dtype=int)
print("\\nDataFrame with One-Hot Encoded Color (using pd.get_dummies):")
print(df_with_dummies)
        """, language="python")
        st.markdown(
            "This code demonstrates `LabelEncoder` and `OneHotEncoder` from `scikit-learn`, as well as `pandas.get_dummies` for convenience.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Choose the Right Encoding")
        st.markdown("""
        You are preparing data for a **Linear Regression** model. One of your features is `Education Level` with categories: "High School", "Bachelors", "Masters", "PhD".

        Which encoding method would be most appropriate for this feature, and why?
        """)

        user_encoding_choice = st.radio("Choose the best encoding method:",
                                        ("Label Encoding", "One-Hot Encoding"),
                                        key='encoding_task_choice')
        user_reason = st.text_area("Briefly explain your reasoning:", key='encoding_task_reason')

        if st.button("Check My Answer - Encoding"):
            correct_choice = "Label Encoding"

            if user_encoding_choice == correct_choice and (
                    "order" in user_reason.lower() or "ordinal" in user_reason.lower()):
                st.success(
                    f"Correct! **{correct_choice}** is appropriate here because 'Education Level' is **ordinal data** (there's a clear order: High School < Bachelors < Masters < PhD). Label Encoding preserves this order, which can be beneficial for models that understand numerical relationships.")
                st.balloons()
            elif user_encoding_choice == "One-Hot Encoding" and (
                    "no order" in user_reason.lower() or "nominal" in user_reason.lower()):
                st.warning(
                    f"While One-Hot Encoding *would* work, it's not the *most* appropriate. 'Education Level' *does* have an inherent order (it's ordinal data). One-Hot Encoding would create unnecessary columns and lose the ordinal information. The best choice is **Label Encoding**.")
            else:
                st.warning(
                    f"Incorrect or incomplete reasoning. Think about whether the categories have an inherent order. For 'Education Level', there is a clear order. The correct answer is **'{correct_choice}'** because it's **ordinal data**.")

    # 7. Bonus: When to Use Which Encoding
    with st.expander("✨ Bonus: When to Use Which Encoding?", expanded=False):
        st.subheader("Decision Guide for Encoding")
        st.markdown("""
        Choosing between Label Encoding and One-Hot Encoding depends on two main factors:

        1.  **Nature of the Categorical Data:**
            * **Nominal Data:** Categories have no inherent order (e.g., colors, cities, gender).
                * **Best for:** One-Hot Encoding. This avoids implying a false numerical order.
            * **Ordinal Data:** Categories have a clear, meaningful order (e.g., "Small", "Medium", "Large"; "Good", "Better", "Best"; "Low", "Medium", "High").
                * **Best for:** Label Encoding. This preserves the order. If you use One-Hot Encoding, you lose this ordinal information and create more features.

        2.  **Type of Machine Learning Algorithm:**
            * **Tree-based algorithms (Decision Trees, Random Forests, Gradient Boosting):**
                * Can often handle Label Encoded ordinal data well because they make splits based on thresholds.
                * Can also handle One-Hot Encoded data, but might become less efficient with many new columns.
            * **Linear models (Linear Regression, Logistic Regression, SVMs):**
                * **Require One-Hot Encoding for nominal data.** If you use Label Encoding on nominal data, these models will treat the numerical labels as having an order and magnitude, leading to incorrect assumptions.
                * Can use Label Encoding for ordinal data if the order is truly meaningful and linear.
            * **Distance-based algorithms (K-Nearest Neighbors, K-Means, SVMs):**
                * **Require One-Hot Encoding for nominal data.** Label Encoding would distort distances.
                * Can use Label Encoding for ordinal data, but scaling might be needed.

        **General Rule of Thumb:**
        * **Nominal features (no order):** Use One-Hot Encoding.
        * **Ordinal features (has order):** Use Label Encoding.
        * **Target variable (y):** Always use Label Encoding (or similar integer mapping) as models expect numerical classes.
        """)
        st.info(
            "💡 **Key takeaway:** Incorrect encoding can severely impact your model's performance and interpretability. Choose wisely!")


def topic_hierarchical_clustering():
    st.header("🌲 Hierarchical Clustering")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Hierarchical Clustering?")
        st.markdown("""
        **Hierarchical Clustering** is an **unsupervised machine learning algorithm** that builds a hierarchy of clusters. It does not require you to pre-specify the number of clusters (`k`) like K-means. Instead, it creates a tree-like structure called a **dendrogram**, which shows the nested relationships between clusters.

        There are two main types:
        * **Agglomerative (Bottom-Up):** Starts with each data point as its own cluster, and then iteratively merges the closest pairs of clusters until only one large cluster remains (or a stopping criterion is met). This is the most common type.
        * **Divisive (Top-Down):** Starts with all data points in one large cluster and recursively splits them into smaller clusters until each data point is its own cluster. (Less common).
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine organizing your music collection.

        * **Agglomerative:** You start with every single song in its own tiny folder. Then, you find the two most similar songs and put them in a new folder. Then you find the next most similar pair (could be two songs, or a song and an existing folder of songs) and merge them. You keep merging until all songs are in one giant "Music" folder. The dendrogram visually represents this merging history.

        This method allows you to decide on the number of clusters *after* the clustering process is complete, by "cutting" the dendrogram at a certain height.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Concepts: Linkage and Distance", expanded=False):
        st.subheader("Key Concepts: Distance and Linkage Criteria")
        st.markdown("""
        Hierarchical clustering relies on two main concepts:

        #### 1. Distance Metric
        * Measures the similarity (or dissimilarity) between data points. Common choices:
            * **Euclidean Distance:** The straight-line distance between two points in Euclidean space.
                """)
        latex_equation(r'd(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}')
        st.markdown("""
                * **Manhattan Distance:** Sum of absolute differences between coordinates.
                * **Cosine Similarity:** Measures the cosine of the angle between two vectors (useful for text data).

        #### 2. Linkage Criterion
        * Determines how the "distance" between two *clusters* (not just individual points) is calculated. This is crucial for deciding which clusters to merge.
            * **Single Linkage (Min):** Distance between two clusters is the *minimum* distance between any point in the first cluster and any point in the second cluster. (Tends to form long, "straggly" clusters).
            * **Complete Linkage (Max):** Distance between two clusters is the *maximum* distance between any point in the first cluster and any point in the second cluster. (Tends to form compact, spherical clusters).
            * **Average Linkage:** Distance between two clusters is the *average* distance between all pairs of points, where one point is in the first cluster and the other is in the second.
            * **Ward's Method:** Minimizes the total within-cluster variance. It merges clusters that lead to the smallest increase in the total sum of squared errors within clusters. (Often preferred for general-purpose clustering as it tends to produce more balanced clusters).
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Perform Hierarchical Clustering and See the Dendrogram")

        col1_hc, col2_hc = st.columns(2)
        num_samples_hc = col1_hc.slider("Number of data points:", 50, 300, 100, key='hc_samples')
        linkage_method = col2_hc.selectbox("Linkage Method:", ("ward", "complete", "average", "single"),
                                           key='hc_linkage')

        distance_threshold_hc = st.slider("Distance Threshold for Cutting Dendrogram:", 0.0, 20.0, 5.0, 0.5,
                                          key='hc_dist_threshold')

        # Generate synthetic clusterable data
        from sklearn.datasets import make_blobs
        X_hc, y_true_hc = make_blobs(n_samples=num_samples_hc, n_features=2, centers=4, cluster_std=1.0,
                                     random_state=42)

        df_hc = pd.DataFrame(X_hc, columns=['Feature1', 'Feature2'])

        st.subheader("Dendrogram Visualization")
        st.info("The dendrogram shows the merging history of clusters. Cut the tree horizontally to define clusters.")

        # Perform hierarchical clustering for dendrogram
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(X_hc, method=linkage_method)

        fig_dendro = plt.figure(figsize=(12, 6))
        dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True,
                   leaf_rotation=90., leaf_font_size=12., show_contracted=True)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        plt.axhline(y=distance_threshold_hc, color='r', linestyle='--', label=f'Threshold: {distance_threshold_hc}')
        plt.legend()
        st.pyplot(fig_dendro)

        st.info(
            "💡 **Observation:** The height of the merge point indicates the distance between clusters. A horizontal cut at a certain distance defines the clusters.")

        st.subheader("Clustered Data Points")
        # Apply AgglomerativeClustering to get labels based on threshold
        from sklearn.cluster import AgglomerativeClustering
        agg_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold_hc,
                                              linkage=linkage_method)
        cluster_labels_hc = agg_cluster.fit_predict(X_hc)

        df_hc['Cluster'] = cluster_labels_hc
        df_hc['Cluster'] = df_hc['Cluster'].astype(str)  # For categorical coloring

        st.write(f"**Number of Clusters at Threshold {distance_threshold_hc}:** `{len(df_hc['Cluster'].unique())}`")

        fig_hc_scatter = px.scatter(df_hc, x='Feature1', y='Feature2', color='Cluster',
                                    title=f"Hierarchical Clustering Result (K={len(df_hc['Cluster'].unique())})",
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_hc_scatter, use_container_width=True)

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Hierarchical Clustering in Python")
        st.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

# 1. Generate sample data
X, _ = make_blobs(n_samples=100, centers=4, cluster_std=0.8, random_state=42)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# 2. Perform Hierarchical Clustering (Agglomerative)
# linkage(data, method='...') computes the linkage matrix Z
# 'ward': minimizes the variance of the clusters being merged.
# 'complete': uses the maximum distance between observations of the two sets.
# 'average': uses the average of the distances of all observations of the two sets.
# 'single': uses the minimum of the distances between all observations of the two sets.
Z = linkage(X, method='ward') # Using Ward's method as an example

# 3. Visualize the Dendrogram
plt.figure(figsize=(12, 7))
dendrogram(Z,
           truncate_mode='lastp', # Show only the last p merged clusters
           p=10,                  # Show last 10 merges
           show_leaf_counts=True, # Show original number of points in each leaf
           leaf_rotation=90.,     # Rotate leaf labels for readability
           leaf_font_size=12.,    # Font size for leaf labels
           show_contracted=True   # Show contracted nodes
          )
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=10, color='r', linestyle='--', label='Cutoff at Distance 10') # Example cutoff
plt.legend()
plt.show() # In Streamlit: st.pyplot(plt.gcf())

# 4. Extract clusters based on a distance threshold or number of clusters
# Option A: Cut the dendrogram at a specific distance threshold
distance_threshold = 10 # Adjust based on dendrogram
clusters_by_distance = fcluster(Z, distance_threshold, criterion='distance')
print(f"Clusters by distance threshold {distance_threshold}: {np.unique(clusters_by_distance)}")

# Option B: Specify number of clusters directly (similar to K-means)
n_clusters_desired = 4
clusters_by_n = fcluster(Z, n_clusters_desired, criterion='maxclust')
print(f"Clusters for {n_clusters_desired} clusters: {np.unique(clusters_by_n)}")

# 5. Using AgglomerativeClustering from sklearn (simpler API for direct clustering)
agg_model = AgglomerativeClustering(n_clusters=4, linkage='ward') # Specify K directly
# agg_model = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward') # Or use threshold
cluster_labels = agg_model.fit_predict(X)
df['Cluster'] = cluster_labels
print("\\nDataFrame with Cluster Labels:")
print(df.head())

# Visualize the clustered data (similar to K-means scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Cluster', palette='viridis', s=100)
plt.title(f'Agglomerative Clustering (K={len(np.unique(cluster_labels))})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show() # In Streamlit: st.pyplot(plt.gcf())
        """, language="python")
        st.markdown(
            "This code demonstrates how to perform hierarchical clustering using `scipy.cluster.hierarchy` for dendrograms and `sklearn.cluster.AgglomerativeClustering` for direct cluster assignment.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Interpret a Dendrogram Cut")
        st.markdown("""
        Look at the simplified dendrogram below. If you cut the dendrogram at a **distance threshold of 1.5**, how many clusters would you get?
        """)

        # Simplified dendrogram for task
        task_Z = linkage([[0, 0], [0.1, 0.1], [0.5, 0.5], [0.6, 0.6], [2, 2], [2.1, 2.1]],
                         method='single')  # Creates 2 main clusters
        fig_task_dendro = plt.figure(figsize=(8, 5))
        dendrogram(task_Z)
        plt.axhline(y=1.5, color='r', linestyle='--', label='Threshold: 1.5')
        plt.title("Task: Simplified Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        plt.legend()
        st.pyplot(fig_task_dendro)

        user_num_clusters_hc_task = st.number_input("Number of clusters at threshold 1.5:", min_value=1, step=1,
                                                    key='hc_task_clusters')

        if st.button("Check My Answer - Hierarchical Clustering"):
            # Correct calculation for the task_Z and threshold 1.5
            correct_clusters = fcluster(task_Z, 1.5, criterion='distance')
            num_correct_clusters = len(np.unique(correct_clusters))

            if user_num_clusters_hc_task == num_correct_clusters:
                st.success(
                    f"Correct! If you cut the dendrogram at a distance of 1.5, you would get **{num_correct_clusters}** clusters.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. Count how many vertical lines are crossed by the horizontal line at distance 1.5. The correct answer is **{num_correct_clusters}**.")

    # 7. Bonus: Agglomerative vs. Divisive
    with st.expander("✨ Bonus: Agglomerative vs. Divisive Clustering", expanded=False):
        st.subheader("Two Approaches to Building a Hierarchy")
        st.markdown("""
        While both methods build a hierarchy, their approaches are opposite:

        #### **Agglomerative (Bottom-Up):**
        * **Process:** Starts with individual data points as clusters. At each step, it merges the two closest clusters.
        * **Analogy:** Building a family tree by starting with individuals and merging them upwards into couples, then families, then extended families.
        * **Output:** A dendrogram that shows the merges from leaves up to the root.
        * **Pros:** Simpler to implement and visualize (dendrograms are naturally read from bottom-up). More computationally efficient for smaller datasets.
        * **Cons:** Once a merge is made, it cannot be undone.

        #### **Divisive (Top-Down):**
        * **Process:** Starts with all data points in one large cluster. At each step, it splits the "worst" cluster (e.g., the one that is most heterogeneous) into two smaller clusters.
        * **Analogy:** Breaking down a large organization into departments, then teams, then individuals.
        * **Output:** A dendrogram that shows the splits from the root down to the leaves.
        * **Pros:** Can be more efficient for very large datasets where initial merges in agglomerative might be computationally expensive. Can potentially find more natural clusters by focusing on large-scale splits first.
        * **Cons:** More complex to implement. The "optimal" split at each step can be computationally intensive to find.

        **In practice, Agglomerative clustering is far more common** due to its simpler implementation and the intuitive nature of its dendrogram visualization.
        """)
        st.info(
            "💡 **Key takeaway:** Both methods create a hierarchy, but Agglomerative builds up from individual points, while Divisive breaks down from one large cluster.")


def topic_grid_search():
    st.header("🔍 Grid Search")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Grid Search?")
        st.markdown("""
        **Grid Search** is a hyperparameter tuning technique used in machine learning. **Hyperparameters** are parameters that are not learned by the model during training (e.g., `max_depth` in a Decision Tree, `n_neighbors` in KNN, `C` in Logistic Regression). Instead, they are set *before* the training process.

        Grid Search works by exhaustively searching through a manually specified subset of the hyperparameter space of a learning algorithm. It evaluates every possible combination of hyperparameters in a defined "grid" and selects the combination that yields the best performance (e.g., highest accuracy) on a validation set (often using cross-validation).
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're baking a new cake recipe, and you want to find the perfect combination of oven temperature and baking time.

        * **Hyperparameters:** Oven Temperature (e.g., 170°C, 180°C, 190°C), Baking Time (e.g., 25 min, 30 min, 35 min).
        * **Grid Search:** You would try *every single combination*:
            * (170°C, 25 min), (170°C, 30 min), (170°C, 35 min)
            * (180°C, 25 min), (180°C, 30 min), (180°C, 35 min)
            * (190°C, 25 min), (190°C, 30 min), (190°C, 35 min)
        * You'd then taste (evaluate) each cake and pick the combination that resulted in the "best" cake.

        Grid Search systematically explores all options you provide to find the optimal settings for your model.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Concepts: Hyperparameter Space", expanded=False):
        st.subheader("The Concept of Hyperparameter Space")
        st.markdown("""
        While there's no single equation for Grid Search, it's about exploring a defined **hyperparameter space**.

        Imagine you have two hyperparameters:
        * Hyperparameter A: `[value_A1, value_A2, value_A3]`
        * Hyperparameter B: `[value_B1, value_B2]`

        The "grid" of combinations would be:
        `[(A1, B1), (A1, B2), (A2, B1), (A2, B2), (A3, B1), (A3, B2)]`

        For each combination, a model is trained and evaluated (often using cross-validation) to get a robust performance score. The combination with the highest score is chosen as the "best parameters."

        #### **Cross-Validation (often used with Grid Search)**
        * To get a more reliable estimate of a model's performance for a given set of hyperparameters, Grid Search typically uses **cross-validation**. Instead of a single train/test split, the training data is further split into multiple folds. The model is trained on some folds and validated on others, and this process is repeated. The scores are then averaged. This reduces the variance of the performance estimate.
        """)
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/K-fold_cross_validation_EN.svg/1200px-K-fold_cross_validation_EN.svg.png",
            caption="K-Fold Cross-Validation (Source: Wikipedia)",
            use_column_width=True)
        st.info(
            "💡 **Key Idea:** Grid Search tries *every combination* you specify, and cross-validation makes sure the evaluation of each combination is robust.")

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Tune a K-Nearest Neighbors (KNN) Model with Grid Search")
        st.info("Let's use Grid Search to find the best `n_neighbors` for a KNN classifier.")

        col1_gs, col2_gs = st.columns(2)
        num_samples_gs = col1_gs.slider("Number of data points:", 100, 500, 200, key='gs_samples')
        cv_folds_gs = col2_gs.slider("Number of Cross-Validation Folds:", 2, 10, 5, key='gs_cv_folds')

        # Generate synthetic classification data
        from sklearn.datasets import make_classification
        X_gs, y_gs = make_classification(n_samples=num_samples_gs, n_features=2, n_redundant=0,
                                         n_informative=2, n_clusters_per_class=1, random_state=42)

        # Define the parameter grid for KNN
        param_grid_knn = {'n_neighbors': list(range(1, 16))}  # Test n_neighbors from 1 to 15

        st.write("---")
        st.write(f"**Parameters to search:** `n_neighbors` from 1 to 15")
        st.write(f"**Cross-validation folds:** `{cv_folds_gs}`")

        if st.button("Run Grid Search for KNN"):
            with st.spinner("Running Grid Search... This might take a moment."):
                # Initialize KNN model
                knn = KNeighborsClassifier()

                # Initialize GridSearchCV
                grid_search = GridSearchCV(knn, param_grid_knn, cv=cv_folds_gs, scoring='accuracy', n_jobs=-1)

                # Fit Grid Search to the data
                # We use the full dataset X_gs, y_gs here, as GridSearchCV handles its own splits
                grid_search.fit(X_gs, y_gs)

                st.success("Grid Search Completed!")
                st.write(f"**Best Parameters found:** `{grid_search.best_params_}`")
                st.write(f"**Best Cross-Validation Score (Accuracy):** `{grid_search.best_score_:.4f}`")

                st.subheader("Grid Search Results (Accuracy for each n_neighbors)")
                results_df = pd.DataFrame(grid_search.cv_results_)

                # Filter for relevant columns and sort by n_neighbors
                results_df_filtered = results_df[['param_n_neighbors', 'mean_test_score', 'std_test_score']]
                results_df_filtered = results_df_filtered.sort_values(by='param_n_neighbors')
                results_df_filtered.columns = ['n_neighbors', 'Mean Accuracy', 'Std Dev Accuracy']
                st.dataframe(results_df_filtered.set_index('n_neighbors'))

                fig_gs = px.line(results_df_filtered, x='n_neighbors', y='Mean Accuracy',
                                 title="Grid Search Results: Accuracy vs. n_neighbors",
                                 labels={'n_neighbors': 'Number of Neighbors (K)',
                                         'Mean Accuracy': 'Mean Cross-Validation Accuracy'},
                                 markers=True, color_discrete_sequence=['#264653'])
                fig_gs.add_scatter(x=results_df_filtered['n_neighbors'],
                                   y=results_df_filtered['Mean Accuracy'] - results_df_filtered['Std Dev Accuracy'],
                                   mode='lines', line=dict(width=0), showlegend=False)
                fig_gs.add_scatter(x=results_df_filtered['n_neighbors'],
                                   y=results_df_filtered['Mean Accuracy'] + results_df_filtered['Std Dev Accuracy'],
                                   mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                                   showlegend=False, name='Std Dev Range')

                st.plotly_chart(fig_gs, use_container_width=True)
                st.info(
                    "💡 **Observation:** The plot shows how accuracy changes with different `n_neighbors` values. The shaded area represents the standard deviation of accuracy across folds.")

        else:
            st.info("Click 'Run Grid Search for KNN' to start the hyperparameter tuning process.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Grid Search in Python (scikit-learn)")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # Example for another model

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and (hold-out) test set
# GridSearchCV will handle further splits for cross-validation on the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define the model (estimator)
# Let's use K-Nearest Neighbors as an example
knn = KNeighborsClassifier()

# 3. Define the parameter grid (hyperparameter space to search)
# This is a dictionary where keys are hyperparameter names and values are lists of values to try.
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11], # Values for K in KNN
    'weights': ['uniform', 'distance'], # How to weight neighbors
    'metric': ['euclidean', 'manhattan'] # Distance metric
}
print("Parameter Grid to search:")
print(param_grid)

# 4. Initialize GridSearchCV
# estimator: The model object (e.g., knn)
# param_grid: The dictionary of hyperparameters to search
# cv: Number of cross-validation folds
# scoring: Metric to optimize (e.g., 'accuracy', 'f1_weighted', 'roc_auc')
# n_jobs: Number of CPU cores to use (-1 means use all available)
grid_search = GridSearchCV(estimator=knn,
                           param_grid=param_grid,
                           cv=5, # 5-fold cross-validation
                           scoring='accuracy',
                           n_jobs=-1, # Use all available CPU cores
                           verbose=1) # Print progress

# 5. Fit Grid Search to the training data
# This will train and evaluate a model for every combination in param_grid using cross-validation
print("\\nStarting Grid Search...")
grid_search.fit(X_train, y_train)
print("Grid Search Complete!")

# 6. Get the best parameters and best score
print(f"\\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score (accuracy): {grid_search.best_score_:.4f}")

# 7. Access all results
results_df = pd.DataFrame(grid_search.cv_results_)
print("\\nPartial Grid Search Results (first 5 rows):")
print(results_df[['param_n_neighbors', 'param_weights', 'param_metric', 'mean_test_score', 'rank_test_score']].head())

# 8. Use the best estimator to make predictions on the unseen test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"\\nAccuracy on unseen test set with best model: {test_accuracy:.4f}")

# Example with Decision Tree
print("\\n--- Example with Decision Tree ---")
dt = DecisionTreeClassifier(random_state=42)
dt_param_grid = {
    'max_depth': [3, 5, 7, None], # None means no limit
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
dt_grid_search = GridSearchCV(dt, dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid_search.fit(X_train, y_train)
print(f"Best DT parameters: {dt_grid_search.best_params_}")
print(f"Best DT score: {dt_grid_search.best_score_:.4f}")
        """, language="python")
        st.markdown(
            "This code demonstrates how to set up and run `GridSearchCV` to find optimal hyperparameters for a machine learning model.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Identify the Best Hyperparameter")
        st.markdown("""
        You ran a Grid Search for a model with a single hyperparameter `alpha`, testing values `[0.1, 1.0, 10.0]`.

        Here are the cross-validation accuracies obtained for each `alpha`:
        * `alpha = 0.1`: Accuracy = 0.78
        * `alpha = 1.0`: Accuracy = 0.85
        * `alpha = 10.0`: Accuracy = 0.82

        Which `alpha` value is the best hyperparameter based on these results?
        """)

        user_best_alpha = st.number_input("Enter the best alpha value:", format="%.1f", key='gs_task_alpha')

        if st.button("Check My Answer - Grid Search"):
            correct_alpha = 1.0
            if abs(user_best_alpha - correct_alpha) < 0.001:
                st.success(
                    f"Correct! The best `alpha` value is **{correct_alpha}** because it yielded the highest accuracy of 0.85.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. The best hyperparameter is the one that gives the highest performance score. The correct answer is **{correct_alpha}**.")

    # 7. Bonus: Alternatives to Grid Search
    with st.expander("✨ Bonus: Beyond Grid Search", expanded=False):
        st.subheader("More Efficient Hyperparameter Tuning Methods")
        st.markdown("""
        While Grid Search is straightforward, it can become computationally expensive very quickly, especially when dealing with many hyperparameters or a large range of values. This is because it tries *every single combination*.

        Fortunately, there are more efficient alternatives:

        1.  **Random Search (`RandomizedSearchCV`):**
            * **Concept:** Instead of trying every combination, it samples a fixed number of random combinations from the specified hyperparameter distributions.
            * **Pros:** Often finds a good set of hyperparameters much faster than Grid Search, especially when some hyperparameters have a much larger impact than others. More efficient for higher-dimensional search spaces.
            * **Cons:** Not guaranteed to find the absolute best combination, but often finds a "good enough" one.

        2.  **Bayesian Optimization:**
            * **Concept:** Builds a probabilistic model of the objective function (e.g., accuracy vs. hyperparameters) and uses this model to intelligently choose the next set of hyperparameters to evaluate. It tries to balance exploration (trying new areas) and exploitation (refining promising areas).
            * **Pros:** Much more efficient than Grid Search or Random Search for complex, expensive-to-evaluate objective functions.
            * **Cons:** More complex to implement, requires specialized libraries (e.g., `hyperopt`, `scikit-optimize`).

        3.  **Genetic Algorithms:**
            * **Concept:** Inspired by natural selection. It evolves a population of hyperparameter combinations over generations, selecting the "fittest" ones to "reproduce" and create new combinations.
            * **Pros:** Can explore complex, non-linear search spaces effectively.
            * **Cons:** Can be slow to converge for some problems.

        **When to use which:**
        * **Grid Search:** Small number of hyperparameters, small range of values, when you need to be sure you've explored every specific combination.
        * **Random Search:** Larger number of hyperparameters, when you want to find a good solution faster than Grid Search.
        * **Bayesian Optimization / Genetic Algorithms:** Very complex models, large hyperparameter spaces, when training is very time-consuming.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*C4FjVf1Hl-yQ6JvC6oN52g.png",
                 # Reusing for conceptual comparison
                 caption="Conceptual comparison of Grid Search vs. Random Search (Source: Medium - adapted)",
                 use_column_width=True)
        st.info(
            "💡 **Key takeaway:** Grid Search is a good starting point, but for larger problems, more advanced methods offer significant efficiency gains.")


def topic_cross_validation():
    st.header("🔄 Cross Validation")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Cross Validation?")
        st.markdown("""
        **Cross-validation** is a robust technique for evaluating the performance of a machine learning model. Instead of a single train/test split, it involves partitioning the original dataset into multiple subsets (or "folds"). The model is then trained and evaluated multiple times, each time using a different fold as the test set and the remaining folds as the training set.

        This method helps to:
        * Get a more reliable estimate of the model's performance on unseen data.
        * Reduce the variance of the performance estimate compared to a single train/test split.
        * Make better use of the available data for both training and testing.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you're a chef testing a new recipe.

        * **Single Train/Test Split:** You make one batch, taste half (training), and then taste the other half (testing). If that one "test" half was unusually good or bad, your evaluation might be biased.
        * **Cross-Validation:** You make 5 batches (5 "folds"). You taste batch 1 while cooking with batches 2-5. Then you taste batch 2 while cooking with 1, 3-5, and so on. By tasting all batches as "test" at some point, you get a much more reliable opinion of the recipe's overall quality.

        It's like getting multiple opinions on your model's performance, making the evaluation more robust.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Concepts of K-Fold Cross Validation", expanded=False):
        st.subheader("K-Fold Cross Validation")
        st.markdown("""
        The most common type of cross-validation is **K-Fold Cross Validation**.

        1.  **Divide Data:** The dataset is divided into $K$ equally sized subsets (folds).
        2.  **Iterate:** The process is repeated $K$ times (or "folds"). In each iteration:
            * One fold is used as the **test set**.
            * The remaining $K-1$ folds are combined to form the **training set**.
        3.  **Evaluate:** The model is trained on the training set and evaluated on the test set. The performance metric (e.g., accuracy, R-squared) is recorded for that fold.
        4.  **Average:** After $K$ iterations, the $K$ performance scores are averaged to get a single, more robust estimate of the model's performance.
        """)
        latex_equation(r'\text{Average Performance} = \frac{1}{K} \sum_{i=1}^{K} \text{Performance}_i')
        st.markdown("""
        * $K$: The number of folds (e.g., 5, 10).
        * $\text{Performance}_i$: The metric (e.g., accuracy) obtained in the $i^{th}$ fold.

        **Common values for K:** 5 or 10 are frequently used.
        * **Stratified K-Fold:** For classification problems, `StratifiedKFold` is preferred. It ensures that each fold has approximately the same percentage of samples of each target class as the complete set, preserving class distribution.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Visualize K-Fold Cross Validation")

        col1_cv, col2_cv = st.columns(2)
        num_samples_cv = col1_cv.slider("Number of data points:", 50, 500, 100, key='cv_samples')
        n_splits_cv = col2_cv.slider("Number of Folds (K):", 2, 10, 5, key='cv_k_folds')

        # Generate synthetic classification data
        X_cv, y_cv = make_classification(n_samples=num_samples_cv, n_features=2, n_redundant=0,
                                         n_clusters_per_class=1, random_state=42, n_classes=2)
        df_cv = pd.DataFrame(X_cv, columns=['Feature1', 'Feature2'])
        df_cv['Target'] = y_cv

        st.markdown(f"**Dataset Size:** {num_samples_cv} samples")
        st.markdown(f"**Number of Folds (K):** {n_splits_cv}")

        # Use KFold for visualization
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=42)

        fold_data = []
        fold_num = 0
        for train_index, test_index in kf.split(X_cv):
            fold_num += 1
            train_df = df_cv.iloc[train_index].copy()
            test_df = df_cv.iloc[test_index].copy()

            train_df['Set'] = f'Fold {fold_num} - Train'
            test_df['Set'] = f'Fold {fold_num} - Test'

            fold_data.append(train_df)
            fold_data.append(test_df)

        combined_fold_df = pd.concat(fold_data)

        st.subheader("K-Fold Split Visualization")
        st.markdown(
            "Each row represents a fold. The colored blocks show which data points are used for training and testing in that fold.")

        # Create a simplified visualization of the folds
        fold_vis_data = []
        for i, (train_index, test_index) in enumerate(kf.split(X_cv)):
            row_data = np.zeros(num_samples_cv, dtype=int)
            row_data[train_index] = 1  # Train
            row_data[test_index] = 2  # Test
            fold_vis_data.append(row_data)

        fig_vis_cv = go.Figure()
        for i, row in enumerate(fold_vis_data):
            for j, val in enumerate(row):
                color = '#2a9d8f' if val == 1 else '#e76f51' if val == 2 else 'lightgray'
                text = 'Train' if val == 1 else 'Test' if val == 2 else ''
                fig_vis_cv.add_shape(
                    type="rect",
                    x0=j, y0=i, x1=j + 1, y1=i + 1,
                    fillcolor=color, line_width=0
                )
                # Add text for each block
                # fig_vis_cv.add_annotation(
                #     x=j + 0.5, y=i + 0.5, text=text,
                #     showarrow=False, font=dict(size=8, color='white')
                # )
            fig_vis_cv.add_annotation(
                x=-0.5, y=i + 0.5, text=f"Fold {i + 1}",
                showarrow=False, xanchor='right', font=dict(size=10, color='black')
            )

        fig_vis_cv.update_layout(
            title=f'{n_splits_cv}-Fold Cross-Validation Data Split',
            xaxis=dict(title='Data Sample Index', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='Fold', showgrid=False, zeroline=False, showticklabels=False),
            height=200 + n_splits_cv * 20,  # Adjust height based on number of folds
            showlegend=False,
            margin=dict(l=80, r=20, t=50, b=20)
        )
        # Add a custom legend
        fig_vis_cv.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#2a9d8f'), name='Train Set'))
        fig_vis_cv.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#e76f51'), name='Test Set'))

        st.plotly_chart(fig_vis_cv, use_container_width=True)
        st.info(
            "💡 **Observation:** In each row (fold), a different portion of the data (red) is held out for testing, while the rest (green) is used for training.")

        st.subheader("Demonstration: Model Evaluation with Cross-Validation")
        st.markdown("Let's train a simple Logistic Regression model and evaluate it using cross-validation.")

        # Train and evaluate a Logistic Regression model using cross_val_score
        model_cv = LogisticRegression(random_state=42, solver='liblinear')

        # Scale data for Logistic Regression
        scaler_cv = StandardScaler()
        X_scaled_cv = scaler_cv.fit_transform(X_cv)

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model_cv, X_scaled_cv, y_cv, cv=n_splits_cv, scoring='accuracy')

        st.write(f"**Accuracy Scores for each of the {n_splits_cv} folds:**")
        for i, score in enumerate(scores):
            st.write(f"Fold {i + 1}: `{score:.4f}`")

        st.write(f"**Mean Accuracy across all folds:** `{np.mean(scores):.4f}`")
        st.write(f"**Standard Deviation of Accuracy across all folds:** `{np.std(scores):.4f}`")
        st.info(
            "💡 The mean accuracy is your robust estimate of model performance. The standard deviation indicates how much the performance varies across different data splits.")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Cross Validation in Python")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification # For sample data

# 1. Generate sample data
X, y = make_classification(n_samples=300, n_features=5, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 2. Scale the data (important for many models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Define the model
model = LogisticRegression(random_state=42, solver='liblinear')

# --- K-Fold Cross Validation ---
print("--- K-Fold Cross Validation (for Regression or Classification) ---")
n_splits_kfold = 5 # Number of folds
kf = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=42)

# Evaluate model using cross_val_score
# 'scoring' parameter specifies the evaluation metric (e.g., 'accuracy', 'r2', 'neg_mean_squared_error')
scores_kfold = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')

print(f"Individual fold accuracies: {scores_kfold}")
print(f"Mean accuracy: {np.mean(scores_kfold):.4f}")
print(f"Standard deviation of accuracy: {np.std(scores_kfold):.4f}")

# --- Stratified K-Fold Cross Validation (Recommended for Classification) ---
print("\\n--- Stratified K-Fold Cross Validation (for Classification) ---")
n_splits_stratified = 5
skf = StratifiedKFold(n_splits=n_splits_stratified, shuffle=True, random_state=42)

scores_stratified = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')

print(f"Individual fold accuracies (stratified): {scores_stratified}")
print(f"Mean accuracy (stratified): {np.mean(scores_stratified):.4f}")
print(f"Standard deviation of accuracy (stratified): {np.std(scores_stratified):.4f}")

# You can use other metrics too:
# from sklearn.metrics import make_scorer, precision_score
# precision_scorer = make_scorer(precision_score, average='weighted')
# precision_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring=precision_scorer)
# print(f"Mean Precision (stratified): {np.mean(precision_scores):.4f}")
        """, language="python")
        st.markdown(
            "This code demonstrates how to use `KFold` and `StratifiedKFold` with `cross_val_score` to get a robust evaluation of your model.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Advantages of Cross-Validation")
        st.markdown("""
        You've built a model and evaluated it using a single 80/20 train/test split, getting an accuracy of 92%.

        What is the primary reason you might still want to use **K-Fold Cross Validation** instead of relying solely on this single split?
        """)

        user_cv_advantage = st.radio("Primary advantage of K-Fold Cross Validation:",
                                     ("It always gives higher accuracy.",
                                      "It reduces training time.",
                                      "It provides a more reliable estimate of model performance on unseen data.",
                                      "It makes the model simpler."),
                                     key='cv_task_advantage')

        if st.button("Check My Answer - Cross Validation"):
            correct_answer = "It provides a more reliable estimate of model performance on unseen data."
            if user_cv_advantage == correct_answer:
                st.success(
                    f"Correct! **{correct_answer}** A single split can be lucky or unlucky, leading to a biased performance estimate. Cross-validation averages results over multiple splits.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. While accuracy might vary, the main benefit is the reliability of the estimate. The correct answer is: **{correct_answer}**.")

    # 7. Bonus: Why it's Better than Single Split
    with st.expander("✨ Bonus: Why Cross-Validation is Superior", expanded=False):
        st.subheader("Addressing the Limitations of a Single Train/Test Split")
        st.markdown("""
        A single train/test split is simple, but it has two main drawbacks that cross-validation addresses:

        1.  **High Variance in Performance Estimate:**
            * The performance estimate (e.g., accuracy) can be highly dependent on the particular random split of the data. If you get a "lucky" split where the test set is easy, your model might appear better than it truly is. If you get an "unlucky" split, it might appear worse.
            * Cross-validation averages the performance over multiple splits, providing a more stable and less biased estimate.

        2.  **Less Data for Training (especially with small datasets):**
            * With a single split (e.g., 80/20), 20% of your data is held out purely for testing and is never used for training the final model.
            * In K-Fold cross-validation, every data point gets to be in the test set exactly once, and in the training set $K-1$ times. This makes more efficient use of your data, which is particularly beneficial for smaller datasets.

        **Visualizing the problem with a single split:**
        Imagine you have 10 data points. A single 70/30 split might always pick the same 3 "easy" points for testing, making your model look great. Cross-validation ensures that different combinations of 3 points are used for testing across different folds.
        """)
        st.info(
            "💡 **Key takeaway:** Cross-validation provides a more robust and less biased evaluation of your model's generalization ability.")


def topic_auc_roc_curve():
    st.header("📈 AUC - ROC Curve")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What are ROC Curve and AUC?")
        st.markdown("""
        The **Receiver Operating Characteristic (ROC) curve** is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It plots two parameters:

        * **True Positive Rate (TPR)** on the Y-axis.
        * **False Positive Rate (FPR)** on the X-axis.

        The **Area Under the ROC Curve (AUC)** is a single scalar value that summarizes the overall performance of a binary classifier across all possible classification thresholds.
        * **AUC = 1:** Perfect classifier.
        * **AUC = 0.5:** Classifier performs no better than random guessing (diagonal line).
        * **AUC < 0.5:** Classifier performs worse than random (rare, implies you can invert its predictions).

        It's particularly useful for evaluating models on **imbalanced datasets**, where accuracy alone can be misleading.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine a medical test designed to detect a rare disease.

        * **TPR (Sensitivity/Recall):** The proportion of actual sick people who correctly test positive. (We want this high).
        * **FPR (False Positive Rate):** The proportion of actual healthy people who incorrectly test positive. (We want this low).

        The ROC curve shows how these two rates change as you adjust the "threshold" for a positive diagnosis. If you make the test very sensitive (high TPR), you might also get more false alarms (high FPR). The AUC tells you how good the test is at distinguishing sick from healthy people overall, regardless of where you set that threshold.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Equations: TPR, FPR, and Interpretation", expanded=False):
        st.subheader("Formulas for TPR and FPR")
        st.markdown("""
        To understand TPR and FPR, we first need the confusion matrix terms:
        * **True Positive (TP):** Actual Positive, Predicted Positive
        * **True Negative (TN):** Actual Negative, Predicted Negative
        * **False Positive (FP):** Actual Negative, Predicted Positive (Type I error)
        * **False Negative (FN):** Actual Positive, Predicted Negative (Type II error)
        """)
        latex_equation(
            r'\text{TPR (True Positive Rate) / Sensitivity / Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}')
        st.markdown("""
        * **TPR:** The proportion of all actual positive cases that were correctly identified as positive.

        """)
        latex_equation(r'\text{FPR (False Positive Rate)} = \frac{\text{FP}}{\text{FP} + \text{TN}}')
        st.markdown("""
        * **FPR:** The proportion of all actual negative cases that were incorrectly identified as positive.

        The ROC curve is created by calculating TPR and FPR at various threshold settings (from 0 to 1) for the predicted probabilities of the positive class.

        **AUC Interpretation:**
        The AUC can be interpreted as the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("Explore ROC Curve and AUC")

        col1_auc, col2_auc = st.columns(2)
        num_samples_auc = col1_auc.slider("Number of data points:", 100, 1000, 300, key='auc_samples')
        class_sep_auc = col2_auc.slider("Class Separability (higher = easier):", 0.5, 3.0, 1.5, 0.1, key='auc_sep')

        st.info(
            "💡 Adjust 'Class Separability'. Higher values make the classes easier to distinguish, leading to a better AUC.")

        # Generate synthetic binary classification data
        from sklearn.datasets import make_blobs
        X_auc, y_auc = make_blobs(n_samples=num_samples_auc, centers=2, cluster_std=class_sep_auc, random_state=42)

        # Train a Logistic Regression model
        model_auc = LogisticRegression(solver='liblinear', random_state=42)
        X_train_auc, X_test_auc, y_train_auc, y_test_auc = train_test_split(X_auc, y_auc, test_size=0.3,
                                                                            random_state=42, stratify=y_auc)

        # Scale data
        scaler_auc = StandardScaler()
        X_train_auc_scaled = scaler_auc.fit_transform(X_train_auc)
        X_test_auc_scaled = scaler_auc.transform(X_test_auc)

        model_auc.fit(X_train_auc_scaled, y_train_auc)

        # Get predicted probabilities for the positive class (class 1)
        y_pred_proba_auc = model_auc.predict_proba(X_test_auc_scaled)[:, 1]

        # Calculate FPR, TPR, and thresholds
        fpr, tpr, thresholds = roc_curve(y_test_auc, y_pred_proba_auc)
        roc_auc = auc(fpr, tpr)

        st.subheader("ROC Curve")
        fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.2f})',
                          labels=dict(x='False Positive Rate (FPR)', y='True Positive Rate (TPR)'),
                          width=700, height=500)
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, y0=0, x1=1,
                          y1=1)  # Diagonal line (random classifier)
        fig_roc.update_layout(hovermode="x unified")  # Show info on hover
        st.plotly_chart(fig_roc, use_container_width=True)
        st.write(f"**Area Under the Curve (AUC):** `{roc_auc:.4f}`")
        st.info(
            "💡 **Observation:** A good classifier's ROC curve bows towards the top-left corner, indicating high TPR and low FPR. The dashed line represents a random classifier (AUC=0.5).")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Generating ROC Curve and AUC in Python")
        st.code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification # For sample data

# 1. Generate synthetic binary classification data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train a classifier (e.g., Logistic Regression)
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Get predicted probabilities for the positive class (class 1)
# roc_curve expects probabilities, not just predicted classes
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# 6. Calculate FPR, TPR, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# 7. Calculate AUC
roc_auc = auc(fpr, tpr)
# Alternatively, use roc_auc_score directly:
# roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Area Under the ROC Curve (AUC): {roc_auc:.4f}")

# 8. Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show() # In Streamlit, use st.pyplot(plt.gcf())
        """, language="python")
        st.markdown(
            "This code shows how to calculate and plot the ROC curve and AUC score for a binary classification model.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Interpret AUC Value")
        st.markdown("""
        You are evaluating two machine learning models for disease detection:
        * **Model A:** AUC = 0.91
        * **Model B:** AUC = 0.75

        Which model is generally better at distinguishing between patients with and without the disease?
        """)

        user_auc_choice = st.radio("Which model is better?", ("Model A", "Model B", "They are equally good"),
                                   key='auc_task_choice')

        if st.button("Check My Answer - AUC"):
            correct_model = "Model A"
            if user_auc_choice == correct_model:
                st.success(
                    f"Correct! **{correct_model}** is generally better. A higher AUC value indicates a better ability to discriminate between positive and negative classes across various thresholds.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. Remember that a higher AUC means better overall performance in distinguishing classes. The correct answer is **{correct_model}**.")

    # 7. Bonus: Thresholds and Trade-offs
    with st.expander("✨ Bonus: Thresholds and the Trade-off", expanded=False):
        st.subheader("The Role of Thresholds in Classification")
        st.markdown("""
        Most classification models (like Logistic Regression) output a probability (e.g., 0.7 for class 1). A **threshold** is then applied to these probabilities to make a final class prediction.

        * **Default Threshold:** Usually 0.5. If probability > 0.5, predict class 1; otherwise, predict class 0.

        #### **How Threshold Affects TPR and FPR:**
        * **Lowering the threshold (e.g., to 0.3):**
            * You become more lenient in predicting the positive class.
            * **Increases TPR:** More actual positives are caught (good!).
            * **Increases FPR:** More actual negatives are also incorrectly classified as positive (bad!).
            * Moving *left* on the ROC curve.
        * **Raising the threshold (e.g., to 0.7):**
            * You become more strict in predicting the positive class.
            * **Decreases TPR:** Fewer actual positives are caught (bad!).
            * **Decreases FPR:** Fewer actual negatives are incorrectly classified as positive (good!).
            * Moving *right* on the ROC curve.

        The ROC curve shows this trade-off: as you increase TPR, FPR also tends to increase. The ideal point on the curve depends on the specific problem's costs of false positives vs. false negatives.

        **Example:**
        * **Disease Detection:** You might prefer a very high TPR (don't miss sick people) even if it means a slightly higher FPR (some healthy people get false alarms). This would mean choosing a lower threshold.
        * **Spam Detection:** You might prefer a very low FPR (don't mark legitimate emails as spam) even if it means a slightly lower TPR (some spam gets through). This would mean choosing a higher threshold.
        """)
        st.info(
            "💡 **Key takeaway:** The ROC curve helps you choose an optimal threshold based on your specific problem's needs and the costs associated with different types of errors.")


def topic_bootstrap_aggregation():
    st.header("🌳 Bootstrap Aggregation (Bagging)")
    st.markdown("---")

    # 1. Definition Section
    with st.expander("📝 Definition", expanded=True):
        st.subheader("What is Bootstrap Aggregation (Bagging)?")
        st.markdown("""
        **Bootstrap Aggregation (Bagging)** is an **ensemble learning method** that aims to improve the stability and accuracy of machine learning algorithms, primarily by reducing variance and helping to avoid overfitting.

        The core idea is to train **multiple versions of the same base learning algorithm** on different **bootstrap samples** of the training data, and then combine their predictions.

        * **Bootstrap Sample:** A random sample of the original dataset, taken with replacement. This means some data points might appear multiple times in a single bootstrap sample, while others might not appear at all. Each bootstrap sample has the same size as the original dataset.
        * **Aggregation:**
            * For **classification:** The predictions of all individual models are combined by **majority voting**.
            * For **regression:** The predictions are averaged.

        Bagging works because the base models trained on different bootstrap samples are slightly different from each other (they have high variance). By averaging or voting their predictions, the errors tend to cancel out, leading to a more robust and generalized overall prediction.
        """)
        st.markdown("""
        **Daily-life Example:**
        Imagine you need to estimate the weight of a large crowd of people, but you can only weigh small groups.

        * **Without Bagging:** You pick one small group, weigh them, and use that average as your estimate. This estimate might be very inaccurate if your single group was unusually heavy or light.
        * **With Bagging:** You randomly pick *many* different small groups (with some people potentially picked multiple times, others not at all). You get an average weight from each group. Then, you average all these averages to get your final estimate.

        By combining many individual, slightly varied estimates, you get a much more reliable and stable overall estimate.
        """)

    # 2. Equation/Math Section
    with st.expander("➕ Concepts of Bootstrapping and Aggregation", expanded=False):
        st.subheader("The Bootstrap Sampling Process")
        st.markdown("""
        For a dataset with $N$ samples:

        1.  **Bootstrap Sample Generation:**
            * Randomly draw $N$ samples from the original dataset **with replacement**.
            * Repeat this process $M$ times to create $M$ different bootstrap samples ($D_1, D_2, \dots, D_M$).
            * Each $D_i$ will be of size $N$.

        2.  **Model Training:**
            * Train a base learner (e.g., a Decision Tree) on each bootstrap sample $D_i$. This results in $M$ different models ($h_1, h_2, \dots, h_M$).

        3.  **Aggregation (Prediction):**
            * For a new input $x_{new}$:
                * **Classification:** The final prediction is the class that receives the majority vote from $h_1(x_{new}), h_2(x_{new}), \dots, h_M(x_{new})$.
                * **Regression:** The final prediction is the average of $h_1(x_{new}), h_2(x_{new}), \dots, h_M(x_{new})$.
        """)
        latex_equation(r'\text{Classification (Voting): } H(x) = \text{mode}\{h_1(x), h_2(x), \dots, h_M(x)\}')
        latex_equation(r'\text{Regression (Averaging): } H(x) = \frac{1}{M} \sum_{m=1}^{M} h_m(x)')
        st.markdown("""
        * $H(x)$: The final aggregated prediction for input $x$.
        * $h_m(x)$: The prediction of the $m^{th}$ base model for input $x$.
        * $M$: The number of base models (estimators) in the ensemble.
        """)

    # 3. Interactive Inputs & 4. Visualization
    with st.expander("💡 Interactive Experiment", expanded=True):
        st.subheader("See Bagging in Action")

        col1_bag, col2_bag = st.columns(2)
        num_samples_bag = col1_bag.slider("Number of original data points:", 50, 500, 100, key='bag_samples')
        n_estimators_bag = col2_bag.slider("Number of Base Estimators (M):", 5, 50, 10, key='bag_estimators')

        st.info(
            "💡 We'll use a simple Decision Tree as the base estimator. Bagging helps reduce its tendency to overfit.")

        # Generate synthetic classification data
        X_bag, y_bag = make_classification(n_samples=num_samples_bag, n_features=2, n_redundant=0,
                                           n_clusters_per_class=1, random_state=42, n_classes=2)
        X_train_bag, X_test_bag, y_train_bag, y_test_bag = train_test_split(X_bag, y_bag, test_size=0.3,
                                                                            random_state=42, stratify=y_bag)

        # Scale data
        scaler_bag = StandardScaler()
        X_train_bag_scaled = scaler_bag.fit_transform(X_train_bag)
        X_test_bag_scaled = scaler_bag.transform(X_test_bag)

        # Train a single Decision Tree
        single_tree = DecisionTreeClassifier(random_state=42, max_depth=5)
        single_tree.fit(X_train_bag_scaled, y_train_bag)
        single_tree_accuracy = accuracy_score(y_test_bag, single_tree.predict(X_test_bag_scaled))

        # Train a Bagging Classifier
        from sklearn.ensemble import BaggingClassifier
        bagging_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42, max_depth=5),  # Base estimator
            n_estimators=n_estimators_bag,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        bagging_model.fit(X_train_bag_scaled, y_train_bag)
        bagging_accuracy = accuracy_score(y_test_bag, bagging_model.predict(X_test_bag_scaled))

        st.subheader("Performance Comparison: Single Tree vs. Bagging")
        st.write(f"**Accuracy of a Single Decision Tree:** `{single_tree_accuracy:.4f}`")
        st.write(f"**Accuracy of Bagging Classifier (M={n_estimators_bag} trees):** `{bagging_accuracy:.4f}`")

        if bagging_accuracy > single_tree_accuracy:
            st.success("🎉 Bagging improved accuracy! This often happens by reducing variance.")
        elif bagging_accuracy < single_tree_accuracy:
            st.warning(
                "Bagging did not improve accuracy in this instance. This can happen, especially with very simple datasets or if the base estimator is already very stable.")
        else:
            st.info(
                "Bagging accuracy is similar to the single tree. Try adjusting the number of estimators or data points.")

        st.subheader("Conceptual Visualization: Bootstrap Samples")
        st.markdown("Here's how bootstrap samples are created from the original training data.")

        # Display a few conceptual bootstrap samples
        num_bootstrap_samples_to_show = 3
        bootstrap_dfs = []
        original_indices = np.arange(len(X_train_bag))

        for i in range(num_bootstrap_samples_to_show):
            # Sample with replacement
            bootstrap_indices = np.random.choice(original_indices, size=len(original_indices), replace=True)
            bootstrap_sample_df = pd.DataFrame({
                'Original Index': original_indices,
                'Included in Sample': ['Yes' if idx in bootstrap_indices else 'No' for idx in original_indices],
                'Count in Sample': [list(bootstrap_indices).count(idx) for idx in original_indices]
            })
            bootstrap_dfs.append(bootstrap_sample_df)
            st.write(f"**Bootstrap Sample {i + 1} (showing inclusion of original indices):**")
            st.dataframe(bootstrap_sample_df.head(10))  # Show first 10 for brevity
            st.markdown("---")
        st.info(
            "💡 **Observation:** Notice how some original indices appear multiple times ('Count in Sample' > 1) and some don't appear at all ('No' in 'Included in Sample').")

    # 5. Python Code
    with st.expander("🐍 Python Code Snippets", expanded=False):
        st.subheader("Implementing Bagging in Python")
        st.code("""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification # For sample data

# 1. Generate synthetic data
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train a single base estimator (e.g., Decision Tree) for comparison ---
single_tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
single_tree_model.fit(X_train_scaled, y_train)
single_tree_pred = single_tree_model.predict(X_test_scaled)
single_tree_accuracy = accuracy_score(y_test, single_tree_pred)
print(f"Accuracy of a single Decision Tree: {single_tree_accuracy:.4f}")

# --- Implement Bagging ---
# 4. Define the base estimator (the type of model to be bagged)
base_estimator = DecisionTreeClassifier(random_state=42, max_depth=5)

# 5. Create a BaggingClassifier instance
# n_estimators: The number of base estimators (trees) in the ensemble
# max_samples: The number of samples to draw from X to train each base estimator
#              (default is 1.0 * n_samples, meaning same size as original training set)
# bootstrap: Whether samples are drawn with replacement (True for bagging)
# n_jobs: Number of jobs to run in parallel (-1 means use all processors)
bagging_model = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=50, # Train 50 decision trees
    max_samples=1.0, # Each tree gets a sample of size equal to training set
    bootstrap=True,  # Samples are drawn with replacement
    random_state=42,
    n_jobs=-1
)

# 6. Train the Bagging model
bagging_model.fit(X_train_scaled, y_train)

# 7. Make predictions
bagging_pred = bagging_model.predict(X_test_scaled)

# 8. Evaluate the Bagging model
bagging_accuracy = accuracy_score(y_test, bagging_pred)
print(f"Accuracy of Bagging Classifier: {bagging_accuracy:.4f}")

# You'll often find that bagging_accuracy is higher or more stable than single_tree_accuracy
        """, language="python")
        st.markdown(
            "This code demonstrates how to use `BaggingClassifier` from `sklearn.ensemble` to create an ensemble of Decision Trees and compare its performance to a single tree.")

    # 6. Practical Task
    with st.expander("🎯 Practical Task", expanded=False):
        st.subheader("Your Task: Identify a Characteristic of Bootstrapping")
        st.markdown("""
        When creating a bootstrap sample from an original dataset of size N, which of the following is true?
        """)

        user_bag_char = st.radio("Select the correct statement about bootstrap sampling:",
                                 ("Each sample is taken without replacement.",
                                  "Each bootstrap sample is always smaller than the original dataset.",
                                  "Some data points from the original dataset may appear multiple times in a single bootstrap sample.",
                                  "Each bootstrap sample contains completely new, unseen data points."),
                                 key='bag_task_char')

        if st.button("Check My Answer - Bagging"):
            correct_statement = "Some data points from the original dataset may appear multiple times in a single bootstrap sample."
            if user_bag_char == correct_statement:
                st.success(
                    f"Correct! **{correct_statement}** This is the defining characteristic of sampling with replacement.")
                st.balloons()
            else:
                st.warning(
                    f"Incorrect. The key is 'sampling with replacement'. This allows for duplicates and for some original data points to be left out. The correct answer is: **{correct_statement}**.")

    # 7. Bonus: Out-of-Bag (OOB) Samples
    with st.expander("✨ Bonus: Out-of-Bag (OOB) Samples", expanded=False):
        st.subheader("Evaluating Bagging Models with OOB Samples")
        st.markdown("""
        Because bootstrap samples are drawn **with replacement**, each base estimator (e.g., each Decision Tree in a BaggingClassifier) is trained on only about **63.2%** of the original training data on average. The remaining **~36.8%** of the data points that were *not* included in a particular bootstrap sample are called **Out-of-Bag (OOB)** samples for that specific base estimator.

        #### **How OOB Samples are Used:**
        * These OOB samples can be used as a **validation set** for the base estimator that did *not* see them during training.
        * By averaging the predictions of all base estimators on their respective OOB samples, we can get an **internal, unbiased estimate of the ensemble's performance** without needing a separate validation set.
        * This is a powerful feature of Bagging, as it provides a "free" cross-validation-like evaluation.

        **Conceptual Steps for OOB Score:**
        1.  For each data point in the original training set, identify which base estimators *did not* include it in their bootstrap sample (i.e., it's an OOB sample for those estimators).
        2.  For each data point, get predictions only from the base estimators for which it was an OOB sample.
        3.  Aggregate these predictions (e.g., majority vote for classification).
        4.  Compare these aggregated OOB predictions to the actual labels to calculate the OOB score (e.g., OOB accuracy).

        This OOB score is often a very good estimate of the model's generalization performance, comparable to cross-validation.
        """)
        st.info(
            "💡 **Tip:** In `scikit-learn`, you can enable OOB scoring by setting `oob_score=True` in `BaggingClassifier`.")


# --- Main Application Logic (Router) ---
if selected_topic == "Introduction":
    show_introduction()
elif selected_topic == "Mean, Median, Mode":
    topic_mean_median_mode()
elif selected_topic == "Standard Deviation":
    topic_standard_deviation()
elif selected_topic == "Percentiles":
    topic_percentiles()
elif selected_topic == "Data Distribution":
    topic_data_distribution()
elif selected_topic == "Normal Data Distribution":
    topic_normal_data_distribution()
elif selected_topic == "Scatter Plot":
    topic_scatter_plot()
elif selected_topic == "Linear Regression":
    topic_linear_regression()
elif selected_topic == "K-means Clustering":
    topic_k_means_clustering()
elif selected_topic == "Polynomial Regression":
    topic_polynomial_regression()
elif selected_topic == "Multiple Regression":
    topic_multiple_regression()
elif selected_topic == "Feature Scaling":
    topic_feature_scaling()
elif selected_topic == "Train/Test Split":
    topic_train_test_split()
elif selected_topic == "K-nearest Neighbors (KNN)":
    topic_k_nearest_neighbors()
elif selected_topic == "Cross Validation":
    topic_cross_validation()
elif selected_topic == "AUC - ROC Curve":
    topic_auc_roc_curve()
elif selected_topic == "Decision Tree":
    topic_decision_tree()
elif selected_topic == "Confusion Matrix":
    topic_confusion_matrix()
elif selected_topic == "Logistic Regression":
    topic_logistic_regression()
elif selected_topic == "Categorical Data Encoding":
    topic_categorical_data_encoding()
elif selected_topic == "Hierarchical Clustering":
    topic_hierarchical_clustering()
elif selected_topic == "Grid Search":
    topic_grid_search()
elif selected_topic == "Bootstrap Aggregation (Bagging)":
    topic_bootstrap_aggregation()
elif selected_topic == "ML Playground (Coming Soon!)":
    st.info("The ML Playground is under construction! Stay tuned for a sandbox environment where you can apply algorithms end-to-end to your own datasets.")