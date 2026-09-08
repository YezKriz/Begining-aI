# creating student grade manage using sreamlit python Ui

import streamlit as st
import pandas as pd


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Grade Manager",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fb 0%, #eef2f7 100%);
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.06);
    }

    .stat-number {
        font-size: 30px;
        font-weight: 800;
        color: #2563eb;
    }

    .stat-label {
        font-size: 14px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Section heading */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #172554;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #172554;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Grade Function
# ---------------------------------------------------------
def calculate_grade(marks):
    """Return grade based on marks."""

    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------
if "students" not in st.session_state:
    st.session_state.students = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:

    st.markdown("## 🎓 Grade Manager")

    st.markdown("---")

    st.markdown("""
    ### 📌 Grading System

    **A** → 90–100  
    **B** → 80–89  
    **C** → 70–79  
    **D** → 60–69  
    **F** → Below 60
    """)

    st.markdown("---")

    st.info(
        "Enter student details and marks between 0 and 100."
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🎓 Student Grade Manager</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Manage student marks and automatically calculate grades.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Number of Students
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">👥 Student Information</div>',
    unsafe_allow_html=True
)

with st.container():

    st.markdown('<div class="info-card">', unsafe_allow_html=True)

    number_of_students = st.number_input(
        "How many students do you want to enter?",
        min_value=1,
        max_value=100,
        value=3,
        step=1
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Student Input Form
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">📝 Enter Student Details</div>',
    unsafe_allow_html=True
)

with st.form("student_form"):

    student_data = []

    for i in range(int(number_of_students)):

        st.markdown(f"### Student {i + 1}")

        col1, col2 = st.columns([2, 1])

        with col1:
            name = st.text_input(
                "Student Name",
                placeholder="Enter student's name",
                key=f"name_{i}"
            )

        with col2:
            marks = st.number_input(
                "Marks (0–100)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key=f"marks_{i}"
            )

        student_data.append({
            "name": name,
            "marks": marks
        })

        if i < int(number_of_students) - 1:
            st.divider()

    st.markdown("")

    submit_button = st.form_submit_button(
        "🚀 Generate Grades",
        use_container_width=True
    )


# ---------------------------------------------------------
# Process Student Data
# ---------------------------------------------------------
if submit_button:

    try:

        results = []

        validation_error = False

        for i, student in enumerate(student_data):

            name = student["name"].strip()
            marks = student["marks"]

            # Validate name
            if not name:
                st.error(
                    f"❌ Please enter a name for Student {i + 1}."
                )
                validation_error = True
                continue

            # Validate marks
            if marks < 0 or marks > 100:
                st.error(
                    f"❌ Marks for {name} must be between 0 and 100."
                )
                validation_error = True
                continue

            # Calculate grade
            grade = calculate_grade(marks)

            results.append({
                "Student": name,
                "Marks": marks,
                "Grade": grade
            })

        # Display only if validation is successful
        if not validation_error and results:

            st.session_state.students = results

            st.success(
                "✅ Grades generated successfully!"
            )

    except Exception as e:

        st.error(
            f"⚠️ An unexpected error occurred: {e}"
        )


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
if st.session_state.students:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Student Results</div>',
        unsafe_allow_html=True
    )

    df = pd.DataFrame(st.session_state.students)

    # Statistics
    total_students = len(df)
    average_marks = df["Marks"].mean()
    highest_marks = df["Marks"].max()
    passed_students = len(df[df["Grade"] != "F"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{total_students}</div>
                <div class="stat-label">Total Students</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{average_marks:.1f}</div>
                <div class="stat-label">Average Marks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{highest_marks:.0f}</div>
                <div class="stat-label">Highest Marks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{passed_students}</div>
                <div class="stat-label">Passed Students</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    # -----------------------------------------------------
    # Grade Color Function
    # -----------------------------------------------------
    def color_grade(value):

        if value == "A":
            return "background-color: #dcfce7; color: #166534; font-weight: bold;"
        elif value == "B":
            return "background-color: #dbeafe; color: #1e40af; font-weight: bold;"
        elif value == "C":
            return "background-color: #fef3c7; color: #92400e; font-weight: bold;"
        elif value == "D":
            return "background-color: #ffedd5; color: #9a3412; font-weight: bold;"
        elif value == "F":
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"

        return ""

    # Display dataframe
    styled_df = df.style.map(
        color_grade,
        subset=["Grade"]
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Reset Button
    # -----------------------------------------------------
    if st.button("🔄 Clear Results", use_container_width=True):

        st.session_state.students = []

        st.rerun()


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:10px;">
        🎓 <b>Student Grade Manager</b> &nbsp;|&nbsp;
        Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
