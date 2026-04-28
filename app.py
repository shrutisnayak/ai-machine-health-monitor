import streamlit as st
import numpy as np
import time
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
import os

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "data_ready" not in st.session_state:
    st.session_state.data_ready = False

if "df" not in st.session_state:
    st.session_state.df = None

if "anomalies_df" not in st.session_state:
    st.session_state.anomalies_df = None

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Smart Machine Monitor", layout="wide")

# CUSTOM CSS
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.stMetric {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🛠️ AI-Powered Predictive Maintenance Dashboard")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Machine Details")
machine_id = st.sidebar.text_input("Machine ID", "MCH-101")
location = st.sidebar.text_input("Location", "Mumbai Plant")

st.sidebar.markdown("---")
st.sidebar.write(f"Monitoring: **{machine_id}**")
st.sidebar.write(f"Location: **{location}**")

# -------------------------------
# SIMULATE SENSOR DATA
# -------------------------------
np.random.seed(42)

normal_temp = np.random.normal(loc=70, scale=5, size=100)
anomaly_temp = np.random.normal(loc=95, scale=2, size=10)
temperature = np.concatenate([normal_temp, anomaly_temp])

vibration = np.random.normal(loc=5, scale=1, size=len(temperature))

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(normal_temp.reshape(-1, 1))

# -------------------------------
# UI COMPONENTS
# -------------------------------
col1, col2, col3 = st.columns(3)

temp_metric = col1.empty()
vib_metric = col2.empty()
status_metric = col3.empty()

chart = st.line_chart([])

status_box = st.empty()

st.subheader("📜 Alert History")
alert_placeholder = st.empty()

# -------------------------------
# RUN SIMULATION BUTTON
# -------------------------------
if st.button("▶️ Run Simulation"):
    st.session_state.alerts = []
    st.session_state.data_ready = False

    data = []
    window_size = 5
    progress = st.progress(0)

    for i, temp in enumerate(temperature):
        progress.progress((i + 1) / len(temperature))
        vib = vibration[i]
        data.append(temp)

        # Trend detection
        if len(data) > window_size:
            trend = np.mean(data[-5:]) - np.mean(data[-10:-5])
            if trend > 2:
                status_box.warning("📈 Rising temperature trend detected!")

        # AI prediction
        prediction = model.predict([[temp]])

        # Update metrics
        temp_metric.metric("🌡️ Temperature", f"{temp:.2f} °C")
        vib_metric.metric("🔧 Vibration", f"{vib:.2f}")

        # Decision logic
        if temp > 90 and vib > 7:
            status_metric.metric("Status", "🚨 Critical")
            status_box.error(f"🚨 CRITICAL ALERT!\nTemp: {temp:.2f}°C | Vibration: {vib:.2f}")
            st.session_state.alerts.append(f"Time {i}: CRITICAL - Temp {temp:.2f}, Vib {vib:.2f}")

        elif prediction[0] == -1:
            status_metric.metric("Status", "⚠️ Warning")
            status_box.warning(f"⚠️ Warning: Unusual pattern\nTemp: {temp:.2f}°C")
            st.session_state.alerts.append(f"Time {i}: WARNING - Temp {temp:.2f}")

        else:
            status_metric.metric("Status", "✅ Normal")
            status_box.success(f"✅ Normal Operation\nTemp: {temp:.2f}°C")

        chart.add_rows([temp])
        alert_placeholder.write(st.session_state.alerts[-5:])

        time.sleep(0.05)

    # -------------------------------
    # STORE DATA AFTER SIMULATION
    # -------------------------------
    df = pd.DataFrame({
        "Temperature": temperature,
        "Vibration": vibration
    })

    df["Temp_MA"] = df["Temperature"].rolling(window=5).mean()

    preds = model.predict(df[["Temperature"]])
    df["Anomaly"] = preds

    anomalies_df = df[df["Anomaly"] == -1]

    st.session_state.df = df
    st.session_state.anomalies_df = anomalies_df
    st.session_state.data_ready = True

# -------------------------------
# SHOW ANALYTICS (PERSISTENT)
# -------------------------------
def generate_pdf(df, anomalies_df, machine_id="MCH-101", location="Plant"):
    file_path = "machine_report.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # -------------------------
    # HEADER
    # -------------------------
    elements.append(Paragraph("AI-Powered Machine Health Report", styles['Title']))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"<b>Machine ID:</b> {machine_id}", styles['Normal']))
    elements.append(Paragraph(f"<b>Location:</b> {location}", styles['Normal']))
    elements.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # -------------------------
    # HEALTH SUMMARY
    # -------------------------
    total = len(df)
    anomalies = len(anomalies_df)
    health_score = 100 - (anomalies / total) * 100

    summary_data = [
        ["Metric", "Value"],
        ["Total Readings", total],
        ["Anomalies Detected", anomalies],
        ["Health Score", f"{health_score:.1f}%"]
    ]

    table = Table(summary_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER')
    ]))

    elements.append(Paragraph("System Summary", styles['Heading2']))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # -------------------------
    # CHARTS
    # -------------------------

    # Temperature Trend
    plt.figure()
    plt.plot(df["Temperature"], label="Temperature")
    plt.plot(df["Temp_MA"], label="Moving Avg")
    plt.legend()
    plt.title("Temperature Trend")
    temp_chart = "temp_chart.png"
    plt.savefig(temp_chart)
    plt.close()

    elements.append(Paragraph("Temperature Trend", styles['Heading3']))
    elements.append(Image(temp_chart, width=400, height=220))
    elements.append(Spacer(1, 10))

    # Distribution
    plt.figure()
    plt.hist(df["Temperature"], bins=20)
    plt.title("Temperature Distribution")
    dist_chart = "dist_chart.png"
    plt.savefig(dist_chart)
    plt.close()

    elements.append(Paragraph("Temperature Distribution", styles['Heading3']))
    elements.append(Image(dist_chart, width=400, height=220))
    elements.append(Spacer(1, 10))

    # Scatter
    plt.figure()
    plt.scatter(df["Temperature"], df["Vibration"])
    plt.xlabel("Temperature")
    plt.ylabel("Vibration")
    plt.title("Temp vs Vibration")
    scatter_chart = "scatter_chart.png"
    plt.savefig(scatter_chart)
    plt.close()

    elements.append(Paragraph("Sensor Correlation", styles['Heading3']))
    elements.append(Image(scatter_chart, width=400, height=220))
    elements.append(Spacer(1, 12))

    # -------------------------
    # CRITICAL EVENTS TABLE
    # -------------------------
    elements.append(Paragraph("Critical Events (Sample)", styles['Heading2']))

    if not anomalies_df.empty:
        sample = anomalies_df.head(5)

        table_data = [["Index", "Temp", "Vibration"]]
        for idx, row in sample.iterrows():
            table_data.append([idx, f"{row['Temperature']:.2f}", f"{row['Vibration']:.2f}"])

        event_table = Table(table_data)
        event_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.red),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))

        elements.append(event_table)
    else:
        elements.append(Paragraph("No anomalies detected.", styles['Normal']))

    # -------------------------
    # BUILD PDF
    # -------------------------
    doc.build(elements)

    # cleanup images
    for f in [temp_chart, dist_chart, scatter_chart]:
        if os.path.exists(f):
            os.remove(f)

    return file_path
  
if st.session_state.data_ready:

    df = st.session_state.df
    anomalies_df = st.session_state.anomalies_df

    st.success("✅ Simulation Completed. Showing Analytics Below")

    st.markdown("---")
    st.header("📊 Historical Analytics")

    st.subheader("📈 Temperature Trend with Moving Average")
    st.line_chart(df[["Temperature", "Temp_MA"]])

    st.subheader("📉 Temperature Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Temperature"], bins=20)
    st.pyplot(fig)

    st.subheader("🔥 Anomaly Timeline")
    st.line_chart(df["Temperature"])

    st.write("Detected Anomalies:")
    st.dataframe(anomalies_df)

    st.subheader("📊 Temperature vs Vibration")
    st.scatter_chart(df[["Temperature", "Vibration"]])

    st.subheader("🧠 Insights Summary")
    st.write(f"Average Temperature: {df['Temperature'].mean():.2f} °C")
    st.write(f"Max Temperature: {df['Temperature'].max():.2f} °C")
    st.write(f"Total Anomalies Detected: {len(anomalies_df)}")

    st.markdown("### 🧾 System Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Readings", len(df))
    col2.metric("Anomalies", len(anomalies_df))
    col3.metric("Health Score", f"{100 - (len(anomalies_df)/len(df))*100:.1f}%")

    # -------------------------------
    # DOWNLOAD BUTTON (NO RESET ISSUE)
    # -------------------------------
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="machine_health_report.csv",
        mime="text/csv",
    )

    pdf_file = generate_pdf(df,anomalies_df,machine_id=machine_id,location=location)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name="machine_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Click 'Run Simulation' to start monitoring")