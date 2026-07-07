import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Support Routing System", 
    page_icon="🎫", 
    layout="wide"
)

# Custom Styling for UI - Light Pink Theme Background
st.markdown("""
    <style>
    /* Main background color set to Light Pink */
    .stApp {
        background-color: #FFF0F5; 
    }
    .main-title { font-size: 36px; font-weight: bold; color: #9C27B0; margin-bottom: 5px; }
    .sub-title { font-size: 15px; color: #4A148C; margin-bottom: 25px; }
    .metric-box { background-color: #FFFFFF; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #F8BBD0; box-shadow: 2px 2px 10px rgba(0,0,0,0.03); }
    .customer-card { background-color: #FCE4EC; padding: 15px; border-radius: 8px; border-left: 5px solid #E91E63; margin-top: 15px; }
    
    /* Input field text colors stability */
    div[data-baseweb="select"] > div { background-color: white !important; }
    div[data-baseweb="textarea"] > div { background-color: white !important; }
    div[data-baseweb="input"] > div { background-color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎫 Smart AI Support CRM & Router</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Machine Learning Engine customized with an Elegant Theme for Account Management.</div>', unsafe_allow_html=True)

# 2. Optimized Data Loading & ML Pipeline
@st.cache_resource
def load_and_train_pipeline():
    df = pd.read_csv('customer_support_tickets.csv')
    
    # Text Processing
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text
        return ""
    
    # Fill missing values for security
    df['Ticket Description'] = df['Ticket Description'].fillna('')
    df['Ticket Type'] = df['Ticket Type'].fillna('General')
    df['Ticket Priority'] = df['Ticket Priority'].fillna('Low')
    df['Customer Name'] = df['Customer Name'].fillna('Unknown Customer')
    df['Customer Email'] = df['Customer Email'].fillna('No Email Provided')
    
    cleaned_data = df['Ticket Description'].apply(clean_text)
    
    # Train Models
    tfidf = TfidfVectorizer(max_features=2500, stop_words='english')
    X = tfidf.fit_transform(cleaned_data)
    
    model_type = LogisticRegression(max_iter=300, n_jobs=-1).fit(X, df['Ticket Type'])
    model_priority = LogisticRegression(max_iter=300, n_jobs=-1).fit(X, df['Ticket Priority'])
    
    return tfidf, model_type, model_priority, df

with st.spinner('⚙️ Syncing CRM database and spinning up AI models...'):
    tfidf, model_type, model_priority, df = load_and_train_pipeline()

# 3. Sidebar Engine Metrics
with st.sidebar:
    st.header("⚙️ CRM Settings")
    st.info(f"📁 Dataset Loaded: **{len(df):,} Rows**")
    st.success("🤖 Neural Text Link: Connected")
    st.markdown("---")
    st.write("✨ **Theme:** Soft Pink Corporate Layout")

# 4. Main Layout Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Live AI Ticket Router", "📂 Customer Issue Logs", "📊 Visual Dashboard"])

with tab1:
    st.subheader("📝 New Ticket Profiler")
    
    # Form fields adding extra CSV attributes
    col_a, col_b = st.columns(2)
    with col_a:
        cust_name = st.text_input("Customer Name:", placeholder="Anjali Saroj")
    with col_b:
        cust_email = st.text_input("Customer Email:", placeholder="customer@example.com")
    
    st.markdown("---")
    st.write("💡 **Quick Templates:** Niche diye dropdown se koi bhi select karein ya niche box me apna khud ka issue type karein:")
    
    # Dictionary of common customer issues matching the patterns in standard ticket datasets
    issue_templates = {
        "Custom Issue (Type manually below)": "",
        "💻 Technical Issue": "My software keeps crashing upon startup, and I am getting a memory dump error. Please assist.",
        "💳 Billing & Payment Problem": "I was charged twice for my monthly subscription premium package. Please process a refund.",
        "🔑 Account Access Issue": "I am unable to login to my dashboard, it says user not found even after password reset.",
        "📦 Product Delivery / Damage": "The package arrived today but the main electronic component inside is completely broken and shattered.",
        "⚠️ Network / Connectivity Error": "The server is responding very slowly and dropping connection repeatedly while fetching data."
    }
    
    # Dropdown for Quick Issue selection
    selected_template = st.selectbox("Select a Common Query Type:", list(issue_templates.keys()))
    default_text = issue_templates[selected_template]
        
    user_input = st.text_area(
        "Enter Issue / Complaint Description:", 
        value=default_text,
        height=120,
        placeholder="Type the exact customer issue here..."
    )
    
    if st.button("⚡ Process & Route Ticket", type="primary"):
        if user_input.strip() != "":
            # Prediction Logic
            cleaned = user_input.lower()
            cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
            vectorized = tfidf.transform([cleaned])
            
            p_type = model_type.predict(vectorized)[0]
            p_priority = model_priority.predict(vectorized)[0]
            
            st.markdown("---")
            st.markdown("### 🎯 Automation Routing Meta")
            
            # Display result boxes
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-box">📬 <b>Assigned Department</b><br><h2 style="color:#2563EB; margin:10px 0;">{p_type}</h2></div>', unsafe_allow_html=True)
            with c2:
                p_color = "#DC2626" if p_priority in ["High", "Urgent", "Critical"] else "#D97706" if p_priority == "Medium" else "#059669"
                st.markdown(f'<div class="metric-box">🚨 <b>Calculated Priority</b><br><h2 style="color:{p_color}; margin:10px 0;">{p_priority}</h2></div>', unsafe_allow_html=True)
            
            # Print Customer Profile summary if provided
            if cust_name:
                st.markdown(f"""
                <div class="customer-card">
                    👤 <b>Generated Dispatch Profile:</b><br>
                    <b>Account Holder:</b> {cust_name} | <b>Routing Destination:</b> {cust_email if cust_email else 'N/A'}<br>
                    <small><i>Status: Ticket automatically generated and assigned to {p_type} queue.</i></small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Error: Cannot analyze empty issue description!")

with tab2:
    st.subheader("🔍 Historical Database & Customer Audit")
    st.write("Aap niche diye gaye search box se Kisi bhi Customer ke tickets dhoondh sakte hain:")
    
    search_query = st.text_input("🔍 Search by Customer Name:", placeholder="Type a name to filter... (e.g., John)")
    
    # Filter dataset based on customer name search
    if search_query:
        filtered_df = df[df['Customer Name'].str.contains(search_query, case=False, na=False)]
    else:
        filtered_df = df

    st.write(f"Showing **{len(filtered_df[:100])}** matches out of entire dataset:")
    st.dataframe(
        filtered_df[['Customer Name', 'Customer Email', 'Ticket Subject', 'Ticket Type', 'Ticket Priority']].head(20), 
        use_container_width=True
    )

with tab3:
    st.subheader("📈 Operational Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Total Load by Department**")
        st.bar_chart(df['Ticket Type'].value_counts(), color="#E91E63")
    with col2:
        st.write("**Ticket Distribution by Priority**")
        st.bar_chart(df['Ticket Priority'].value_counts(), color="#9C27B0")
