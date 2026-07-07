# 🎫 Smart AI Customer Support CRM & Routing Engine
An Enterprise-Grade, Machine Learning-Powered Customer Relationship Management (CRM) and Automated Ticket Categorization System built with Python and Streamlit.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Architecture & Upgrades](#-key-architecture--upgrades)
3. [How the AI Models Work](#-how-the-ai-models-work)
4. [File & Directory Structure](#-file--directory-structure)
5. [Step-by-Step Installation Guide](#-step-by-step-installation-guide)
6. [Detailed Tab Breakdown (UI Components)](#-detailed-tab-breakdown-ui-components)
7. [Future Scope & Enhancements](#-future-scope--enhancements)

---

## 🌟 Project Overview
Jab kisi bade platform (jaise E-commerce ya Telecom company) par hazaron customers daily apni complaints bhejte hain, toh unhe manually padhna aur sahi department me bhejna bohot bada task hota hai. 

Yeh project usi problem ko AI ke zariye hal karta hai. Yeh system customer ke dwara likhi gayi problem/text ko khud padhta hai, use analyze karta hai, aur real-time me decide karta hai ki:
1. Yeh ticket kis department (`Ticket Type`) ke paas jaana chahiye.
2. Is issue ki urgency (`Ticket Priority`) kitni zyada hai.

Isko ek premium aur soothing **Light Pink (#FFF0F5) Theme** ke sath design kiya gaya hai taaki corporate desk users ke liye dashboard interactive aur stress-free lage.

---

## 🎀 Key Architecture & Upgrades

* **🎨 Soft Pink Corporate UI:** Pure interface ko custom CSS styles se aesthetic light pink look diya gaya hai, jisme bright white input tiles aur distinct status containers hain.
* **⚡ Double-Engine Classification:** Background me ek nahi, balki **Do alag-alag Logistic Regression Models** chalte hain—ek Ticket Type predict karne ke liye aur ek Priority ke liye.
* **📋 Intelligent Templates Dropdown:** Dropdown me common customer queries (Billing, Technical, Access, Damage) ke structural scripts hain. Jaise hi user unhe select karta hai, description auto-fill ho jata hai.
* **🔍 Live Customer Audit Logs:** Database tab me customer-centric filtration mechanism diya gaya hai jo poore CSV dataset me se real-time input keywords ke matching names ko dynamic table me output karta hai.
* **📊 Visual Charts (Analytics Mode):** Streamlit integration ke sath automated Bar charts hain jo dynamically analytics update karte hain.

---

## ⚙️ How the AI Models Work

Yeh application Natural Language Processing (NLP) aur Machine Learning ke underlying concepts par kaam karti hai:

1. **Text Preprocessing & Cleaning:** Puraane text me se special characters, symbols aur punctuation marks ko Regular Expressions (`re`) ke zariye saaf kiya jata hai aur saare words ko lower-case me convert kiya jata hai.
2. **Feature Extraction (TF-IDF Vectorizer):** ML models text ko direct nahi samajhte. Isliye `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) text words ko mathematical numbers (Vectors) me convert karta hai. Humne top **2,500 highly repeated words (max_features)** ko vector dimensions set kiya hai.
3. **Multi-class Logistic Regression Classifier:** Multi-class training mathematical engine ke zariye processed numbers par probability run hoti hai aur exact class output calculate hokar screen par display hota hai.

---

## 📁 File & Directory Structure

Aapka project folder structure hamesha is hierarchy me hona chahiye taaki script file data file ko automatic search kar sake:

```text
sports tiket/
│
├── customer_support_tickets.csv    # Pura original customer complaints dataset (CSV)
├── tiket1.py                       # Main application python control script 
└── README.md                       # Yeh advanced technical guide file
