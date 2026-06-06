import streamlit as st
import pandas as pd
import numpy as np
import joblib

# إعدادات الصفحة
st.set_page_config(page_title="توقع أسعار السيارات 🚗", layout="centered")
st.title("🚗 تطبيق توقع أسعار السيارات")
st.write("قم بإدخال مواصفات السيارة للحصول على السعر المتوقع مباشرة عبر الموديل.")

# تحميل الموديل فقط مع خاصية الكاش
@st.cache_resource
def load_model():
    # تحميل ملف الموديل الأساسي
    model = joblib.load('car_price_model.pkl')
    return model

try:
    model = load_model()
    # استخراج أسماء الأعمدة مباشرة من الموديل المسبق التدريب
    model_columns = list(model.feature_names_in_)
except Exception as e:
    st.error("⚠️ فشل تحميل ملف الموديل. تأكد من رفع ملف car_price_model.pkl إلى الـ GitHub بنجاح.")
    st.stop()

# استخراج الأسماء النظيفة للماركات والموديلات وحجم السيارة من أعمدة الـ Dummy المستنتجة
all_makes = sorted(list(set([col.split('make_')[1] for col in model_columns if col.startswith('make_')])))
all_models = sorted(list(set([col.split('model_')[1] for col in model_columns if col.startswith('model_')])))
all_sizes = sorted(list(set([col.split('vehicle_size_')[1] for col in model_columns if col.startswith('vehicle_size_')])))

# تصميم الواجهة لإدخال البيانات
st.subheader("📋 مواصفات السيارة المدخلة")
col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("الماركة (Make)", all_makes)
    model_car = st.selectbox("الموديل (Model)", all_models)
    vehicle_size = st.selectbox("حجم السيارة (Vehicle Size)", all_sizes)
    year = st.number_input("سنة الصنع (Year)", min_value=1990, max_value=2026, value=2015, step=1)

with col2:
    engine_hp = st.number_input("قوة المحرك (Engine HP)", min_value=10.0, max_value=1000.0, value=200.0, step=10.0)
    engine_cylinders = st.number_input("عدد السلندرات (Engine Cylinders)", min_value=0.0, max_value=16.0, value=4.0, step=1.0)
    number_of_doors = st.number_input("عدد الأبواب (Number of Doors)", min_value=2.0, max_value=5.0, value=4.0, step=1.0)
    highway_mpg = st.number_input("استهلاك الوقود في الطرق السريعة (Highway MPG)", min_value=5.0, max_value=100.0, value=25.0, step=1.0)
    city_mpg = st.number_input("استهلاك الوقود داخل المدينة (City MPG)", min_value=5.0, max_value=100.0, value=18.0, step=1.0)
    popularity = st.number_input("الشعبية (Popularity)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)

# عند الضغط على زر التوقع
if st.button("احسب السعر المتوقع 💰", type="primary"):
    # تجهيز البيانات الرقمية الأساسية
    input_df = pd.DataFrame([{
        'year': year,
        'engine_hp': engine_hp,
        'engine_cylinders': engine_cylinders,
        'number_of_doors': number_of_doors,
        'highway_mpg': highway_mpg,
        'city_mpg': city_mpg,
        'popularity': popularity
    }])
    
    # بناء الـ One-Hot Encoding ليتطابق مع الـ Features الأصلية للموديل
    encoded_features = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # دمج البيانات الرقمية في الجدول الجديد
    for col in input_df.columns:
        if col in encoded_features.columns:
            encoded_features[col] = input_df[col].values
            
    # تفعيل خيارات المستخدم وتحويلها لـ 1 (One-Hot)
    if f'make_{make}' in encoded_features.columns:
        encoded_features[f'make_{make}'] = 1
    if f'model_{model_car}' in encoded_features.columns:
        encoded_features[f'model_{model_car}'] = 1
    if f'vehicle_size_{vehicle_size}' in encoded_features.columns:
        encoded_features[f'vehicle_size_{vehicle_size}'] = 1
        
    # التنبؤ بالسعر (الناتج يكون Log Price)
    log_prediction = model.predict(encoded_features)[0]
    
    # تحويل السعر من الـ Log إلى القيمة الفعلية بالدولار
    final_prediction = np.expm1(log_prediction)
    
    # عرض النتيجة النهائية للمستخدم
    st.success(f"💵 السعر المتوقع لهذه السيارة هو: **${final_prediction:,.2f}**")
