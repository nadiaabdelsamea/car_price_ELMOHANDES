import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. إعدادات الصفحة والعنوان
st.set_page_config(page_title="توقع أسعار السيارات 🚗", layout="centered")
st.title("🚗 تطبيق توقع أسعار السيارات")
st.write("قم بإدخال مواصفات السيارة للحصول على السعر المتوقع.")

# 2. تحميل الموديل وقائمة الأعمدة
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('car_price_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_model_artifacts()
except FileNotFoundError:
    st.error("⚠️ عذراً، لم يتم العثور على ملفات الموديل. تأكد من وجود car_price_model.pkl و model_columns.pkl في نفس الفولدر.")
    st.stop()

# 3. استخراج القيم المتاحة للماركات والموديلات وحجم السيارة من الأعمدة المحفوظة
# الأعمدة بتبدأ بـ 'make_', 'model_', 'vehicle_size_' بسبب get_dummies
all_makes = sorted(list(set([col.split('make_')[1] for col in model_columns if col.startswith('make_')])))
all_models = sorted(list(set([col.split('model_')[1] for col in model_columns if col.startswith('model_')])))
all_sizes = sorted(list(set([col.split('vehicle_size_')[1] for col in model_columns if col.startswith('vehicle_size_')])))

# 4. تصميم واجهة المستخدم لإدخال البيانات (Inputs)
st.subheader("📋 مواصفات السيارة")

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

# 5. معالجة البيانات المدخلة لتطابق شكل الـ One-Hot Encoding للموديل
if st.button("احسب السعر المتوقع 💰", type="primary"):
    
    # إنشاء قاموس ببيانات المستخدم (القيم الرقمية أولاً)
    input_data = {
        'year': year,
        'engine_hp': engine_hp,
        'engine_cylinders': engine_cylinders,
        'number_of_doors': number_of_doors,
        'highway_mpg': highway_mpg,
        'city_mpg': city_mpg,
        'popularity': popularity
    }
    
    # تحويله لـ DataFrame من صف واحد
    input_df = pd.DataFrame([input_data])
    
    # إنشاء أعمدة الـ One-Hot Encoding الافتراضية وإعطائها قيمة 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # تفعيل العمود المناسب للماركة والموديل والحجم اللي اختارهم المستخدم (تغيير قيمتهم لـ 1)
    if f'make_{make}' in input_df.columns:
        input_df[f'make_{make}'] = 1
    if f'model_{model_car}' in input_df.columns:
        input_df[f'model_{model_car}'] = 1
    if f'vehicle_size_{vehicle_size}' in input_df.columns:
        input_df[f'vehicle_size_{vehicle_size}'] = 1
        
    # إعادة ترتيب الأعمدة لتطابق تماماً ترتيب أعمدة التدريب
    input_df = input_df[model_columns]
    
    # 6. التنبؤ بالسعر (Predict)
    log_prediction = model.predict(input_df)[0]
    
    # بما إنك مدرب الموديل على np.log1p(price)، لازم نعكس العملية بـ np.expm1
    final_prediction = np.expm1(log_prediction)
    
    # 7. عرض النتيجة للمستخدم
    st.success(f"💵 السعر المتوقع للسيارة هو: **${final_prediction:,.2f}**")
