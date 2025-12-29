import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io

# إعداد الصفحة
st.set_page_config(
    page_title="تحليل أسعار لاعبي كرة القدم",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص التصميم باستخدام CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1E3A8A;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. توليد بيانات وهمية (Synthetic Data Generation)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_samples=2000):
    """
    توليد بيانات لاعبين وهمية لتدريب النموذج عليها لغرض العرض.
    """
    np.random.seed(42)
    
    positions = ['GK', 'CB', 'LB', 'RB', 'CM', 'CAM', 'CDM', 'LW', 'RW', 'ST']
    feet = ['يمين', 'يسار']
    injury_levels = ['لا توجد', 'خفيفة', 'متوسطة', 'خطيرة']
    fame_levels = ['غير معروف', 'محلي', 'عالمي']
    contract_statuses = ['نعم', 'لا']
    match_statuses = ['أساسي', 'احتياطي', 'تدويري']
    
    data = {
        'age': np.random.randint(16, 40, n_samples),
        'height_cm': np.random.randint(160, 200, n_samples),
        'weight_kg': np.random.randint(60, 100, n_samples),
        'preferred_foot': np.random.choice(feet, n_samples),
        'position': np.random.choice(positions, n_samples),
        # مهارات (مرتبطة قليلاً بالمركز عشوائياً لواقعية بسيطة)
        'pace': np.random.randint(40, 99, n_samples),
        'physic': np.random.randint(40, 99, n_samples),
        'shooting': np.random.randint(30, 99, n_samples),
        'passing': np.random.randint(40, 99, n_samples),
        'dribbling': np.random.randint(40, 99, n_samples),
        'controlling': np.random.randint(40, 99, n_samples),
        # انضباط وإصابات
        'discipline': np.random.randint(1, 11, n_samples),
        'is_injured': np.random.choice(['نعم', 'لا'], n_samples, p=[0.2, 0.8]),
        'injury_degree': np.random.choice(injury_levels, n_samples),
        # إحصائيات
        'matches_played': np.random.randint(0, 50, n_samples),
        'goals': np.random.randint(0, 30, n_samples),
        'assists': np.random.randint(0, 20, n_samples),
        'participation_status': np.random.choice(match_statuses, n_samples),
        # شهرة وتعاقد
        'fame_level': np.random.choice(fame_levels, n_samples, p=[0.5, 0.3, 0.2]),
        'has_contract': np.random.choice(contract_statuses, n_samples),
        'contract_years': np.random.randint(0, 6, n_samples),
        'league_strength': np.random.randint(1, 6, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # منطق بسيط لحساب سعر وهمي (Target) بناءً على الميزات
    # السعر = عامل * (المهارات + الشهرة + (صغر العمر للشباب) - الإصابات)
    
    base_price = (
        df['pace'] * 1000 + 
        df['shooting'] * 1500 + 
        df['passing'] * 1200 + 
        df['dribbling'] * 1300 + 
        df['matches_played'] * 5000 +
        df['goals'] * 10000 +
        (40 - df['age']) * 20000 # اللاعب الأصغر أغلى
    )
    
    # تأثير الشهرة
    fame_multiplier = df['fame_level'].map({'غير معروف': 1, 'محلي': 5, 'عالمي': 20})
    df['price'] = base_price * fame_multiplier
    
    # تأثير الدوري
    df['price'] = df['price'] * df['league_strength'] * 0.5
    
    # تأثير الإصابة
    injury_penalty = df['injury_degree'].map({'لا توجد': 1, 'خفيفة': 0.9, 'متوسطة': 0.7, 'خطيرة': 0.4})
    df['price'] = df['price'] * injury_penalty
    
    # إضافة عشوائية
    df['price'] = df['price'] + np.random.normal(0, df['price']*0.1, n_samples)
    
    return df

# -----------------------------------------------------------------------------
# 2. بناء النموذج (Model Building)
# -----------------------------------------------------------------------------
@st.cache_resource
def build_model(df):
    """
    بناء وتدريب نموذج RandomForestRegressor.
    """
    X = df.drop('price', axis=1)
    y = df['price']
    
    # تحديد الأعمدة الرقمية والفئوية
    numeric_features = [
        'age', 'height_cm', 'weight_kg', 'pace', 'physic', 'shooting', 
        'passing', 'dribbling', 'controlling', 'discipline', 
        'matches_played', 'goals', 'assists', 'contract_years', 'league_strength'
    ]
    
    categorical_features = [
        'preferred_foot', 'position', 'is_injured', 'injury_degree', 
        'participation_status', 'fame_level', 'has_contract'
    ]
    
    # خطوات المعالجة
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # النموذج النهائي
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # التدريب
    model.fit(X_train, y_train)
    
    # التقييم
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, r2, mae, X_train, y_train

# -----------------------------------------------------------------------------
# 3. واجهة المستخدم (UI Layout)
# -----------------------------------------------------------------------------

# تحميل البيانات وتجهيز النموذج
with st.spinner('جاري تحميل البيانات وتدريب النموذج الذكي...'):
    df_data = generate_synthetic_data(3000)
    model, r2_score_val, mae_val, X_train_ref, y_train_ref = build_model(df_data)

st.title("⚽ توقع القيمة السوقية للاعبي كرة القدم")
st.markdown("### نظام ذكي لتحليل وتوقع أسعار اللاعبين باستخدام الذكاء الاصطناعي")

st.sidebar.header("🎯 لوحة التحكم")
st.sidebar.success(f"دقة النموذج (R²): {r2_score_val:.2f}")
st.sidebar.info(f"متوسط الخطأ المطلق: {mae_val:,.0f} $")

# النموذج داخل Form لتنظيم المدخلات
with st.form("player_data_form"):
    
    # --- القسم 1: البيانات الأساسية ---
    st.markdown("#### 1️⃣ البيانات الأساسية")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("العمر (سنة)", 15, 45, 24)
        height = st.number_input("الطول (سم)", 150, 220, 180)
        weight = st.number_input("الوزن (كغ)", 50, 110, 75)
    
    with col2:
        position = st.selectbox("مركز اللعب", 
            ['GK', 'CB', 'LB', 'RB', 'CM', 'CAM', 'CDM', 'LW', 'RW', 'ST'])
        foot = st.selectbox("القدم المفضلة", ['يمين', 'يسار'])
    
    with col3:
        pass # مساحة فارغة أو إضافة شعار لاحقاً

    st.markdown("---")

    # --- القسم 2: المهارات الفنية ---
    st.markdown("#### 2️⃣ المهارات الفنية (0 - 100)")
    c1, c2, c3 = st.columns(3)
    with c1:
        pace = st.slider("السرعة", 0, 100, 70)
        shooting = st.slider("التسديد", 0, 100, 60)
    with c2:
        physic = st.slider("القوة البدنية", 0, 100, 75)
        passing = st.slider("التمرير", 0, 100, 65)
    with c3:
        dribbling = st.slider("المراوغة", 0, 100, 70)
        controlling = st.slider("التحكم بالكرة", 0, 100, 72)

    st.markdown("---")

    # --- القسم 3 & 4: الانضباط والإحصائيات ---
    st.markdown("#### 3️⃣ الأداء، الانضباط، والإصابات")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.caption("الانضباط والإصابات")
        discipline = st.slider("مستوى الانضباط (1-10)", 1, 10, 8)
        is_injured_val = st.radio("هل يعاني من إصابة؟", ['لا', 'نعم'], horizontal=True)
        injury_degree = st.selectbox("درجة الإصابة", ['لا توجد', 'خفيفة', 'متوسطة', 'خطيرة'])
        if is_injured_val == 'لا':
            injury_degree = 'لا توجد' # تصحيح تلقائي
        
    with col_b:
        st.caption("الإحصائيات للموسم الحالي")
        matches = st.number_input("عدد المباريات", 0, 100, 20)
        goals = st.number_input("عدد الأهداف", 0, 100, 5)
        assists = st.number_input("عدد الصناعات (Assists)", 0, 100, 3)
        part_status = st.selectbox("حالة المشاركة", ['أساسي', 'احتياطي', 'تدويري'])

    st.markdown("---")

    # --- القسم 5: الشهرة والتعاقد ---
    st.markdown("#### 4️⃣ الشهرة والعقد")
    col_x, col_y = st.columns(2)
    with col_x:
        fame = st.selectbox("مستوى الشهرة", ['غير معروف', 'محلي', 'عالمي'])
        league_str = st.slider("قوة الدوري الحالي (1-5)", 1, 5, 3)
    with col_y:
        has_contract_val = st.radio("هل مرتبط بعقد؟", ['نعم', 'لا'], horizontal=True)
        contract_years = 0
        if has_contract_val == 'نعم':
            contract_years = st.slider("سنوات العقد المتبقية", 0, 10, 2)

    submitted = st.form_submit_button("🚀 تحليل وتوقع سعر اللاعب")

# -----------------------------------------------------------------------------
# 4. منطق التوقع (Prediction) والمقارنة
# -----------------------------------------------------------------------------
if submitted:
    # 1. تجهيز بيانات الإدخال
    input_data = {
        'age': [age], 'height_cm': [height], 'weight_kg': [weight],
        'preferred_foot': [foot], 'position': [position],
        'pace': [pace], 'physic': [physic], 'shooting': [shooting],
        'passing': [passing], 'dribbling': [dribbling], 'controlling': [controlling],
        'discipline': [discipline], 'is_injured': [is_injured_val],
        'injury_degree': [injury_degree],
        'matches_played': [matches], 'goals': [goals], 'assists': [assists],
        'participation_status': [part_status],
        'fame_level': [fame], 'has_contract': [has_contract_val],
        'contract_years': [contract_years], 'league_strength': [league_str]
    }
    
    player_df = pd.DataFrame(input_data)
    
    # 2. التوقع
    predicted_price = model.predict(player_df)[0]
    
    # تصحيح عدم وجود قيم سالبة
    predicted_price = max(0, predicted_price)
    
    # 3. تصنيف المستوى (Logic بسيط بناءً على السعر للتبسيط)
    if predicted_price < 1_000_000:
        level = "ضعيف / ناشئ"
        color = "gray"
    elif predicted_price < 10_000_000:
        level = "جيد"
        color = "blue"
    elif predicted_price < 50_000_000:
        level = "جيد جداً"
        color = "orange"
    else:
        level = "ممتاز / نجم"
        color = "green"

    # -------------------------------------------------------------------------
    # 5. عرض النتائج والمقارنة
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("📊 نتائج التحليل")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"### 💰 السعر المتوقع: :green[{predicted_price:,.2f} $]")
        st.markdown(f"### ⭐ تصنيف المستوى: :{color}[{level}]")
    
    # المقارنة الذكية
    # نحصل على متوسطات اللاعبين في نفس المركز من بيانات التدريب
    same_pos_data = X_train_ref[X_train_ref['position'] == position].copy()
    same_pos_data['price'] = y_train_ref.loc[same_pos_data.index] # استرجاع السعر
    
    avg_price_pos = same_pos_data['price'].mean()
    avg_goals_pos = same_pos_data['goals'].mean()
    avg_pace_pos = same_pos_data['pace'].mean()
    
    price_diff = predicted_price - avg_price_pos
    
    if price_diff < -avg_price_pos * 0.2:
        verdict = "أقل من المتوسط (صفقة محتملة)"
        verdict_color = "green"
    elif price_diff > avg_price_pos * 0.2:
        verdict = "أعلى من المتوسط (قد يكون مبالغ فيه)"
        verdict_color = "red"
    else:
        verdict = "سعر عادل (قريب من المتوسط)"
        verdict_color = "orange"
        
    with res_col2:
        st.info("💡 مقارنة مع نفس المركز")
        st.write(f"متوسط سعر المركز ({position}): **{avg_price_pos:,.2f} $**")
        st.markdown(f"الحكم: :{verdict_color}[**{verdict}**]")

    # جدول تفصيلي للمقارنة
    st.subheader("مقارنة المهارات بالأرقام")
    comp_df = pd.DataFrame({
        'المعيار': ['السعر المتوقع', 'الأهداف', 'السرعة', 'القوة البدنية'],
        'اللاعب الحالي': [predicted_price, goals, pace, physic],
        f'متوسط المركز ({position})': [avg_price_pos, avg_goals_pos, avg_pace_pos, same_pos_data['physic'].mean()],
    })
    st.table(comp_df.set_index('المعيار').style.format("{:,.1f}"))

    # الرسم البياني
    st.subheader("📈 الرسم البياني للمقارنة")
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Player Price', 'Avg Position Price']
    values = [predicted_price, avg_price_pos]
    colors = ['#4CAF50', '#1E3A8A']
    
    ax.bar(categories, values, color=colors)
    ax.set_ylabel('Price (USD)')
    ax.set_title(f'Comparison: Player vs {position} Average')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    st.pyplot(fig)

    # -------------------------------------------------------------------------
    # 6. تصدير التقرير (Excel Export)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 تصدير التقرير")
    
    # إعداد ملف Excel في الذاكرة
    output = io.BytesIO()
    workbook = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # ورقة 1: بيانات اللاعب
    player_df['predicted_price'] = predicted_price
    player_df['level_class'] = level
    player_df['comparison_verdict'] = verdict
    player_df.to_excel(workbook, sheet_name='بيانات اللاعب', index=False)
    
    # ورقة 2: تفاصيل المقارنة
    comp_df.to_excel(workbook, sheet_name='المقارنة', index=True)
    
    workbook.close()
    processed_data = output.getvalue()
    
    st.download_button(
        label="📄 تحميل التقرير (Excel)",
        data=processed_data,
        file_name=f'player_report_{position}_{age}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

st.markdown("---")
st.caption("تم التطوير بواسطة: مساعد الذكاء الاصطناعي (Agentic AI) 🤖 | جميع الحقوق محفوظة © 2025")
