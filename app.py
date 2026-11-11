import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Thang màu chuẩn cho seaborn/matplotlib
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Dự đoán giá nhà - K-Means Clustering", layout="wide")

st.title("🏡 Đồ án Phân tích & Dự đoán Giá nhà bằng K-Means Clustering")

# Định nghĩa tập đặc trưng và biến mục tiêu
FEATURES = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'HouseAge']
TARGET = 'SalePrice'

# --- 1. ĐỌC DỮ LIỆU ---
@st.cache_data
def load_data():
    try:
        train_df = pd.read_excel('nhom4_train_data.xlsx')
        test_df = pd.read_excel('nhom4_test_data.xlsx')
    except Exception:
        # Trong trường hợp file ở thư mục khác hoặc upload trực tiếp
        train_df = pd.read_excel('nhom4_train_data_2.xlsx')
        test_df = pd.read_excel('nhom4_test_data_2.xlsx')
    return train_df, test_df

try:
    train_df, test_df = load_data()
except Exception as e:
    st.error(f"Không thể đọc file dữ liệu. Vui lòng kiểm tra file excel đầu vào! Chi tiết lỗi: {e}")
    st.stop()

# --- SIDEBAR: CẤU HÌNH & DỰ ĐOÁN ĐỒNG THỜI ---
st.sidebar.header("⚙️ Cấu hình Mô hình")
k_clusters = st.sidebar.slider("Số lượng cụm (K):", min_value=2, max_value=8, value=3, step=1)

st.sidebar.header("🔮 Nhập thông số nhà mới để dự đoán")
user_input = {}
for col in FEATURES:
    val_min = float(train_df[col].min())
    val_max = float(train_df[col].max())
    val_mean = float(train_df[col].mean())
    user_input[col] = st.sidebar.number_input(f"{col}:", value=round(val_mean, 2), min_value=val_min, max_value=val_max)

# --- THUẬT TOÁN K-MEANS TRÊN TẬP TRAIN ---
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

# Khởi tạo và huấn luyện mô hình K-Means
kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
train_df['Cluster'] = kmeans_model.fit_predict(X_train)

# Tâm cụm (Centroids)
centroids = kmeans_model.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=FEATURES)
centroids_df.index = [f"Cụm {i}" for i in range(k_clusters)]

# Giá nhà trung bình từng cụm
cluster_price_map = train_df.groupby('Cluster')[TARGET].mean().to_dict()
centroids_df['SalePrice_TrungBình'] = [cluster_price_map[i] for i in range(k_clusters)]

# 📑 TAB GIAO DIỆN CHÍNH
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dữ liệu gốc & Elbow", 
    "🎯 Cụm & Centroid (K=3)", 
    "📈 Đánh giá trên Test Set", 
    "🔮 Dự đoán Nhà mới"
])

# --- TAB 1: DỮ LIỆU & PHƯƠNG PHÁP ELBOW ---
with tab1:
    st.subheader("1. XEM TRƯỚC DỮ LIỆU GỐC (GIỮ NGUYÊN SỐ LIỆU)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Tập Huấn luyện (Train Data - 800 mẫu)")
        st.dataframe(train_df[FEATURES + [TARGET]].head(10))
    with col2:
        st.write("##### Tập Kiểm thử (Test Data - 200 mẫu)")
        st.dataframe(test_df[FEATURES + [TARGET]].head(10))

    st.subheader("2. XÁC ĐỊNH ĐIỂM GÃY ELBOW (WCSS)")
    wcss_list = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train)
        wcss_list.append(km.inertia_)

    wcss_df = pd.DataFrame({"Số cụm K": list(k_range), "WCSS": wcss_list})
    
    col_wcss1, col_wcss2 = st.columns([1, 2])
    with col_wcss1:
        st.write("Bảng giá trị WCSS theo K:")
        st.dataframe(wcss_df.style.format({"WCSS": "{:,.2f}"}))
    with col_wcss2:
        fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))
        ax_elbow.plot(k_range, wcss_list, marker='o', color='crimson', linewidth=2)
        ax_elbow.set_title("Biểu đồ Elbow Method (WCSS theo K)")
        ax_elbow.set_xlabel("Số lượng cụm K")
        ax_elbow.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
        ax_elbow.axvline(x=3, color='navy', linestyle='--', label='Điểm gãy tối ưu (K=3)')
        ax_elbow.legend()
        st.pyplot(fig_elbow)

# --- TAB 2: CHI TIẾT TÂM CỤM & KHOẢNG CÁCH CENTROID ---
with tab2:
    st.subheader("1. BẢNG TÂM CENTROID VÀ ĐẶC ĐIỂM TRUNG BÌNH TỪNG CỤM")
    st.dataframe(centroids_df.style.format("{:.4f}"))

    st.subheader("2. TÍNH KHOẢNG CÁCH EUCLIDEAN VÀ GÁN CỤM CHO TẬP TRAIN")
    # Tính khoảng cách từ từng mẫu đến từng Centroid
    distances = np.linalg.norm(X_train.values[:, np.newaxis] - centroids, axis=2)
    dist_cols = [f"KhoangCach_Cum_{i}" for i in range(k_clusters)]
    dist_df = pd.DataFrame(distances, columns=dist_cols)
    
    train_detail = pd.concat([X_train, dist_df, train_df[['Cluster', TARGET]]], axis=1)
    st.write("Bảng tính khoảng cách chi tiết tới Centroids (10 mẫu đầu tập Train):")
    st.dataframe(train_detail.head(10).style.format({c: "{:.4f}" for c in dist_cols}))

    st.subheader("3. BIỂU ĐỒ PHÂN BỐ GIÁ VÀ CỤM")
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        sns.histplot(train_df[TARGET], kde=True, color='teal', ax=ax_dist)
        ax_dist.set_title("Biểu đồ phân bố Giá nhà (SalePrice)")
        st.pyplot(fig_dist)

    with col_fig2:
        fig_cluster, ax_cluster = plt.subplots(figsize=(6, 4))
        scatter = ax_cluster.scatter(X_train['OverallQual'], X_train['GrLivArea'], c=train_df['Cluster'], cmap='viridis', alpha=0.6)
        ax_cluster.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
        ax_cluster.set_xlabel('OverallQual (Đã chuẩn hóa)')
        ax_cluster.set_ylabel('GrLivArea (Đã chuẩn hóa)')
        ax_cluster.set_title('Phân cụm nhà (OverallQual vs GrLivArea)')
        ax_cluster.legend()
        st.pyplot(fig_cluster)

    st.subheader("4. GIẢI THÍCH Ý NGHĨA TỪNG CỤM (K=3)")
    st.markdown("""
    * **Cụm 0 (Chất lượng cao & Nhà mới):** Có `OverallQual` cao (0.69) và `HouseAge` rất thấp (-0.85, nhà mới xây). Giá nhà trung bình cao nhất (~391,778 USD).
    * **Cụm 1 (Chất lượng trung bình/thấp):** Có `OverallQual` thấp (-1.00), diện tích garage nhỏ. Giá nhà trung bình thấp hơn (~387,431 USD).
    * **Cụm 2 (Garage rộng & Nhà cũ hơn):** Có `GarageCars` cao nhất (0.55), nhưng `HouseAge` cao hơn (0.76). Giá nhà trung bình ở mức tương đương cụm 0 (~391,607 USD).
    """)

# --- TAB 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---
with tab3:
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Dự đoán cụm cho tập Test
    test_df['Cluster_Pred'] = kmeans_model.predict(X_test)
    test_df['SalePrice_Pred'] = test_df['Cluster_Pred'].map(cluster_price_map)

    # Tính các chỉ số đánh giá
    mae = mean_absolute_error(y_test, test_df['SalePrice_Pred'])
    rmse = np.sqrt(mean_squared_error(y_test, test_df['SalePrice_Pred']))
    r2 = r2_score(y_test, test_df['SalePrice_Pred'])

    st.subheader("1. CÁC CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP TEST (`nhom4_test_data.xlsx`)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (Mean Absolute Error)", f"{mae:,.2f} USD")
    m2.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.2f} USD")
    m3.metric("R² Score", f"{r2:.4f}")

    st.subheader("2. BẢNG ĐỐI CHIẾU GIÁ THỰC TẾ VÀ GIÁ ƯỚC TÍNH (TEST DATA)")
    result_display = test_df[FEATURES + [TARGET, 'Cluster_Pred', 'SalePrice_Pred']].copy()
    result_display['SaiSố_TuyếtĐối'] = np.abs(result_display[TARGET] - result_display['SalePrice_Pred'])
    st.dataframe(result_display.style.format({
        TARGET: "{:,.0f}",
        'SalePrice_Pred': "{:,.2f}",
        'SaiSố_TuyếtĐối': "{:,.2f}"
    }))

    st.subheader("3. BIỂU ĐỒ SO SÁNH GIÁ THỰC TẾ VS GIÁ DỰ ĐOÁN")
    fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
    ax_comp.scatter(y_test, test_df['SalePrice_Pred'], alpha=0.6, color='darkorange', label='Mẫu dữ liệu Test')
    ax_comp.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Đường lý tưởng (Actual = Predicted)')
    ax_comp.set_xlabel("Giá nhà Thực tế (SalePrice Actual)")
    ax_comp.set_ylabel("Giá nhà Dự đoán (SalePrice Predicted)")
    ax_comp.set_title("So sánh Giá thực tế vs Giá ước tính từ K-Means")
    ax_comp.legend()
    st.pyplot(fig_comp)

# --- TAB 4: DỰ ĐOÁN NHÀ MỚI ---
with tab4:
    st.subheader("DỰ ĐOÁN GIÁ CHO CẤU HÌNH NHÀ ĐÃ NHẬP Ở SIDEBAR")
    input_data = pd.DataFrame([user_input])
    
    # Tính khoảng cách tới các centroid
    input_cluster = kmeans_model.predict(input_data)[0]
    predicted_price = cluster_price_map[input_cluster]

    st.write("##### Thông số nhập vào:")
    st.table(input_data)

    col_res1, col_res2 = st.columns(2)
    col_res1.success(f"🏡 **Cụm thuộc về:** Cụm {input_cluster}")
    col_res2.info(f"💵 **Giá nhà dự đoán:** {predicted_price:,.2f} USD")
