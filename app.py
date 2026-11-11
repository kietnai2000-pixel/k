import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import cdist


# --- 1. ĐỌC DỮ LIỆU GỐC --- 
train_path = '/content/nhom4_train_data.xlsx'
test_path = '/content/nhom4_test_data.xlsx'

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'HouseAge']
target = 'SalePrice'

# Loại bỏ khuyết thiếu để đảm bảo độ chính xác thuật toán
df_train_clean = df_train.dropna(subset=features + [target]).copy()
df_test_clean = df_test.dropna(subset=features + [target]).copy()

# --- 2. CHUẨN HÓA DỮ LIỆU --- 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train_clean[features])
X_test_scaled = scaler.transform(df_test_clean[features])

# --- 3. TÍNH WCSS CHO LẦN LƯỢT CÁC K & VẼ BIỂU ĐỒ ELBOW --- 
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Biểu đồ Elbow (Xác định điểm gãy K tối ưu)')
plt.xlabel('Số lượng cụm K')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# --- 4. KHỞI TẠO 3 CỤM & TÍNH KHOẢNG CÁCH CENTROID --- 
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X_train_scaled)
centroids = kmeans_model.cluster_centers_

# Tính khoảng cách đến centroid của mỗi cụm
distances = cdist(X_train_scaled, centroids, metric='euclidean')

# Gán mỗi nhà vào cụm có khoảng cách gần nhất
train_assigned_clusters = np.argmin(distances, axis=1)
df_train_clean['Cluster'] = train_assigned_clusters

# Tính lại trung bình từng cụm để thiết lập centroid mới
new_centroids = []
for i in range(3):
    cluster_points = X_train_scaled[train_assigned_clusters == i]
    if len(cluster_points) > 0:
        new_centroids.append(cluster_points.mean(axis=0))
    else:
        new_centroids.append(centroids[i])
new_centroids = np.array(new_centroids)

# --- 5. HIỂN THỊ TÂM CENTROID VÀ ĐẶC ĐIỂM TRUNG BÌNH --- 
centroids_original_space = scaler.inverse_transform(new_centroids)
df_centroids_summary = pd.DataFrame(centroids_original_space, columns=features)

# Giá trị đại diện cho mỗi cụm (Trung bình SalePrice của cụm đó ở tập Train)
cluster_prices = df_train_clean.groupby('Cluster')[target].mean().to_dict()
df_centroids_summary['Mean_SalePrice'] = df_centroids_summary.index.map(cluster_prices)

print("\n=== BẢNG THỐNG KÊ TÂM CỤM (CENTROIDS) & ĐẶC TRƯNG TRUNG BÌNH ===")
display(df_centroids_summary)

print("\n=== GIẢI THÍCH Ý NGHĨA CÁC CỤM CHÍNH KHÁI QUÁT ===")
for idx, row in df_centroids_summary.iterrows():
    print(f"- Cụm {idx}: Chất lượng trung bình {row['OverallQual']:.1f}, diện tích {row['GrLivArea']:.1f} sqft, giá nhà trung bình: ${row['Mean_SalePrice']:,.2f}.")

# --- 6. DỰ ĐOÁN VÀ ĐỐI CHIẾU KẾT QUẢ TRÊN TẬP TEST --- 
test_distances = cdist(X_test_scaled, new_centroids, metric='euclidean')
test_pred_clusters = np.argmin(test_distances, axis=1)

df_results = df_test_clean.copy()
df_results['Assigned_Cluster'] = test_pred_clusters
df_results['Predicted_SalePrice'] = df_results['Assigned_Cluster'].map(cluster_prices)
df_results['Error_Absolute'] = (df_results[target] - df_results['Predicted_SalePrice']).abs()

print("\n=== BẢNG KẾT QUẢ ĐỐI CHIẾU GIÁ THỰC TẾ VÀ GIÁ ƯỚC TÍNH TRÊN TẬP TEST (15 dòng tiêu biểu) ===")
display(df_results[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'HouseAge', target, 'Predicted_SalePrice', 'Error_Absolute']].head(15))

# --- 7. ĐÁNH GIÁ MÔ HÌNH --- 
y_true = df_results[target].values
y_pred = df_results['Predicted_SalePrice'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n=== CÁC CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP TEST ===")
print(f"MAE (Mean Absolute Error): ${mae:,.2f}")
print(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"R2 Score (Hệ số xác định): {r2:.4f}")

# --- 8. VẼ CÁC BIỂU ĐỒ TRỰC QUAN HÓA --- 
fig, axes = plt.subplots(1, 3, figsize=(22, 5))

# Biểu đồ 1: Phân bố giá nhà
sns.histplot(df_train_clean[target], kde=True, color='blue', alpha=0.4, ax=axes[0], label='Train Price')
sns.histplot(df_test_clean[target], kde=True, color='orange', alpha=0.4, ax=axes[0], label='Test Price')
axes[0].set_title('Biểu đồ phân bố giá nhà (SalePrice)')
axes[0].legend()

# Biểu đồ 2: Phân cụm nhà và tâm cụm
sns.scatterplot(data=df_train_clean, x='GrLivArea', y='OverallQual', hue='Cluster', palette='viridis', alpha=0.7, ax=axes[1])
axes[1].scatter(df_centroids_summary['GrLivArea'], df_centroids_summary['OverallQual'], s=250, c='red', marker='X', label='Centroids')
axes[1].set_title('Biểu đồ phân cụm (Diện tích & Chất lượng)')
axes[1].legend()

# Biểu đồ 3: So sánh giá thực tế vs giá dự đoán
sns.scatterplot(data=df_results, x=target, y='Predicted_SalePrice', hue='Assigned_Cluster', palette='Set1', s=80, alpha=0.8, ax=axes[2])
line_range = np.linspace(min(y_true), max(y_true), 100)
axes[2].plot(line_range, line_range, color='red', linestyle='--', label='Đường lý tưởng')
axes[2].set_xlabel('Giá thực tế ($)')
axes[2].set_ylabel('Giá dự đoán ($)')
axes[2].set_title('So sánh Giá nhà thực tế vs Ước tính')
axes[2].legend()

plt.tight_layout()
plt.show()
