# -*- coding: utf-8 -*-
"""Dự báo điểm thi sinh viên - Code hoàn chỉnh với dataset thực tế"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Load và khám phá dữ liệu
print("=== Bước 1: Load và khám phá dữ liệu ===")
# Đọc dữ liệu từ file CSV (đảm bảo file cùng thư mục với script)
try:
    df = pd.read_csv('StudentsPerformance.csv')
    print("\nĐọc file CSV thành công!")
except FileNotFoundError:
    print("\nLỗi: Không tìm thấy file 'StudentsPerformance.csv'")
    print("Vui lòng kiểm tra lại đường dẫn hoặc đặt file cùng thư mục với script này")
    exit()

# Hiển thị thông tin cơ bản
print("\nThông tin dataset:")
print(df.info())
print("\n5 dòng đầu tiên:")
print(df.head())
print("\nThống kê mô tả:")
print(df.describe())

# 2. Tiền xử lý dữ liệu
print("\n=== Bước 2: Tiền xử lý dữ liệu ===")
# Kiểm tra dữ liệu thiếu
print("\nSố lượng giá trị thiếu trong từng cột:")
print(df.isnull().sum())

# Xử lý dữ liệu thiếu (nếu có)
if df.isnull().sum().sum() > 0:
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print("\nĐã xử lý dữ liệu thiếu")
else:
    print("\nKhông có dữ liệu thiếu")

# Chuẩn hóa tên cột (bỏ khoảng trắng)
df.columns = df.columns.str.replace(' ', '_').str.lower()

# Chuyển đổi biến phân loại thành số
label_encoders = {}
categorical_cols = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Phân tích và visualization
print("\n=== Bước 3: Phân tích và visualization ===")
plt.figure(figsize=(18, 12))

# Phân phối điểm số các môn
plt.subplot(2, 3, 1)
sns.kdeplot(data=df, x='math_score', label='Toán', fill=True)
sns.kdeplot(data=df, x='reading_score', label='Đọc', fill=True)
sns.kdeplot(data=df, x='writing_score', label='Viết', fill=True)
plt.title('Phân phối điểm các môn học')
plt.xlabel('Điểm số')
plt.legend()

# Tương quan giữa các điểm số
plt.subplot(2, 3, 2)
sns.scatterplot(data=df, x='math_score', y='writing_score', alpha=0.6)
plt.title('Tương quan điểm Toán và Viết')

# Ảnh hưởng của giới tính
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='gender', y='math_score')
plt.title('Điểm Toán theo giới tính')
plt.xticks([0, 1], ['Nữ', 'Nam'])

# Ảnh hưởng của bữa trưa
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='lunch', y='math_score')
plt.title('Điểm Toán theo loại bữa trưa')
plt.xticks([0, 1], ['Free/Reduced', 'Standard'])

# Ảnh hưởng của ôn tập
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='test_preparation_course', y='math_score')
plt.title('Điểm Toán theo ôn tập')
plt.xticks([0, 1], ['Chưa ôn', 'Đã ôn'])

# Ảnh hưởng của trình độ phụ huynh
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='parental_level_of_education', y='math_score')
plt.title('Điểm Toán theo trình độ phụ huynh')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Heatmap tương quan
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Ma trận tương quan giữa các biến')
plt.show()

# 4. Xây dựng mô hình dự đoán điểm Toán
print("\n=== Bước 4: Xây dựng mô hình dự đoán ===")
# Chọn features và target
X = df.drop(['math_score', 'reading_score', 'writing_score'], axis=1)
y = df['math_score']  # Có thể thay bằng reading_score hoặc writing_score

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': mae, 'R2 Score': r2})

    # Visualization kết quả dự đoán
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    plt.xlabel('Điểm thực tế')
    plt.ylabel('Điểm dự đoán')
    plt.title(f'Thực tế vs Dự đoán - {name}\nMAE: {mae:.2f}, R2: {r2:.2f}')
    plt.show()

# Hiển thị kết quả đánh giá
results_df = pd.DataFrame(results)
print("\nKết quả đánh giá mô hình:")
print(results_df)

# 5. Dự đoán điểm cho dữ liệu mới
print("\n=== Bước 5: Dự đoán điểm cho dữ liệu mới ===")
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Tạo dữ liệu mẫu để dự đoán (thay đổi giá trị tùy ý)
sample_data = {
    'gender': ['female'],  # Giới tính
    'race/ethnicity': ['group C'],  # Nhóm dân tộc
    'parental_level_of_education': ['some college'],  # Trình độ phụ huynh
    'lunch': ['standard'],  # Loại bữa trưa
    'test_preparation_course': ['none']  # Ôn tập
}

# Chuyển đổi dữ liệu mẫu thành DataFrame
sample_df = pd.DataFrame(sample_data)

# Áp dụng cùng LabelEncoder đã huấn luyện
for col in categorical_cols:
    sample_df[col] = label_encoders[col].transform(sample_df[col])

# Chuẩn hóa dữ liệu
sample_scaled = scaler.transform(sample_df)

# Dự đoán
predicted_score = best_model.predict(sample_scaled)
print(f"\nDự đoán điểm Toán cho sinh viên mẫu: {predicted_score[0]:.1f}")

# 6. Feature importance
print("\n=== Bước 6: Đánh giá độ quan trọng các đặc trưng ===")
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Độ quan trọng của các đặc trưng')
    plt.show()

    print("\nTop 5 đặc trưng quan trọng nhất:")
    print(feature_importance.head())
else:
    print("Mô hình không hỗ trợ đánh giá độ quan trọng đặc trưng.")

# 7. Phân tích điểm số theo các yếu tố
print("\n=== Bước 7: Phân tích sâu điểm số ===")
# Điểm trung bình theo các nhóm
print("\nĐiểm trung bình theo giới tính:")
print(df.groupby('gender')[['math_score', 'reading_score', 'writing_score']].mean())

print("\nĐiểm trung bình theo nhóm dân tộc:")
print(df.groupby('race/ethnicity')[['math_score', 'reading_score', 'writing_score']].mean())

# Biểu đồ điểm theo nhóm dân tộc
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='race/ethnicity', y='math_score')
plt.title('Phân bố điểm Toán theo nhóm dân tộc')
plt.show()

# Biểu đồ điểm theo trình độ phụ huynh
plt.figure(figsize=(12, 6))
order = df.groupby('parental_level_of_education')['math_score'].mean().sort_values().index
sns.boxplot(data=df, x='parental_level_of_education', y='math_score', order=order)
plt.title('Phân bố điểm Toán theo trình độ phụ huynh')
plt.xticks(rotation=45)
plt.show()

print("\n=== KẾT THÚC CHƯƠNG TRÌNH ===")