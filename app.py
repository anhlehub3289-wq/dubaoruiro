import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# LỆNH STREAMLIT ĐẦU TIÊN
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện gian lận tại AGRIBANK",
    page_icon="❤️"
)

# ==============================================================================
# HÀM NẠP DỮ LIỆU DÙNG CHUNG (CACHE)
# ==============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes và cache lại để tối ưu hiệu năng.
    Hỗ trợ cả file CSV và Excel.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None, "Định dạng file không được hỗ trợ. Vui lòng tải lên file CSV hoặc Excel."
        
        if df.empty:
            return None, "File dữ liệu trống."
        return df, None
    except Exception as e:
        return None, f"Lỗi khi đọc file: {str(e)}"

# ==============================================================================
# SIDEBAR — VÙNG CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện", 
        type=["csv", "xlsx", "xls"],
        help="Chọn tệp chứa dữ liệu lịch sử giao dịch (ví dụ: dataset1.csv) bao gồm các biến X_1 đến X_14 và biến mục tiêu 'default'."
    )
    
    st.divider()
    
    # 2. Lựa chọn mô hình (Notebook sử dụng 3 mô hình)
    st.subheader("🤖 Cấu hình Mô hình")
    model_choice = st.selectbox(
        "Chọn thuật toán phân loại",
        options=["Logistic Regression (Model 1)", "Decision Tree (Model 2)", "Random Forest (Model 3)"],
        index=2, # Mặc định chọn Random Forest tương tự như phần cuối notebook tập trung tối ưu
        help="Lựa chọn một trong ba thuật toán đã được thử nghiệm trong notebook huấn luyện."
    )
    
    # 3. Tham số mô hình động theo thuật toán được chọn
    st.markdown("#### Siêu tham số mặc định")
    
    # Gom các tham số cơ bản và nâng cao một cách hợp lý
    if model_choice == "Logistic Regression (Model 1)":
        max_iter = st.slider("max_iter", min_value=100, max_value=2000, value=1000, step=100, help="Số lượng vòng lặp tối đa cho thuật toán tối ưu.")
        solver = st.selectbox("solver", options=["lbfgs", "liblinear", "saga"], index=0, help="Thuật toán tối ưu hóa bài toán.")
        random_state = st.number_input("random_state", value=42, step=1, help="Giá trị hạt giống để tái hiện kết quả.")
        model_params = {"max_iter": max_iter, "solver": solver, "random_state": random_state}
        
    elif model_choice == "Decision Tree (Model 2)":
        criterion = st.selectbox("criterion", options=["gini", "entropy", "log_loss"], index=0, help="Tiêu chí đo lường chất lượng phân tách nhánh.")
        max_depth = st.slider("max_depth (None nếu không chọn)", min_value=1, max_value=50, value=10, help="Độ sâu tối đa của cây quyết định.")
        use_max_depth = st.checkbox("Giới hạn max_depth", value=False, help="Bỏ chọn nếu muốn cây phát triển tối đa.")
        random_state = st.number_input("random_state", value=42, step=1, help="Giá trị hạt giống để tái hiện kết quả.")
        
        model_params = {
            "criterion": criterion, 
            "max_depth": max_depth if use_max_depth else None, 
            "random_state": random_state
        }
        
    else: # Random Forest (Model 3)
        n_estimators = st.slider("n_estimators", min_value=10, max_value=500, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        criterion = st.selectbox("criterion", options=["gini", "entropy"], index=0, help="Tiêu chí đo lường phân tách.")
        random_state = st.number_input("random_state", value=42, step=1, help="Giá trị hạt giống để tái hiện kết quả.")
        
        with st.expander("Tham số nâng cao"):
            max_depth_rf = st.slider("max_depth (RF)", min_value=1, max_value=50, value=15)
            use_max_depth_rf = st.checkbox("Giới hạn max_depth (RF)", value=False)
            min_samples_split = st.slider("min_samples_split", min_value=2, max_value=10, value=2)
            
        model_params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "random_state": random_state,
            "max_depth": max_depth_rf if use_max_depth_rf else None,
            "min_samples_split": min_samples_split
        }
        
    st.divider()
    
    # Tỷ lệ chia tập Train/Test dữ liệu
    test_size = st.slider("Tỷ lệ tập kiểm định (Test Size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Tỷ lệ lượng dữ liệu dùng để chấm điểm mô hình.")
    
    # 4. NÚT HÀNH ĐỘNG DUY NHẤT ĐỂ HUẤN LUYỆN
    train_clicked = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

# ==============================================================================
# HEADER — VÙNG ĐỊNH HƯỚNG
# ==============================================================================
st.title("😎 Hệ thống Học máy Phát hiện giao dịch gian lận tại AGRIBANK 😎")
st.caption("Ứng dụng hỗ trợ phân tích rủi ro tài chính và phân loại giao dịch gian lận tự động dựa trên các chỉ số hoạt động.")

# Biến lưu trữ dữ liệu chính toàn app
df_main = None

if uploaded_file is None:
    st.info("👋 Chào mừng bạn! Vui lòng tải lên tệp dữ liệu huấn luyện mẫu (ví dụ: `dataset1.csv`) từ Sidebar để kích hoạt hệ thống.")
    st.stop()
else:
    # Đọc dữ liệu thông qua hàm dùng chung
    file_bytes = uploaded_file.getvalue()
    df_main, error_msg = load_data(file_bytes, uploaded_file.name)
    
    if error_msg:
        st.error(error_msg)
        st.stop()
        
    st.caption(f"📁 Đang dùng tệp dữ liệu: `{uploaded_file.name}` | Tổng số dòng: **{df_main.shape[0]}** | Tổng số cột: **{df_main.shape[1]}**")

st.divider()

# Định nghĩa các biến đặc trưng và biến mục tiêu rút ra từ dữ liệu/notebook
features = [f"X_{i}" for i in range(1, 15)] # X_1 đến X_14
target = "default"

# Kiểm tra tính hợp lệ của Schema dữ liệu đầu vào
missing_cols = [col for col in features + [target] if col not in df_main.columns]
if missing_cols:
    st.error(f"❌ Tệp dữ liệu không đúng cấu trúc yêu cầu của Notebook. Thiếu các cột: {missing_cols}")
    st.stop()

# ==============================================================================
# KHỐI HUẤN LUYỆN (Chạy khi nhấn nút và lưu kết quả vào session_state)
# ==============================================================================
if train_clicked:
    with st.spinner("🔄 Đang phân tách dữ liệu và huấn luyện mô hình..."):
        # 1. Tách biến
        X = df_main[features]
        y = df_main[target]
        
        # 2. Phân chia dữ liệu Train/Test giống hệt logic kiểm định của notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(model_params.get("random_state", 42))
        )
        
        # 3. Khởi tạo thuật toán tương ứng dựa trên cấu hình sidebar
        if "Logistic Regression" in model_choice:
            model = LogisticRegression(
                max_iter=model_params["max_iter"], 
                solver=model_params["solver"], 
                random_state=model_params["random_state"]
            )
        elif "Decision Tree" in model_choice:
            model = DecisionTreeClassifier(
                criterion=model_params["criterion"],
                max_depth=model_params["max_depth"],
                random_state=model_params["random_state"]
            )
        else:
            model = RandomForestClassifier(
                n_estimators=model_params["n_estimators"],
                criterion=model_params["criterion"],
                random_state=model_params["random_state"],
                max_depth=model_params["max_depth"],
                min_samples_split=model_params["min_samples_split"]
            )
            
        # 4. Huấn luyện mô hình
        model.fit(X_train, y_train)
        
        # 5. Dự đoán và tính toán kết quả kiểm định
        y_pred = model.predict(X_test)
        
        # Thử nghiệm lấy xác suất nếu mô hình hỗ trợ để hiển thị chi tiết rủi ro
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_prob = None
            
        # Lưu trữ 3 thành phần cốt lõi vào session_state để dùng chung cho các tab
        st.session_state["trained_model"] = model
        st.session_state["model_name"] = model_choice
        st.session_state["evaluation_metrics"] = {
            "y_test": y_test.values,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred)
        }
        st.success(f"🎉 Đã huấn luyện thành công mô hình `{model_choice}`!")

# ==============================================================================
# ĐỊNH NGHĨA KHỐI TAB CHỨA NỘI DUNG CHÍNH (TP3 -> TP6)
# ==============================================================================
tab_overview, tab_viz, tab_train, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🔬 Kết quả kiểm định mô hình", 
    "🔮 Sử dụng mô hình dự báo"
])

# ------------------------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------------------------------------------
with tab_overview:
    st.subheader("📋 Thống kê sơ bộ tập dữ liệu")
    
    # 1. Kích thước dữ liệu thông qua st.metric
    col_size1, col_size2, col_size3 = st.columns(3)
    with col_size1:
        st.metric(label="Số dòng (Giao dịch)", value=f"{df_main.shape[0]:,}")
    with col_size2:
        st.metric(label="Số cột đặc trưng", value=df_main.shape[1])
    with col_size3:
        # Tính toán dung lượng file xấp xỉ dựa trên bộ nhớ dataframe
        df_mem = df_main.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric(label="Dung lượng bộ nhớ xấp xỉ", value=f"{df_mem:.2f} MB")
        
    st.markdown("---")
    
    # 2. Xem dữ liệu thô dạng Head
    st.subheader("🔍 Bản xem trước dữ liệu (5 dòng đầu tiên)")
    st.dataframe(df_main.head(5), use_container_width=True)
    
    st.markdown("---")
    
    # 3. Thống kê mô tả - CHỈ mô tả các biến đưa vào mô hình (X_1 đến X_14 và default)
    st.subheader("📐 Chỉ số mô tả các biến đặc trưng")
    selected_cols = features + [target]
    st.dataframe(df_main[selected_cols].describe().T, use_container_width=True)

# ------------------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------------------------------------------
with tab_viz:
    st.subheader("📊 Phân tích đặc trưng bằng đồ thị trực quan")
    
    # Đồ thị phân phối của biến mục tiêu y (Phải được ưu tiên hàng đầu)
    st.markdown("#### 1. Phân phối của biến mục tiêu Nhãn gian lận (`default`)")
    target_counts = df_main[target].value_counts().reset_index()
    target_counts.columns = ['Trạng thái', 'Số lượng']
    target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: 'Hợp lệ (0)', 1: 'Gian lận/Rủi ro (1)'})
    
    fig_target = px.bar(
        target_counts, x='Trạng thái', y='Số lượng',
        color='Trạng thái',
        color_discrete_map={'Hợp lệ (0)': '#2ecc71', 'Gian lận/Rủi ro (1)': '#e74c3c'},
        text_auto=True,
        title="Tỷ lệ phân lớp dữ liệu giao dịch"
    )
    fig_target.update_layout(height=350)
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("---")
    
    # Vẽ các biến đầu vào theo cơ chế lưới 2x2 hoặc tùy chọn bộ lọc nếu quá nhiều
    st.markdown("#### 2. Phân tích phân phối các chỉ số đặc trưng đầu vào (X)")
    
    # Cho phép người dùng lọc nhanh tối đa 4 biến để hiển thị lưới 2x2 cân đối
    default_features = ["X_1", "X_2", "X_5", "X_13"] # Lựa chọn ngẫu nhiên một số cột đặc sắc
    selected_viz_features = st.multiselect(
        "Chọn các biến đặc trưng muốn hiển thị (Tối đa nên chọn 4 biến để giữ giao diện lưới 2x2 cân đối)",
        options=features,
        default=[f for f in default_features if f in features]
    )
    
    if len(selected_viz_features) > 0:
        # Thiết lập ma trận hiển thị 2 cột
        cols_viz = st.columns(2)
        for idx, col_name in enumerate(selected_viz_features[:4]): # Giới hạn 4 đồ thị đầu tiên để tránh tràn trang
            current_col = cols_viz[idx % 2]
            with current_col:
                # Dựa vào dtype để sinh loại đồ thị, vì tất cả là float64 nên dùng Histogram kết hợp Box plot xem ngoại lai
                fig_feat = px.histogram(
                    df_main, x=col_name, color=target,
                    marginal="box", # hiển thị boxplot nhỏ phía trên để xem điểm ngoại lai ngoại lai
                    barmode="overlay",
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                    title=f"Phân phối tần suất của biến {col_name} phân loại theo nhãn rủi ro",
                    labels={target: "Nhãn rủi ro"}
                )
                fig_feat.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("⚠️ Vui lòng lựa chọn ít nhất một biến đặc trưng từ hộp danh sách trên để xem biểu đồ phân phối.")

# ------------------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------------------------------------------
with tab_train:
    st.subheader("🔬 Đánh giá độ chính xác và Ma trận nhầm lẫn")
    
    # Kiểm tra xem mô hình đã được chạy huấn luyện chưa
    if "evaluation_metrics" not in st.session_state:
        st.info("💡 Chưa có mô hình nào được huấn luyện. Vui lòng chuyển cấu hình tại bảng Sidebar và bấm nút **🚀 Huấn luyện Mô hình**.")
    else:
        metrics = st.session_state["evaluation_metrics"]
        model_name_trained = st.session_state["model_name"]
        
        # In đậm và tô chữ hiển thị màu xanh dương cho dòng trạng thái mô hình
        st.markdown(f"<span style='color: #1f77b4; font-weight: bold;'>Đang hiển thị kết quả kiểm tra của mô hình vừa huấn luyện: {model_name_trained}</span>", unsafe_allow_html=True)
        
        # 1. Trình bày các chỉ tiêu vô hướng qua st.metric
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="Độ chính xác toàn cục (Accuracy)", value=f"{metrics['accuracy']:.4f}")
        with col_m2:
            st.metric(label="Độ chính xác lớp rủi ro (Precision)", value=f"{metrics['precision']:.4f}")
        with col_m3:
            st.metric(label="Tỷ lệ tìm sót rủi ro (Recall)", value=f"{metrics['recall']:.4f}")
        with col_m4:
            st.metric(label="Chỉ số F1-Score", value=f"{metrics['f1']:.4f}")
            
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns([1, 1])
        
        # 2. Vẽ Ma trận nhầm lẫn (Confusion Matrix) dùng Plotly Heatmap trực quan sinh động
        with col_chart1:
            st.markdown("#### Ma trận nhầm lẫn (Confusion Matrix)")
            cm = metrics["cm"]
            
            # Cấu trúc văn bản nhãn hiển thị trong ô của ma trận nhầm lẫn
            z_text = [[str(y) for y in x] for x in cm]
            
            # Chỉ sử dụng fig_cm để tối ưu hóa
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Dự đoán Hợp lệ (0)', 'Dự đoán Gian lận (1)'],
                y=['Thực tế Hợp lệ (0)', 'Thực tế Gian lận (1)'],
                colorscale='Blues',
                text=z_text,
                texttemplate="%{text}",
                showscale=False
            ))
            fig_cm.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        # 3. Hiển thị Báo cáo phân loại văn bản chuẩn xác dạng DataFrame bảng biểu
        with col_chart2:
            st.markdown("#### Chi tiết báo cáo phân loại (Classification Report)")
            
            # Tái tạo lại classification report dạng dict để đưa vào dataframe trực quan sạch đẹp
            report_dict = classification_report(
                metrics["y_test"], metrics["y_pred"], 
                target_names=["Hợp lệ (0)", "Gian lận (1)"], 
                output_dict=True
            )
            df_report = pd.DataFrame(report_dict).transpose()

            # Định dạng chữ in đậm và màu xanh dương cho các ô dữ liệu
            styled_report = df_report.style.format(precision=4).map(
                lambda v: 'color: #004085; font-weight: bold;'
            )
            st.dataframe(styled_report, use_container_width=True)

# ==============================================================================
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ==============================================================================
with tab_inference:
    st.subheader("🔮 Chẩn đoán & Chấm điểm giao dịch thời gian thực")
    
    if "trained_model" not in st.session_state:
        st.info("💡 Bạn cần huấn luyện mô hình thành công bên Sidebar trước khi thực hiện chức năng dự báo rủi ro này.")
    else:
        model = st.session_state["trained_model"]
        
        # Chọn chế độ nhập liệu bằng st.radio ở đầu tab
        inference_mode = st.radio(
            "Chọn phương thức nạp dữ liệu cần dự đoán:",
            options=["Chế độ 1 — Nhập thông số trực tiếp của 1 khách hàng", "Chế độ 2 — Tải tệp danh sách hàng loạt (Cấu trúc X_test)"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP
        # ----------------------------------------------------------------------
        if "Chế độ 1" in inference_mode:
            st.markdown("##### Điền các thông số kỹ thuật của giao dịch cần chấm điểm:")
            
            # Tạo form bao quanh các widget nhập liệu
            with st.form("single_customer_inference_form"):
                
                # Sắp xếp các cột nhập liệu gọn gàng thành lưới 4 cột
                grid_cols = st.columns(4)
                input_data = {}
                
                # Đọc min, max, median từ tập dữ liệu gốc để thiết lập cấu hình thông minh cho widget
                for idx, col_name in enumerate(features):
                    target_col = grid_cols[idx % 4]
                    with target_col:
                        min_val = float(df_main[col_name].min())
                        max_val = float(df_main[col_name].max())
                        median_val = float(df_main[col_name].median())
                        
                        input_data[col_name] = st.number_input(
                            label=f"Chỉ số {col_name}",
                            min_value=min_val * 5.0 if min_val < 0 else 0.0, # Nới rộng biên nhập liệu
                            max_value=max_val * 5.0,
                            value=median_val, # Mặc định lấy giá trị trung vị theo dữ liệu mẫu như yêu cầu đặc tả
                            format="%.6f",
                            help=f"Khoảng dữ liệu mẫu: [{min_val:.4f} đến {max_val:.4f}]"
                        )
                
                submit_predict = st.form_submit_button("🔍 Tiến hành Phân tích rủi ro", type="primary")
                
            if submit_predict:
                # Chuyển dữ liệu vào DataFrame để đảm bảo giữ nguyên tên cột (Feature names) tránh cảnh báo của Scikit-Learn
                df_single = pd.DataFrame([input_data])
                
                prediction = model.predict(df_single)[0]
                
                try:
                    prob = model.predict_proba(df_single)[0][1]
                except:
                    prob = None
                    
                # THAY ĐỔI THEO YÊU CẦU: Đổi màu chữ tiêu đề trong hình sang ĐỎ và IN ĐẬM
                st.markdown("<h4 style='color: #ff0000; font-weight: bold;'>🟥 KẾT QUẢ PHÂN TÍCH CHẨN ĐOÁN:</h4>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.error("🚨 **CẢNH BÁO: Giao dịch này có dấu hiệu GIAN LẬN / RỦI RO CAO!**")
                else:
                    st.success("🟢 **AN TOÀN: Giao dịch được đánh giá HỢP LỆ.**")
                    
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric(label="Mã nhãn phân loại", value=int(prediction))
                with col_res2:
                    if prob is not None:
                        st.metric(label="Xác suất/Điểm số rủi ro gian lận", value=f"{prob * 100:.2f}%")
                    else:
                        st.metric(label="Xác suất rủi ro", value="N/A (Thuật toán không hỗ trợ)")

        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 2 — TẢI FILE HÀNG LOẠT
        # ----------------------------------------------------------------------
        else:
            st.markdown("##### Tải lên tệp dữ liệu danh sách giao dịch mới cần kiểm tra:")
            st.caption("⚠️ Yêu cầu: Tệp phải chứa đầy đủ cấu trúc 14 cột đặc trưng từ `X_1` đến `X_14` giống như định dạng huấn luyện.")
            
            bulk_file = st.file_uploader(
                "Chọn tệp danh sách giao dịch mới", 
                type=["csv", "xlsx", "xls"], 
                key="bulk_uploader"
            )
            
            if bulk_file is not None:
                df_bulk, err_bulk = load_data(bulk_file.getvalue(), bulk_file.name)
                
                if err_bulk:
                    st.error(err_bulk)
                else:
                    # Kiểm tra sự tương thích của schema
                    missing_bulk_cols = [c for c in features if c not in df_bulk.columns]
                    
                    if missing_bulk_cols:
                        st.error(f"❌ Tệp tải lên không hợp lệ. Thiếu các cột đặc trưng bắt buộc sau: {missing_bulk_cols}")
                    else:
                        # Chỉ lọc lấy đúng 14 cột đưa vào mô hình chấm điểm theo đúng thứ tự lúc train
                        X_bulk = df_bulk[features]
                        
                        # Dự báo hàng loạt
                        bulk_predictions = model.predict(X_bulk)
                        
                        # Gắn kết quả dự báo trực tiếp vào dataframe gốc hiển thị cho người dùng
                        df_result = df_bulk.copy()
                        df_result["Du_Bao_Gian_Lan"] = bulk_predictions
                        
                        try:
                            bulk_probs = model.predict_proba(X_bulk)[:, 1]
                            df_result["Xac_Suat_Rui_Ro"] = bulk_probs
                        except:
                            pass
                            
                        st.success(f"🎉 Đã thực hiện chấm điểm thành công cho toàn bộ {df_bulk.shape[0]} dòng giao dịch!")
                        
                        # Thống kê nhanh kết quả vừa dự báo
                        fraud_count = int((bulk_predictions == 1).sum())
                        st.warning(f"📊 Phát hiện **{fraud_count}** giao dịch tiềm ẩn rủi ro gian lận trên tổng số **{df_bulk.shape[0]}** giao dịch vừa nạp.")
                        
                        # Hiển thị bảng kết quả trong một khung cuộn cố định chiều cao
                        st.markdown("👉 Bảng chi tiết kết quả dự báo:")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Tạo nút download xuất dữ liệu kết quả kèm mã UTF-8-sig chống lỗi font
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                        csv_output = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả dạng CSV",
                            data=csv_output,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
