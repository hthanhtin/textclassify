import os
from sklearn.feature_extraction.text import CountVectorizer

from modelutils import BayesModel, modelUtils
model=BayesModel()
util=modelUtils()
path=r"D:\1.CA NHAN\pythonProject\AI_project\data\train"
def test1():
    stws=util.getStopword()
    Texts, Labels=util.getTexts_Labels(path)
    model.setData(Texts=Texts, Labels=Labels, TextPresent="Bow", ngram=1, stopwords=stws, testsize=0.3, kernel="GaussianNB")
    x, y=model.train_model()
    print (x, y)
    #vt=model.Str2Vector(r"Vô cách đây 1 tuần mà giờ mới review , được anh bạn ở gần quán dẫn vô ăn thử xem quán bán thế_nào , để đặt tiệc công_ty cuối năm , không_gian quán gồm 3 sảnh A , B , C_DÃY_PHÒNG_VIP 6 phòng , mới nhìn vào thì thấy rộng_rãi mát , hợp với ý_định của anh bạn , còn về món ăn_theo như mình chấm nếu 10 thì cũng đạt 8 , nhiều món lạ mà khu_vực gần đây không có , đặc_trưng quán , Vịt xông khói Đài_Loan gì đó cũng lạ_miệng và ngon , nhìn menu thì nhiều món quá mình chỉ thử được vài món nhưng đều đạt , có món bạch tuột sống nhúng giấm , trời_ơi nhân_viên bắt vào nó bò lúc_nhúc nhưng ăn ngon thật rất tươi và ngọt . Các bạn thử tới xem sao : ")
    #print (model.NBmodel.predict(vt))
    #model.save_model()
def test2():
    import numpy as np
    stws = util.getStopword()
    model.setData(Texts=None, Labels=None, TextPresent="Bow", ngram=1, stopwords=stws, testsize=0.3, kernel="MultinomialNB")
    model.loadModel()
    str=r'''Ăn_ở quán được khoảng 4 5 lần gì đó , quá tệ'''
    vt = model.vectorizer.transform([str])
    vt = np.array(vt.toarray())
    print (model.NBmodel.predict(vt))
#==================================================
# Create GUI
import streamlit as st
st.markdown("""
        <div style="text-align: center;">
            <h2>BÀI TẬP CUỐI KHOÁ HỌC AI</h2>
            <h3>HV: NGUYỄN VĂN THANH-BRVT</h3>
            <h2>PHÂN LOẠI SẮC THÁI VĂN BẢN</h2>
        </div>
    """, unsafe_allow_html=True)

def Train_page():
    st.header("Huấn luyện Model phân loại sắc thái văn bản")
    grid = st.columns(3)
    with grid[0]:
        cbx_presentText = st.selectbox("**Chọn mô hình biểu diễn văn bản**", ["Bow", "TF-IDF"])
    with grid[1]:
        cbx_gram = st.selectbox("**Chọn kiểu tách từ**", ["1_gram", "2_gram"])
    with grid[2]:
        cbx_kernel = st.selectbox("**Chọn hàm nhân (kernel) Naive Bayes**",
                                  ["MultinomialNB", "GaussianNB", "BernoulliNB"])

    btn_train = st.button("**Huấn luyện model**")
    import time
    mu = modelUtils()
    model = BayesModel()
    
    if btn_train:
        with st.spinner("Đang huấn luyện Model"):
            ct_doc = st.empty()
            ct_hl=st.empty()
            t1 = time.time()
            ct_doc.write("Đang đọc và xử lý dữ liệu")
            Path = os.path.join(os.getcwd(), "data", "train")
            Texts, Labels = mu.getTexts_Labels(Path)
            ct_doc.empty()
            ct_doc.write("Đọc và xử lý dữ liệu trong thời gian: " + str(time.time() - t1)+" giây")
            t2 = time.time()
            ct_hl.write("Đang huấn luyện mô hình...")
            model.setData(Texts=Texts, Labels=Labels, TextPresent=cbx_presentText,ngram=int(cbx_gram.split("_")[0]), kernel=cbx_kernel)
            x,y=model.train_model()
            ct_hl.write("Huấn luyện thành công!")
            st.markdown("**Huấn luyện hoàn tất trong thời gian:" + str(time.time() - t2) + " giây**")
            st.markdown("**Tổng thời gian xử lý và huấn luyện:" + str(time.time() - t1) + " giây**")
            st.markdown(f"**Độ chính xác trên tập huấn luyện: {str(x)}**")
            st.markdown(f"**Độ chính xác trên tập kiểm tra: {str(y)}**")

def PreFix_page():
    m = modelUtils()
    st.header("Tiền xử lý văn bản")
    upfile = st.file_uploader("Chọn file chứa nội dung cần tiền xử lý")
    # Kiểm tra xem có tệp được tải lên hay không
    if upfile is not None:
        # Đọc nội dung của tệp với mã hóa UTF-8
        file_contents = upfile.read().decode('utf-8')
        # Gán nội dung cho root_txt

        root_txt = st.text_area(label='**Văn bản gốc**', value=file_contents)
        fix_str = m.fixstring(root_txt)
        fix_txt = st.text_area(label="**Văn bản sau tiền xử lý (chuyển sang chữ thường, loại bỏ emoji, stopword)**",
                               value=fix_str)
        vectorizer = CountVectorizer(ngram_range=(1, 1))
        X = vectorizer.fit_transform([fix_txt])
        feature_names = vectorizer.get_feature_names_out()
        txt_1gram = st.text_area(label="**Văn bản tách từ-1gram**", value=" | ".join(feature_names))

        vectorizer = CountVectorizer(ngram_range=(2, 2))
        X = vectorizer.fit_transform([fix_txt])
        feature_names = vectorizer.get_feature_names_out()
        txt_2gram = st.text_area(label="**Văn bản tách từ-2gram**", value=" | ".join(feature_names))

def Predict_Page():
    global model, util
    import glob
    Path=os.path.join(os.getcwd(),"models")+"/*"
    X=[os.path.basename(f) for f in glob.glob(Path) if not f.__contains__("vector")]
    st.header("**Phân lớp văn bản**")
    models_name=st.selectbox("**1.Chọn model đã huấn luyện**",options=X)
    predictfile = st.file_uploader("**2.Chọn file chứa nội dung cần phân lớp**", key="predict")
    # Kiểm tra xem có tệp được tải lên hay không
    if predictfile is not None:
        # Đọc nội dung của tệp với mã hóa UTF-8
        file_contents = predictfile.read().decode('utf-8')
        text_predict=st.text_area("**Văn bản cần phân lớp:**", value=file_contents)
        fix_text=util.fixstring(text_predict)
        #load models
        modelPath=os.path.join(os.getcwd(),"models",models_name)
        x=models_name.split(".")
        stw=util.getStopword()
        model.setData(Texts=None, Labels=None,TextPresent=x[0], ngram=int(x[1]),stopwords=stw,testsize=0.3, kernel=x[2])
        model.loadModel()
        c=model.Text2Class(fix_text)
        print(c[0])
        if c[0]==1:
            st.success("**Văn bản thuộc lớp Tích cực**")
        else:
            st.error("**Văn bản xếp vào lớp Tiêu cực**")




def Gui():
    t1, t2, t3=st.tabs(["**1.Tiền xử lý văn bản**","**2.Huấn luyện mô hình**","**3.Phân loại văn bản**"])
    with t1:
        PreFix_page()
    with t2:
        Train_page()
    with t3:
        Predict_Page()
Gui()