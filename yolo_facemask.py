import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO(r"D:\UIT\HK3\XU_LI_ANH\facemask\best.pt")

tab1, tab2 = st.tabs(["Camera", "Chọn ảnh"])

with tab1:
    enable = st.checkbox('Cho phép camera')
    picture = st.camera_input("Take a picture", disabled=not enable)
    if picture:
        image = Image.open(picture)
        results = model.predict(source=image, conf=0.25)
        res_plotted = results[0].plot()
        st.image(image, caption="Ảnh đầu vào")
        st.image(res_plotted, caption="Kết quả dự đoán", channels="BGR")

with tab2:
    upload_im = st.file_uploader("Chọn ảnh của bạn", type=["png", "jpg", "jpeg"])

    if upload_im:
        image = Image.open(upload_im)
        results = model.predict(source=image, conf=0.25)
        res_plotted = results[0].plot()
        st.image(image, caption="Ảnh đầu vào")
        st.image(res_plotted, caption="Kết quả dự đoán", channels="BGR")