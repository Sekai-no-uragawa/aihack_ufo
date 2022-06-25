# import json
# from io import BytesIO
# from PIL import Image
# import os

from pyrsistent import s
import streamlit as st
from streamlit_lottie import st_lottie
import json
from PIL import Image
from io import BytesIO, StringIO
import base64
import numpy as np
import torch
import urllib



# def upload_image_ui():
#     uploaded_image = st.file_uploader("Please choose an image file", type=["png", "jpg", "jpeg"])
#     if uploaded_image is not None:
#         try:
#             image = Image.open(uploaded_image)
#         except Exception:
#             st.error("Error: Invalid image")
#         else:
#             img_array = np.array(image)
#             return img_array


@st.experimental_memo
def load_weights():
    url = 'https://github.com/Sekai-no-uragawa/aihack_ufo/releases/download/weights/one_class_weights.pt'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    return filename

@st.experimental_memo
def load_model(weights_file):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file)
    return model

def main_page():
    file_types = ["png", "jpg", "jpeg"]

    c10, c20, c30 = st.columns([1, 6, 1])
    with c20:
        st.title("Здоровые зубы - ***круто***")
        st.subheader('Поиск кариеса по фотографии с помощью ИИ')
        st.write('''
        Данный сервис позволяет по загруженной вами фотографии зубов обнаружить возможный кариес и получить рекомендации для дальнейшего обследования у специалиста.\n
        **Внимание!** Сервис носит рекомендательный характер и позволяет лишь определить возможные внешние признаки кариеса, независимо от результатов, 
        человек должен проходить регулярные консультации у специалиста!\n
        \n
        _Пример фото для загрузки_:
        ''')
    

    _, c1, c2, _= st.columns([1, 3, 3, 1])
    
    with c1:
        example1 = Image.open("data/teeth_example4.jpg")
        st.image(example1)
    with c2:
        example2 = Image.open('data/teeth_example3.jpg')
        st.image(example2)

    

    c10, c20, c30 = st.columns([1, 6, 1])
    with c20:
        st.write('''
        ___
        Загрузите Ваше фото ниже:
        ''')

    _, c1, _= st.columns([1, 4, 1])
    with c1:
        file = st.file_uploader("Upload file", type=file_types)
        show_file = st.empty()
        if not file:
            show_file.info("Загрузите фото в формате: " + ", ".join(file_types))
        #content = file.getvalue()
    with c1:
        if isinstance(file, BytesIO):
            show_file.image(file)
            
            image = Image.open(file)

            weights_file = load_weights()
            model = load_model(weights_file)

            results = model(image, size=256)

            results.render()
            im_base64 = Image.fromarray(results.imgs[0])
            st.image(im_base64)
            file.close()

                      
    
def doctor_page():
    with st.columns([1,6,1])[1]:
        st.title('Здоровые зубы - ***круто***')
        instructions = """
            Страница работника дошкольных/школьных учреждений
            """
        st.subheader(instructions)
        st.write('Предлагается выбрать кол-во людей для проверки')
    
    #Table
    col1, col2, buff2 = st.columns([1,3,1])
    id = col1.number_input('Введите кол-во детей',step=1, key ="but1")
    st.write('Предлагается выбрать кол-во людей для проверки')
    buff2.title('')

    def show_img(file):
        json_file = json.loads(file.getvalue())
        image = base64.b64decode(json_file[0]['data'])
        st.image(image)

    # if file is not None:
    #     show_img(file)
        #st.write(file)
        #st.write(file.getvalue())
        #st.write(file.getvalue().decode("utf-8"))
        #lottie_json = load_lottieurl(file.name)
        #st_lottie(file.getvalue())
        #image = base64.b64decode(json.loads(file.getvalue()))
        #st.write(image)
        #st.image(image)
    for i in range(int(id)):
        col1, col2, col3, col4 = st.columns([1,1,1,2])
        last_name = col1.text_input('Фамилия', key = f'1{i}')
        first_name = col2.text_input('Имя', key = f'2{i}' )
        father_name = col3.text_input('Отчество', key = f'3{i}')
        image = col4.file_uploader('file upload', key = f'gen{i}')
        preview = col1.empty()
        if image:
            with st.expander("Загруженные фото"):
                col1.preview.image(image)



def sidebar():
    page_names_to_funcs = {
        "Страница Пользователя": main_page,
        "Страница Работника": doctor_page,
    }

    lottie_path_teeth = 'data/36878-clean-tooth.json' 
    with open(lottie_path_teeth, "r") as f:
        lottie_teeth = json.load(f)

    with st.sidebar:
        st_lottie(
            lottie_teeth,
            loop = True,
            quality='high',
            height=300,
            width=200,
        )

    selected_page = st.sidebar.selectbox("Выбрать страницу", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

    st.sidebar.markdown(
        '''
        Автоматическая система, основанная на алгоритмах компьютерного зрения (Yolo)
        Данная система позволяет:
        1. Самостоятельно проверить и получить рекомендации о внешнем состоянии зубов
        2. Сократить нагрузку на профильных специалистов в дошкольных/школьных учреждениях для проверки состояния зубов
        
        Возможно применение в двух сценариях - как для индивидульного использования любым пользователем, так и использование 
        даже непрофильными специалистами в школах и садах для получения списка рекомендуемых к дополнительному обследованию детей.
        \n
        ___
        Developed by team **fit_predict**
        2022 г.
        '''
    )

if __name__ == '__main__':
    st.set_page_config(
        page_title="Кариеса.нет",
        page_icon="🍫",
        layout="wide",
    )

    # STYLES
    with open('data/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
    
    sidebar()
    
    
