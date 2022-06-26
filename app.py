# import json
# from io import BytesIO
# from PIL import Image
# import os

import streamlit as st
from streamlit_lottie import st_lottie
import json
from PIL import Image
from io import BytesIO, StringIO
import base64
import numpy as np
import torch
import urllib
import pandas as pd
import pydeck as pdk



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
        st.title("Технологии ИИ в детской стоматологии")
        st.subheader('Поиск кариеса по фотографии с помощью технологий компьютерного зрения')
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
            exp = st.expander("Загруженные фото", expanded=False)
            exp.image(file)
            
            image = Image.open(file)

            weights_file = load_weights()
            model = load_model(weights_file)

            results = model(image, size=256)

            results.render()
            im_base64 = Image.fromarray(results.imgs[0])
            st.image(im_base64)
            file.close()
    
    st.write('''___''')
    st.markdown('<p class="big-font">Рекомендации для профилактики кариеса</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    lottie_path_teeth = 'data/28719-brush-teeth.json'
    with open(lottie_path_teeth, "r") as f:
        lottie_teeth = json.load(f)
    with col2:
        st_lottie(
            lottie_teeth,
            loop = True,
            quality='high',
            height=280,
            width=280,
            key = 'hello2',
        )
    with col1:
        lst = ['Правильно чистить зубы',
        'Использовать ополаскиватель для полости рта',
        'Восполнять недостаток фтора',
        'Избегать приема контрастной по температуре пищи',
        'Регулярно посещать стоматолога']
        for i in lst:
            st.markdown("- " + i)

    
    st.markdown('<p class="big-font">Ближайшие стоматологические клиники</p>', unsafe_allow_html=True)
    #MAP
    df = pd.DataFrame(
     [[55.7512656095198,37.60662577202603, 'Proffessor Clinic'],
     [55.75690467127408,37.61402866890714, 'Smile'],
     [55.7583938140869,37.609729497404224, 'Clinica #1'],
     [55.74708381669314,37.63060834818813, 'Dental Art']],
     columns=['lat', 'lon', 'name'])
    st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=55.75134389970574,
        longitude=37.6172831338021,
        zoom=13,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
        'ScatterplotLayer',
        data=df,
        get_position='[lon, lat]',
        get_radius = 70,
        get_color = ['255','30','30']
        ),
        pdk.Layer(
        type='TextLayer',
        data=df,
        get_position='[lon, lat]',
        get_text='name',
        getTextAnchor= '"middle"',
        get_alignment_baseline='"bottom"',
        get_size=100,
        sizeUnits='meters',
        ),
    ],
))

                      
    
def doctor_page():
    with st.columns([1,6,1])[1]:
        st.title('Технологии ИИ в детской стоматологии')
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
    pred_flag = False

    id_name_dict = {}
    for i in range(int(id)):
        st.write('''___''')
        col1, col2, col3, col4 = st.columns([1,1,1,2])
        last_name = col1.text_input('Фамилия', key = f'1{i}')
        first_name = col2.text_input('Имя', key = f'2{i}' )
        father_name = col3.text_input('Отчество', key = f'3{i}')
        imgs_worker = col4.file_uploader('file upload', key = f'gen{i}', accept_multiple_files=True)
        preview = col1.empty()
        
        if imgs_worker:
            pred_flag = True
            if last_name and first_name and father_name:
                fio = f'{last_name} {first_name[0]}.{father_name[0]}.'
            elif first_name and last_name:
                fio = f'{last_name} {first_name[0]}.'
            elif last_name:
                fio = f'{last_name}'
            else:
                fio = f'учащегося №{str(i+1)}'
            
            if '№' not in fio:
                id_name_dict.setdefault(i, fio)
            else:
                id_name_dict.setdefault(i, f'Пользователь №{i+1}')
            # if not first_name and not father_name:
            #     if not last_name:
            #         fio = f'учащегося №{str(i)}'
            #     else:
            #         fio = f'{last_name}'
            # elif first_name and last_name:
            #     fio = f'{last_name} {first_name[0]}.'
            # elif last_name and first_name and father_name:
            #     fio = f'{last_name} {first_name[0]}.{father_name[0]}.'
            expand = st.expander(f"Загруженные фото для {fio}")
            with col1:
                expand.image(imgs_worker)
    if pred_flag:
        pred_button = st.button('Начать анализ', key=f'pred')
        if pred_button:
            image_array = []
            st.write(st.session_state)
            for key in st.session_state:
                if 'gen' in key:
                    for upload_images in st.session_state[key]:
                        image_array.append((key[3:], upload_images))
            st.write(image_array)
            
            weights_file = load_weights()
            model = load_model(weights_file)
            
            data_to_out = pd.DataFrame(columns=['id', 'percent'])

            pred_dict = dict()
            for kid_id, one_image in image_array:
                image = Image.open(one_image)
                results = model(image, size=256)
                df_pred = results.pandas().xyxy[0]
                if df_pred.shape[0] != 0:
                    pred_dict.setdefault(kid_id, [df_pred.confidence.max()*100]).append(df_pred.confidence.max()*100)
                else:
                    pred_dict.setdefault(kid_id, [0]).append(0)
            
            data_to_out = [[k, id_name_dict[int(k)], np.max(v)] for k, v in pred_dict.items()]
            st.write(pd.DataFrame(data_to_out, columns=['id', 'Имя проверяемого','Вероятность наличия кариеса, %']))
            st.write('''___''')
            with st.expander(f"Просмотреть результаты детекции"):
                for kid_id, res_img in image_array:
                    img = Image.open(res_img)
                    st.image(img, caption=id_name_dict[int(kid_id)], width=600)
            

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
        Автоматическая система, основанная на алгоритмах компьютерного зрения (YOLOv5)
        Данная система позволяет:
        1. Самостоятельно проверить и получить рекомендации о внешнем состоянии зубов
        2. Сократить нагрузку на профильных специалистов в дошкольных/школьных учреждениях для проверки состояния зубов
        
        Возможно применение в двух сценариях - как для индивидульного использования любым пользователем, так и использование 
        даже непрофильными специалистами в школах и садах для получения списка рекомендуемых к дополнительному обследованию детей.
        \n
        ___
        Developed by team **fit_predict**\n
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
    
    
