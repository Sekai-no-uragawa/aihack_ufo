# import json
# from io import BytesIO
# from PIL import Image
# import os

import streamlit as st
from streamlit_lottie import st_lottie
import json
from PIL import Image
from io import BytesIO, StringIO

def main_page():
    file_types = ["png", "jpg"]

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
            file.close()


def doctor_page():
    st.title('РОТОВАЯ ПОЛОСТЬ cheeeck')
    instructions = """
        Загрузи свою ротовую полость и узнай все свои секреты
        """
    st.write(instructions)

    #TABLE
    col1, col2, buff2 = st.columns([1,3,1])
    id = col1.number_input('Введите id ребёнка',step=1, key ="but1")
    file = col2.file_uploader('Загрузите', type = 'json')
    buff2.title('')
    
    
    if file is not None:
        st.write(file)
        st.write(file.getvalue())
        st.write(file.getvalue().decode("utf-8"))


    


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
    with open('data\style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
    
    sidebar()
    
    
