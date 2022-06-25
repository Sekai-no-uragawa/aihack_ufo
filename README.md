## Решение кейса "Исскуственный интеллект в детской стоматологии" от команды **fit_predict**. 

[Ссылка на решение, прототип веб-сервиса по детекции кариеса](https://sekai-no-uragawa-aihack-ufo-app-16dmut.streamlitapp.com/)

# Содержание

- [Задача](#task1)
- [Как мы решили кейс](#task2)
    - [Скриншоты](#task2_1)
    - [Что может наше решение](#task2_2)
    - [Что мы хотели бы добавить](#task2_3)
- [Соответствие техническим критериям](#task4)
    - [Проверка кода на запускаемость](#task4_1)
    - [Какую выбрали модель/метод](#task4_2)
    - [Какую метрику выбрали для проверки точности](#task4_3)
    - [На основе каких библиотек/ПО построено решение](#task4_5)

# Задача <a class="anchor" id="task1"></a>
Создание системы детекции по определению больных зубов у детей по фото ротовой полости с web интерфейсом.
На основе фотоизображений ротовой полости, с применением технологий искусственного интеллекта, создать MVP системы детекции с web – интерфейсом по определению больных зубов у детей согласно заданного классификатора.

# Как мы решили кейс <a class="anchor" id="task2"></a>
Реализован web сервис, предоставляющий два типа использования:
1. Любой пользователь, решивший проверить состояние сових зубов или своих детей. Возможность загрузки фото ротовой полости, получения подсвеченных участков, на которые возможно стоит обратить внимание в формате изображения с процентом уверености модели в предсказании
2. Работник детского сада / аналогичного учебного учреждения, имеющий возможность пакетной загрузки изображений, разметки имен для фотографий. Результатом обработки является итоговая сводная таблица с именами и процентами уверености модели в наличии проблем с зубами, на которые стоит обратить внимание.

Вероятность наличия кариеса предоставляется в процентном отношении и на основании этого результата предлагает направить исследуемого на дополнительное обследование к стоматологу. Для работников дошкольных учреждений предоставлена возможность вводить ID профиля ребенка и его фотографии в форматах png, jpg, json и получать список ID детей, рекомендуемых к прохождению осмотра у специалиста. Это позволит, как интегрировать в дальнейшем наш сервис к роботу-помощнику и его системе, так и сберечь нервы юных пациентов. 
Также при использовании в ручном режиме не требуются особые знания, а значит такое предварительное обследование может провести любой пользователь ПК, что поможет снизить нагрузку на профильных специалистов, ведь им будет сразу предложен только релевантный список для проверки.

## Скриншоты решения <a class="anchor" id="task2_1"></a>
![image](https://user-images.githubusercontent.com/96841762/175793358-603fdf02-7550-4e07-b136-866c86039b39.png)

![image](https://user-images.githubusercontent.com/96841762/175793361-fbe11fc9-6c92-4163-b2a1-9d1bf282087d.png)

![image](https://user-images.githubusercontent.com/96841762/175793368-e9ff3238-5149-41d2-8717-962c3e6c4ec5.png)


## Что может наше решение <a class="anchor" id="task2_2"></a>
Реализованный веб-сервис имеет несколько основных функций:
1. 
2.
3. Возможность интеграции с частными стоматологическими клиниками, наличие онлайн карты на сайте и возможность предложения таргетированным пользователям услуг данных компаний в виде рекомендаций.
4. При минимальных трудозатратах данный web сервис трансформируется в десктоп приложение, работующее автономно на любом ПК.

## Что мы хотели бы добавить <a class="anchor" id="task2_3"></a>
1. Использование API робота-помощника. Возможность полной автономатизации всего пайплайна действий до получения результата.
2. Сегментация кариеса и расширение классификатора зубных заболеваний. При наличии большего кол-ва данных по каждому типа патологий возможно более точечно выделять больший список проблем полости рта.
3. Запись в ближайшую рекомендованную клинику
4. Монетизация за счет реализованного взаимодействия с заинтересованными клиниками в таргетированном трафике.

# Соответствие техническим критериям <a class="anchor" id="task4"></a>

## Проверка кода на запускаемость <a class="anchor" id="task4_1"></a>
Веб-сервис представлен в рабочем виде, исходный код представлен в репозитории. Код модели подготовлен в colab. Для открытия и запуска кода модели необходимо:
1. Открыть файл по ссылке ниже;
2. Нажать в открывшемся окне "открыть в приложении colab", в верхней части экрана.

- **[train_model.ipynb](![image](https://user-images.githubusercontent.com/96841762/175793356-41955a21-c01a-471c-ae9e-46608ed983bd.png)
)** - ноутбук с обучением модели.

## Какую выбрали модель/метод <a class="anchor" id="task4_2"></a>
Для построения модели нами изначально было выбрано 3 алгоритма для проведения экспериментов и нахождения лучшего из них. Были протестированы такие алгоритмы как 

++ Общее описание

## Какую метрику выбрали для проверки точности <a class="anchor" id="task4_3"></a>
Была выбрана метрика mAP, на итоговой модели она составила значение порядка 0.71.


## На основе каких библиотек/ПО построено решение  <a class="anchor" id="task4_5"></a>
Все использованные библиотеки предоставляются с открытым исходным кодом. Список библиотек можно найти в файле [requirements.txt](https://github.com/Sekai-no-uragawa/aihack/blob/main/requirements.txt)
