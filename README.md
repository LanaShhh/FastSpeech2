# FastSpeech2

Данная модель представляет из себя реализацию FastSpeech2, описанную в статье FastSpeech 2: Fast and High-Quality End-To-End Text To Speech (https://arxiv.org/pdf/2006.04558.pdf). 
Модель реализована на основе кода с семинара курса НИУ ВШЭ Deep Learning for Audio 2023 и реализации модели FastSpeech от xcmyz (https://github.com/xcmyz/FastSpeech).


## Инструкция по использованию кода

- Скачиваем данный git-репозиторий

~~~
git clone git@github.com:LanaShhh/FastSpeech2.git
cd FastSpeech2
~~~

- Скачиваем все необходимые библиотеки

~~~
pip install -r requirements.txt
~~~

- Скачиваем модель waveglow (https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx)
- Создаем папку waveglow/pretrained_model
- Перемещаем модель в папку waveglow/pretrained_model и переименовываем как waveglow_256channels.pt

### Обучение 

~~~
python3 training.py wandb_login_key
~~~

wandb_login_key - обязательный аргумент, ключ для авторизации в wandb (https://wandb.ai), можно получить по ссылке https://wandb.ai/authorize

Результат - run в wandb, папка checkpoints с чекпоинтами модели кажддые 1000 шагов, папка train_audio с синтезированными аудио на каждом чекпоинте

### Инференс 

~~~
python3 inference.py model_state_dict_path
~~~

model_state_path - обязательный параметр, путь до модели. 

Для генерации лучшей модели как в репозитории - скачать модель (https://disk.yandex.ru/d/slZzqDiCjUu-Tw), положить в папку checkpoints, использовать model_state_dict "./checkpoints/checkpoint_15000.pth.tar"

Результат - папка results с генерациями по конфигурации тестовых текстов

## Отчет 

Отчет в wandb, с описанием работы, графиками функций ошибок, выводами и сгенерированными аудио во время обучения - TODO

Ссылка на итоговый run в wandb - https://wandb.ai/lana-shhh/fastspeech2_sdzhumlyakova_implementation/runs/x0118aad?workspace=user-lana-shhh


## Итоговая генерация

Для оценки качества полученной модели производилась генерация аудио по 3 предложениям с разными конфигурациями duration/pitch/energy.

Конфигурации: 

~~~
[
        {"speed": 1.0, "pitch": 1.0, "energy": 1.0},
        {"speed": 0.8, "pitch": 1.0, "energy": 1.0},
        {"speed": 1.2, "pitch": 1.0, "energy": 1.0},
        {"speed": 1.0, "pitch": 0.8, "energy": 1.0},
        {"speed": 1.0, "pitch": 1.2, "energy": 1.0},
        {"speed": 1.0, "pitch": 1.0, "energy": 0.8},
        {"speed": 1.0, "pitch": 1.0, "energy": 1.2},
        {"speed": 0.8, "pitch": 0.8, "energy": 0.8},
        {"speed": 1.2, "pitch": 1.2, "energy": 1.2}
]
~~~

TODO добавить генерации

TODO отрисовать генерации в README



## Источники

- Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu. FastSpeech 2: Fast and High-Quality End-To-End Text To Speech. https://arxiv.org/pdf/2006.04558.pdf

- Курс Deep Learning for Audio, НИУ ВШЭ, ПМИ, 2023. https://github.com/XuMuK1/dla2023

- Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu. FastSpeech: Fast, Robust and Controllable Text to Speech. https://arxiv.org/pdf/1905.09263.pdf

- https://github.com/xcmyz/FastSpeech

- pyworld documentation. https://pypi.org/project/pyworld/

- The CMU Pronouncing Dictionary http://www.speech.cs.cmu.edu/cgi-bin/cmudict

- PyTorch documentation. https://pytorch.org/docs/stable/index.html




