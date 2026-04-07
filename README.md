# Исследование обучения GAN на Fashion-MNIST

Небольшой учебный проект на Python и PyTorch, в котором реализована простая генеративно-состязательная сеть для датасета Fashion-MNIST. Основная цель проекта — показать, как по мере обучения генератор переходит от случайного шума к изображениям, похожим на предметы одежды.

## Возможности проекта

- обучение простой fully-connected GAN на Fashion-MNIST;
- автоматическая загрузка датасета при первом запуске;
- сохранение изображений генератора на разных эпохах;
- построение графика потерь генератора и дискриминатора;
- создание сводной иллюстрации с прогрессом обучения;
- сохранение итоговых весов модели;
- отдельная генерация новых изображений после завершения обучения;
- сохранение CSV-истории и JSON-сводки по запуску.

## Структура проекта

```text
HW5/
|-- config.py
|-- gan_project_walkthrough.ipynb
|-- generate_samples.py
|-- models.py
|-- README.md
|-- report.md
|-- requirements.txt
|-- train_gan.py
`-- utils.py
```

После обучения автоматически появляются:

```text
outputs/
|-- checkpoints/
|-- generated/
|-- plots/
|-- samples/
|-- training_history.csv
`-- training_summary.json
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск обучения

```bash
python train_gan.py
```

Во время обучения скрипт:

- загружает Fashion-MNIST;
- обучает генератор и дискриминатор;
- выводит прогресс обучения в консоль;
- сохраняет изображения на выбранных эпохах;
- строит график лоссов;
- сохраняет итоговые веса;
- формирует таблицу истории обучения и итоговую сводку.

## Генерация изображений после обучения

```bash
python generate_samples.py
```

Пример с параметрами:

```bash
python generate_samples.py --checkpoint outputs/checkpoints/generator_final.pth --num-samples 16 --grid-cols 4 --seed 42 --output outputs/generated/my_samples.png
```

## Основные параметры

Основные настройки находятся в `config.py`.

- `batch_size = 128`
- `latent_dim = 100`
- `learning_rate = 0.0002`
- `epochs = 30`
- `sample_interval = 5`
- `log_interval = 50`
- `random_seed = 42`

## Ключевые результаты

### График потерь

![График потерь GAN](outputs/plots/loss_curve.png)

### Прогресс генерации по эпохам

![Прогресс генерации по эпохам](outputs/plots/training_progress.png)

### Итоговая генерация после обучения

<p align="center">
  <img src="outputs/generated/final_training_samples.png" alt="Итоговая генерация после обучения" width="360">
  <img src="outputs/generated/generated_samples.png" alt="Новая генерация из сохраненной модели" width="360">
</p>

### Сравнение ранней и поздней эпохи

<p align="center">
  <img src="outputs/samples/sample_epoch_001.png" alt="Эпоха 1" width="260">
  <img src="outputs/samples/sample_epoch_030.png" alt="Эпоха 30" width="260">
</p>

## Какие файлы сохраняются

- `outputs/samples/sample_epoch_001.png`, `sample_epoch_005.png` и другие изображения по эпохам;
- `outputs/checkpoints/generator_final.pth` — веса генератора;
- `outputs/checkpoints/discriminator_final.pth` — веса дискриминатора;
- `outputs/plots/loss_curve.png` — график потерь;
- `outputs/plots/training_progress.png` — единая картинка, показывающая прогресс генерации по эпохам;
- `outputs/generated/final_training_samples.png` — итоговая сетка после обучения;
- `outputs/generated/generated_samples.png` — новые изображения из отдельного скрипта;
- `outputs/training_history.csv` — история средних потерь по эпохам;
- `outputs/training_summary.json` — краткая сводка по запуску.

## Краткое описание архитектуры

Генератор получает случайный вектор размерности `100`, пропускает его через несколько полносвязных слоёв с `ReLU` и `BatchNorm` и выдаёт изображение `1x28x28` с активацией `Tanh`.

Дискриминатор получает изображение Fashion-MNIST, разворачивает его в вектор и классифицирует как реальное или сгенерированное при помощи полносвязных слоёв с `LeakyReLU` и выходом `Sigmoid`.

## Что удобно показывать преподавателю

- изображения по эпохам из `outputs/samples/`;
- итоговый график `loss_curve.png`;
- общую иллюстрацию `training_progress.png`;
- готовую генерацию `generated_samples.png`;
- краткий отчёт в `report.md`.

## Важно для GitHub

Чтобы изображения отображались прямо в `README.md`, а не как ссылки, нужно загрузить в репозиторий саму папку `outputs/` вместе с файлами изображений. Встроенные картинки в этом README используют относительные пути внутри репозитория, поэтому после `git add`, `git commit` и `git push` GitHub покажет их автоматически.
