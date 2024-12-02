# Kirkpatrick point location algorithm

To repozytorium zawiera implementację algorytmu Kirckpatricka.

## Klonowanie repozytorium

Aby skolonować to repozytorium musimy skorzystać z komendy:

```bash
git clone https://github.com/WSm-77/Kirkpatrick-point-location.git
```

## Narzędzie bit_algo_vis_tool

Podczas realizacji projektu korzystaliśmy z narzędzia dostarczonego przez koło naukowe **_Bit_**.

### Konfiguracja środowiska

Aby odpowiednio skonfigurować środowisko należy wykonać następujące kroki:

#### 1. Stworzenie środowiska

```bash
conda create --name kirkpatrick python=3.9
conda activate kirkpatrick
```
#### 2. Pobranie niezbędnych pakietów

```bash
python3 setup.py sdist
python3 -m pip install -e .
```

#### 3. Użytkowanie

Teraz jeżeli jesteśmy w nowo stworzonym środowisku (w terminalu przed nazwą użytkownika powinno wyświetlać się __(kirkpatrick)__) tworzymy Jupyter notebook, w którym jako kernel wybieramy interpreter pythonowy z nowo stworzonego środowiska.

W przypadku zwykłych skryptów pythonowych (plików _.py_) w VSCode możemy ustawić interpreter na ten z środowiska __kirkpatrick__:

```
Ctrl+Shift+P > Python: Select Interpreter
```

Miejsce, w którym stworzymy nowy plik nie ma znaczenia - konfiguracja środowiska sprawia, że folder _bit_algo_vis_tool_ jest widoczny z każdego miejsca w projekcie tak jak dowolny inny pakiet pythonowy taki jak _numpy_ czy _matplotlib_, więc jeżeli przykładowo chcemy skorzystać z klasy **_Visualizer_** wystarczy ją zaimportować:

```python
from bit_algo_vis_tool.visualizer.visualizer import Visualizer
```

W przypadku problemów pierwszym krokiem powinno być zrestartowanie środowiska:

```bash
conda deactivate
conda activate kirkpatirck
```
