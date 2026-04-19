# BIAI Project — Genetic Algorithms for TSP and CartPole

> Proponowana nazwa repozytorium: **`biai-ga-tsp-cartpole`**

## Requirements given by instructor

Welcome to the BIAI project,
As per the schedule, we are starting the project the day after tomorrow (Wednesday), which will run until the end of June, so please make sure you attend.
What will the project involve?
The aim of your project will be to implement a genetic algorithm to solve a problem of your choice. In other words, you will need to get the programme to learn how to solve the problem (e.g. find the shortest path) on its own. You can see how such a genetic algorithm works in practice in this video:https://www.youtube.com/watch?v=mA8z0GndiYI
How to carry out the project?
The project will be carried out in pairs. The project will consist of two tasks:
At the start, each group will choose one task as per your own selection or choice. The aim of the first task is to familiarise you with how genetic algorithms work.
The second task will involve adapting a genetic algorithm to one of the OpenAI test environments available in the library:  https://gymnasium.farama.org/ The aim of the second task will be to use a genetic algorithm in an environment with a graphical interface to solve a practical problem.
Finally (during the last class), there will be presentations. Each group will prepare a presentation demonstrating how the genetic algorithm works on a problem of their choice.
3. Tools used in the project
The project will be developed in your choice programming platform (e.g., Python etc.). As an example, the  https://pygad.readthedocs.io/en/latest/, library will be used to implement the genetic algorithm; this library provides a simple interface for defining the components of a genetic algorithm, such as population, selection, crossover and mutation. All plots will be created using the Matplotlib library. Ultimately, I would like your projects to be hosted on GitHub.

## Opis projektu

Projekt realizowany w ramach przedmiotu **BIAI** polega na zaprojektowaniu i zaimplementowaniu **algorytmu genetycznego** do rozwiązywania problemów optymalizacyjnych oraz problemów sterowania agentem w środowiskach z biblioteki **Gymnasium**.

Celem projektu jest pokazanie, że algorytm genetyczny może zostać wykorzystany zarówno do klasycznego problemu kombinatorycznego, jak i do nauki polityki działania agenta w środowisku symulacyjnym.

W projekcie wybrano dwa zadania:

1. **Task 1 — Traveling Salesman Problem (TSP)**
2. **Task 2 — CartPole-v1 w Gymnasium**

Taki zestaw zadań pozwala najpierw zrozumieć i zaprezentować podstawowe mechanizmy działania algorytmu genetycznego, a następnie zastosować tę samą ideę do praktycznego problemu sterowania agentem w środowisku posiadającym interfejs graficzny.

---

## Cel projektu

Głównym celem projektu jest:

- zrozumienie, jak działa algorytm genetyczny,
- implementacja podstawowych operatorów ewolucyjnych,
- porównanie wpływu parametrów algorytmu na jakość rozwiązania,
- wykorzystanie algorytmu genetycznego do uczenia agenta w środowisku Gymnasium,
- przygotowanie demonstracji działania algorytmu oraz prezentacji końcowej.

---

## Dlaczego wybraliśmy akurat te dwa zadania?

### 1. Traveling Salesman Problem (TSP)

Problem komiwojażera jest jednym z najbardziej klasycznych problemów optymalizacyjnych. Polega on na znalezieniu najkrótszej trasy odwiedzającej wszystkie miasta dokładnie raz i wracającej do punktu startowego.

To zadanie jest bardzo dobrym wyborem do pokazania działania algorytmu genetycznego, ponieważ:

- rozwiązanie można przedstawić jako **permutację miast**,
- łatwo zdefiniować funkcję celu,
- łatwo wizualizować postęp algorytmu,
- można porównać działanie różnych operatorów mutacji i krzyżowania.

### 2. CartPole-v1

CartPole to klasyczne środowisko sterowania agentem, w którym celem jest utrzymanie kija w równowadze na poruszającym się wózku tak długo, jak to możliwe.

To środowisko zostało wybrane, ponieważ:

- jest stosunkowo proste implementacyjnie,
- dobrze nadaje się do pierwszych eksperymentów z Gymnasium,
- ma czytelną definicję nagrody,
- pozwala łatwo zaprezentować działanie wyuczonego agenta,
- daje możliwość zapisania filmu lub animacji z działania rozwiązania.

Dzięki temu projekt pozostaje ambitny, ale jednocześnie realistyczny do wykonania w ograniczonym czasie.

---

## Technologie i biblioteki

Projekt będzie realizowany w języku **Python** z wykorzystaniem następujących bibliotek:

- **NumPy** — operacje numeryczne,
- **Matplotlib** — tworzenie wykresów i wizualizacji,
- **PyGAD** — implementacja algorytmu genetycznego,
- **Gymnasium** — środowiska testowe dla agentów,
- opcjonalnie **Jupyter Notebook** — analiza wyników i eksperymenty.

---

## Architektura projektu

Proponowana struktura repozytorium:

```text
biai-ga-tsp-cartpole/
├── README.md
├── requirements.txt
├── src/
│   ├── task1_tsp/
│   │   ├── tsp_ga.py
│   │   ├── operators.py
│   │   └── visualize_tsp.py
│   ├── task2_cartpole/
│   │   ├── policy.py
│   │   ├── fitness.py
│   │   ├── train_ga.py
│   │   ├── evaluate.py
│   │   └── record_agent.py
│   └── common/
│       ├── utils.py
│       └── config.py
├── results/
│   ├── plots/
│   ├── videos/
│   └── best_models/
├── notebooks/
│   └── experiments.ipynb
└── slides/
    └── presentation.pptx
```

Taki podział ułatwia organizację kodu, eksperymentów i materiałów do prezentacji.

---

## Task 1 — Traveling Salesman Problem (TSP)

### Opis zadania

W pierwszej części projektu algorytm genetyczny będzie używany do znalezienia możliwie najkrótszej trasy pomiędzy zadanym zbiorem miast.

Miasta będą reprezentowane jako punkty w przestrzeni 2D.

### Reprezentacja chromosomu

Chromosom będzie reprezentowany jako **permutacja indeksów miast**.

Przykład:

```python
[0, 3, 1, 4, 2]
```

oznacza kolejność odwiedzania miast.

### Funkcja fitness

Ponieważ chcemy zminimalizować długość trasy, funkcję fitness można zdefiniować jako:

```python
fitness = 1 / (distance + epsilon)
```

Im krótsza trasa, tym większa wartość fitness.

### Operatory genetyczne

W tej części projektu planowane jest wykorzystanie następujących operatorów:

#### Selekcja
- selekcja turniejowa,
- ewentualnie selekcja ruletkowa do porównania.

#### Krzyżowanie
Dla permutacji zwykły crossover nie działa poprawnie, dlatego należy użyć operatora dostosowanego do permutacji, np.:
- **OX (Order Crossover)**,
- **PMX (Partially Mapped Crossover)**.

#### Mutacja
Możliwe warianty mutacji:
- **swap mutation** — zamiana miejscami dwóch miast,
- **inversion mutation** — odwrócenie fragmentu trasy,
- **scramble mutation** — losowe przetasowanie fragmentu chromosomu.

#### Elityzm
Najlepszy osobnik z danej generacji będzie przenoszony do kolejnej generacji bez zmian.

### Co chcemy pokazać w wynikach?

Dla TSP planujemy przedstawić:

- najlepszą trasę znalezioną przez algorytm,
- długość trasy w kolejnych generacjach,
- średnią wartość fitness populacji,
- wpływ liczby generacji, rozmiaru populacji i współczynnika mutacji,
- porównanie różnych operatorów mutacji lub crossoveru.

### Wizualizacje

Planowane wykresy i rysunki:

- wykres **best fitness per generation**,
- wykres **average fitness per generation**,
- wizualizacja najlepszej trasy na płaszczyźnie 2D,
- porównanie wyników dla różnych parametrów.

---

## Task 2 — CartPole-v1 w Gymnasium

### Opis zadania

W drugiej części projektu algorytm genetyczny będzie używany do optymalizacji polityki sterującej agentem w środowisku **CartPole-v1**.

Celem agenta jest utrzymanie drążka w równowadze przez możliwie długi czas. W każdej iteracji agent obserwuje stan środowiska i wybiera jedną z dwóch akcji:

- ruch w lewo,
- ruch w prawo.

### Reprezentacja rozwiązania

W tej części chromosom nie będzie permutacją, lecz **wektorem parametrów polityki sterowania**.

Rozważamy dwie wersje:

#### Wersja podstawowa — kontroler liniowy

Agent podejmuje decyzję na podstawie prostego modelu liniowego:

```python
action = argmax(W @ state + b)
```

gdzie:
- `state` to wektor obserwacji,
- `W` to macierz wag,
- `b` to bias.

Cały chromosom to po prostu spłaszczony zestaw wag i biasów.

#### Wersja rozszerzona — mała sieć neuronowa

Można także zastosować prostą sieć typu:

- wejście: 4 neurony,
- warstwa ukryta: 8–16 neuronów,
- wyjście: 2 neurony.

W takim wariancie chromosom zawiera wszystkie wagi i biasy sieci.

### Funkcja fitness

Fitness będzie mierzony jako średnia nagroda uzyskana przez danego osobnika w kilku epizodach:

```python
fitness = mean(total_reward over N episodes)
```

Zastosowanie średniej z kilku epizodów zmniejsza wpływ losowości środowiska.

### Przebieg ewolucji

Dla każdego osobnika:

1. dekodujemy chromosom do parametrów polityki,
2. uruchamiamy środowisko,
3. pozwalamy agentowi działać do końca epizodu,
4. obliczamy łączną nagrodę,
5. powtarzamy ocenę kilka razy,
6. średnią nagrodę traktujemy jako fitness.

### Co chcemy pokazać w wynikach?

Dla CartPole planujemy przedstawić:

- wzrost średniej nagrody w kolejnych generacjach,
- najlepszy wynik uzyskany przez populację,
- porównanie różnych parametrów algorytmu,
- zapis działania najlepszego agenta,
- krótką analizę tego, czy GA potrafi nauczyć stabilnej polityki bez klasycznego uczenia gradientowego.

### Dodatkowe materiały do prezentacji

W tej części projektu szczególnie wartościowe będą:

- film lub gif z działania najlepszego agenta,
- wykres nagrody względem liczby generacji,
- porównanie wyników dla różnych ustawień populacji i mutacji.

---

## Algorytm genetyczny — elementy wspólne

W obu zadaniach algorytm genetyczny będzie składał się z tych samych podstawowych etapów:

1. **Inicjalizacja populacji** — generujemy losowe rozwiązania.
2. **Ocena fitness** — mierzymy jakość każdego osobnika.
3. **Selekcja rodziców** — wybieramy najlepszych kandydatów do reprodukcji.
4. **Krzyżowanie** — tworzymy potomstwo z cech rodziców.
5. **Mutacja** — wprowadzamy losowe zmiany zwiększające różnorodność.
6. **Elityzm / zastępowanie populacji** — tworzymy kolejną generację.
7. **Warunek stopu** — kończymy po określonej liczbie generacji lub po osiągnięciu satysfakcjonującego wyniku.

---

## Plan implementacji

### Etap 1 — przygotowanie repozytorium

- utworzenie repozytorium GitHub,
- przygotowanie struktury folderów,
- dodanie `README.md`,
- skonfigurowanie `requirements.txt`.

### Etap 2 — implementacja TSP

- generowanie zbioru miast,
- implementacja reprezentacji chromosomu,
- implementacja funkcji odległości,
- implementacja operatorów genetycznych,
- uruchomienie eksperymentów,
- zapis wykresów i najlepszej trasy.

### Etap 3 — implementacja CartPole

- przygotowanie środowiska Gymnasium,
- zdefiniowanie polityki sterowania,
- zakodowanie parametrów polityki jako chromosomu,
- zdefiniowanie funkcji fitness,
- uruchomienie treningu z wykorzystaniem PyGAD,
- zapis wyników i nagrań najlepszego agenta.

### Etap 4 — analiza wyników

- porównanie ustawień hiperparametrów,
- zebranie wykresów,
- wyciągnięcie wniosków,
- przygotowanie materiałów do prezentacji.

### Etap 5 — prezentacja końcowa

- omówienie działania GA,
- przedstawienie obu zadań,
- pokaz wykresów,
- demonstracja najlepszego rozwiązania,
- podsumowanie zalet i ograniczeń podejścia.

---

## Podział pracy w zespole

### Osoba 1

- implementacja Task 1 — TSP,
- implementacja operatorów genetycznych,
- wizualizacje tras i wykresów.

### Osoba 2

- implementacja Task 2 — CartPole,
- definicja polityki sterowania,
- funkcja fitness i integracja z Gymnasium,
- zapis działania najlepszego agenta.

### Wspólnie

- dobór hiperparametrów,
- analiza wyników,
- przygotowanie README,
- przygotowanie slajdów,
- przygotowanie końcowej prezentacji.

---

## Proponowane hiperparametry początkowe

Na start można użyć następujących ustawień:

- liczba osobników w populacji: `30–80`,
- liczba generacji: `100–300`,
- liczba rodziców: `10–20`,
- współczynnik mutacji: `0.05–0.15`,
- elityzm: `1–2` najlepszych osobników,
- liczba epizodów do oceny jednego osobnika w CartPole: `3–5`.

W dalszej części projektu parametry te będą strojenie eksperymentalnie.

---

## Wymagania i instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/<twoj-login>/biai-ga-tsp-cartpole.git
cd biai-ga-tsp-cartpole
```

### 2. Utworzenie środowiska wirtualnego

```bash
python -m venv venv
source venv/bin/activate
```

W systemie Windows:

```bash
venv\Scripts\activate
```

### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```

Przykładowa zawartość `requirements.txt`:

```txt
numpy
matplotlib
pygad
gymnasium[classic-control]
```

---

## Przykładowy sposób uruchamiania

### Task 1 — TSP

```bash
python -m src.task1_tsp.tsp_ga
```

### Task 2 — CartPole

```bash
python -m src.task2_cartpole.train_ga
```

### Ewaluacja najlepszego agenta

```bash
python -m src.task2_cartpole.evaluate
```

---

## Jakie wyniki chcemy uzyskać?

### Dla TSP

- coraz krótsza trasa wraz z kolejnymi generacjami,
- stabilna poprawa najlepszego osobnika,
- wyraźna różnica między parametrami mutacji i crossoveru.

### Dla CartPole

- wzrost średniej nagrody w czasie,
- uzyskanie stabilnej polityki utrzymującej drążek przez długi czas,
- materiał wideo pokazujący działanie najlepszego agenta.

---

## Możliwe rozszerzenia projektu

Jeśli po wykonaniu podstawowej wersji zostanie czas, można rozważyć:

- porównanie własnej implementacji GA z PyGAD,
- porównanie kilku metod selekcji,
- test innego środowiska Gymnasium, np. LunarLander,
- użycie prostego MLP zamiast kontrolera liniowego,
- zapisanie najlepszego rozwiązania do pliku i ponowne odtwarzanie bez treningu,
- porównanie działania dla różnych seedów losowych.

---

## Potencjalne trudności

Podczas realizacji projektu mogą pojawić się następujące wyzwania:

- zbyt szybka utrata różnorodności w populacji,
- zbyt słaba eksploracja przy niskiej mutacji,
- niestabilne wyniki przy ocenie agenta w pojedynczym epizodzie,
- zbyt duży czas działania przy rozbudowanej polityce sterowania,
- konieczność dobrania odpowiedniej reprezentacji chromosomu.

Dlatego ważne będzie testowanie kilku konfiguracji i dokumentowanie wpływu parametrów na wynik.

---

## Wnioski, które chcemy pokazać na końcu

Po zakończeniu projektu chcemy wykazać, że:

- algorytm genetyczny jest elastycznym narzędziem do rozwiązywania różnych klas problemów,
- odpowiednia reprezentacja chromosomu ma kluczowe znaczenie,
- dobór operatorów i parametrów wpływa bezpośrednio na jakość rozwiązania,
- algorytm genetyczny może działać nie tylko w klasycznych problemach optymalizacyjnych, ale również w środowiskach sterowania agentem.

---

## Materiały do prezentacji końcowej

Na prezentację planujemy przygotować:

- krótki wstęp teoretyczny o algorytmach genetycznych,
- opis obu problemów,
- schemat działania algorytmu,
- najważniejsze wykresy,
- wizualizację najlepszego rozwiązania TSP,
- demo lub wideo działania agenta CartPole,
- krótkie podsumowanie zalet i ograniczeń podejścia.

---

## Autorzy

- **[Imię i nazwisko 1]**
- **[Imię i nazwisko 2]**

---

## Status projektu

Projekt w trakcie realizacji.

---

## Licencja

Projekt edukacyjny przygotowany na potrzeby zajęć BIAI.
