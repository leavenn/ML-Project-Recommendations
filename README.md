
# System Rekomendacji Produktów

Projekt demonstruje system rekomendacji produktów oparty na modelu głębokiego uczenia. Jego kluczowym elementem jest autoenkoder, który uczy się numerycznych reprezentacji (tzw. "embeddingów") dla produktów kosmetycznych, co pozwala na znajdowanie i sugerowanie podobnych artykułów.  
Dołączona aplikacja webowa napisana w frameworku Flask służy jako wizualny interfejs do prezentacji działania modelu.


## Opis Modelu
Główną częścią systemu jest autoenkoder zbudowany w TensorFlow/Keras, który tworzy skompresowaną reprezentację każdego produktu na podstawie jego najważniejszych cech.


## Cechy Wejściowe
Model jest trenowany na podstawie czterech głównych cech każdego produktu:
  * Price_USD (Cena)
  * Rating (Ocena
  * Brand (Marka)
  * Category (Kategoria)

Przed treningiem cechy kategoryczne (Brand, Category) są konwertowane na liczby (LabelEncoder), a cechy numeryczne (Price_USD, Rating) są normalizowane (StandardScaler).


## Architektura i Trening
Autoenkoder składa się z:
  * Encodera: Redukuje wymiarowość danych wejściowych do 8-wymiarowej przestrzeni (embedding).
  * Decodera: Stara się zrekonstruować oryginalne dane wejściowe na podstawie embeddingu.

Jeśli wytrenowany model nie jest dostępny, skrypt trainer.py automatycznie go buduje i trenuje.


## Generowanie Rekomendacji
Dla każdego produktu w bazie generowany jest 8-wymiarowy wektor embeddingu.  
Gdy użytkownik wybierze produkty, system oblicza dla nich średni wektor embeddingu.  
Używając podobieństwa kosinusowego, system porównuje ten średni wektor z wektorami wszystkich innych produktów.  
Produkty o najwyższym wyniku podobieństwa są prezentowane jako rekomendacje.


## Resetowanie Modelu
Aplikacja automatycznie zapisuje wytrenowany model oraz przetworzone dane, aby przyspieszyć kolejne uruchomienia. Jeśli chcesz wymusić ponowne przetworzenie danych i trening modelu od zera (np. po zmianie danych wejściowych w products.csv), musisz usunąć zapisane pliki.  

Aby to zrobić, usuń zawartość następujących folderów:  
```bash  
  
  model/model_data/
  
```
```bash  
  
  data/preprocessors/
  
```
Przy następnym uruchomieniu python app.py, skrypty automatycznie odtworzą te pliki.
## Jak uruchomić projekt
  *  Sklonuj repozytorium lub pobierz pliki  
Upewnij się, że wszystkie pliki projektu znajdują się w jednym folderze.
  * Zainstaluj wymagane biblioteki  
Wpisz poniższą komendę w terminalu, aby zainstalować wszystkie zależności:
```bash
    
pip install Flask tensorflow scikit-learn pandas numpy matplotlib joblib
  
```
  * Uruchom aplikację Flask  
Po pomyślnej instalacji bibliotek, uruchom główny plik aplikacji:
```bash  
  
python app.py
  
```
  * Otwórz aplikację w przeglądarce  
  Aplikacja będzie dostępna pod adresem: localhost:5000
