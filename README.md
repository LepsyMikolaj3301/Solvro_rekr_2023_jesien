# Solvro_rekr_2023_jesien
Zadanie rekrutacyjne SOLVRO - sezon jesien 2023 - Machine Learning

Model rozpoznający Duże Litery alfabetu według części listy MNIST przesłanej przez platforme Kaggle

Pliki:
main.py jest głównym programem, który zawiera menu przez które można model trenować
NIE TRENUJEMY MODELU PRZEZ PLIK presentation_of_results.ipynb !

NOTE !!!
requirements nie działa dla wersji Python 3.8 lub nowszej!

Z tego powodu tu podaję komendy z bibliotekami:

```pip install torch torchvision numpy matplotlib torchaudio --index-url https://download.pytorch.org/whl/cu117```


Disclaimer:

Projekt pisałem w ok. tydzień, zazwyczaj po 3 piwach późnymi wieczorami
najlepsza dokładność jaką osiągnąłem to ok. 94% dla CrossEntropyLoss 
Kod zapewnie nie najlepszej jakości 
Pozdrawiam



DODATEK I WYTŁUMACZENIE

Zmiany w projekcie:
- Dodanie EDA
- Zepsucie kodu xd

Epilog projektu:

    Projekt wykazywał duży potencjał, jednak długa przerwa spowodowana nauką na sesje, sprawiła, że zajmie mi więcej czasu na zrozumienie starego kodu, niż napisanie projektu od nowa, wraz ze zdobytą wiedzą z tego projektu
    Kod "SIĘ ZEPSUŁ", jednak na pewno można sprawić by wszystko działało
    Sprawia jednak problem wersja Pytorcha, która, nie każda zaś, wspiera odpowiednio karty graficzne z CUDA, co znowu odbija się na efektywności i optymalizacji modelu, nie chce mi sie tego naprawiać xd
    
    KONKLUZJA
    Powrócę do pomysłu z podobnym podejściem, jednak z większym zaplanowaniem pracy i na trzeźwo ( może ).

