## Plusy:
- Struktura repozytorium i dependencies.txt
- Pliki są częściowo sformatowane, kod udokumentowany
- Zadresowanie problemu niezbalansowanego rozkładu danych - WeightedRandomSampler
- Architektura modelu
- Ewaluacja wyników jakościowa

## Minusy:
- EDA - brak
- Preprocessing - trochę mało, przydałaby się jakaś normalizacja/standaryzacja danych
- Brak ewaluacji wyników ilościowej. Same accuracy za dużo nam nie mówi, warto by było dodać confusion matrixa, F1 Scora, które lepiej reprezentują działanie modelu.
- CrossEntropyLoss na wejściu powinien dostawać 'surowy' output z modelu po ostatniej warstwie liniowej - bez softmaxa. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

## Uwagi:
- Raczej popularną taktyką na podawanie hipermarametrów takich jak liczba epok, learning rate itp. jest plik konfiguracyjny, nie input z konsoli.
- Warto tworzyć repozytoria do projektów MLowych, które w łatwy sposób umożliwiają reprodukcje eksperymentów, podmienianie hiperparametrów, logowanie wyników itp. Istnieje wiele bibliotek, które ułatwiają ten proces, przykładowo: pytorch-lightning, hydra, tensorboard, wandb.

