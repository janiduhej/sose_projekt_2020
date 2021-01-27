# Code für die Ausführung von Änderungsklassifikation Änderungsmodell + Vortrainierte Embeddings

Dieses Repo enthält den Code für die Erweiterung des in [[1]](https://arxiv.org/pdf/1805.09145.pdf) und [[2]](http://ceur-ws.org/Vol-2377/paper_2.pdf) beschriebenen Ansatzes um ein 
Veränderungsmodell. Der beschriebene Ansatz soll so erweitert werden, dass Veränderungen, die mithilfe 
einer Erweiterung des Protégé-Plugins owl-diff [[3]](https://github.com/mhfj/owl-diff/tree/exportFunctionality)
exportiert werden, in den Klassifikationsprozess einbezogen werden können.

# Ausführen des Codes

## Benötigte Pakere
Zum ausführen erst requirements installieren:
```bash
pip install -r requirements.txt
```

## Daten
Die benötigten Daten liegen nicht alle in diesem Repo. Ein tar-Archiv der übrigen benötigten Daten ist 
unter [https://www.cs.hs-rm.de/~jurisch/embeddings_sose2020_masterprojekt.tar.gz](https://www.cs.hs-rm.de/~jurisch/embeddings_sose2020_masterprojekt.tar.gz) zu finden. **Achtung**: die 
Daten sind ausgepackt etwa 16.8 GB groß, ich würde sie vielleicht nicht unbedingt auf den Hochschulrechnern im Home-Verzeichnis auspacken.

## Ausführen für TransE, TransH, Distmult

Das vortrainierte Embedding kann geändert werden, in dem der Parameter `embedding_file` in der Datei [ChangeModelsAusertung.py](ChangeModelsAuswertung.py) geändert 
wird. Zum Starten von Training und Auswertung:

```bash
python ChangeModelsAuswertung.py
```

## Ausführen für Complex:

```bash
python ChangeModelsAuswertung_COMPLEX.py
```