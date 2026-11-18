# ImageDetector

Erkennung vollständig KI-generierter Bilder mit einer CNN-Referenz und einem
Dual-Resolution-Detektor. Der zweite Ansatz kombiniert das gesamte Bild mit
unskalierten lokalen Ausschnitten. Optionale lokale Referenzmodelle lernen auf
realen Bildern spektrale Rekonstruktion oder Conditional Flow Matching.

## Voraussetzungen

- **Python 3.13** (mit 3.13.15 getestet)
- **Git**
- **VS Code** mit den Extensions *Python* und *Jupyter* (beide von Microsoft)

## Setup

### 1. Repository klonen

```bash
git clone <REPOSITORY-URL>
cd ImageDetector
```

### 2. Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

Falls mehrere Python-Versionen installiert sind (Windows): `py -3.12 -m venv .venv`

### 3. Umgebung aktivieren

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```
Falls PowerShell die Ausführung blockiert, einmalig: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 4. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 5. Umgebungsvariablen anlegen

Die Datei `.env` ist nicht im Repository (enthält lokale Werte und Secrets). Lege sie anhand der Vorlage an:

**Windows:**
```powershell
copy .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Danach in `.env` die Werte anpassen. Mindestens muss dort stehen:

```
PYTHONPATH=.
```

### 6. VS Code einrichten

1. Projektordner öffnen (*File → Open Folder*).
2. Interpreter wählen: `Ctrl+Shift+P` → *Python: Select Interpreter* → Eintrag aus `.venv` auswählen.
3. Für Notebooks in `notebooks/` denselben Interpreter als Kernel auswählen.

### 7. Daten bereitstellen

Datensätze liegen **nicht** im Repository. Lege sie lokal im Ordner `data/` ab.
Der Datensatz ist aktuell nicht enthalten. Die unten beschriebenen synthetischen
Fixtures prüfen die Pipeline; ihre Ergebnisse messen keine reale Erkennungsleistung.

## Projektstruktur

```
ImageDetector/
├── src/                  # Wiederverwendbarer Code
│   ├── model/            # Modellarchitektur
│   ├── data/             # Dataset, Transforms, Preprocessing
│   ├── training.py       # Trainingslogik
│   └── evaluation.py     # Evaluation und Metriken
├── scripts/              # Einstiegspunkte (im Terminal starten)
│   ├── train.py
│   ├── eval.py
│   └── inference.py
├── configs/              # Hyperparameter und Einstellungen (YAML)
├── notebooks/            # Jupyter Notebooks zum Explorieren
├── data/                 # Datensätze (nicht in Git)
├── logs/                 # Logs und Checkpoints (nicht in Git)
├── .env.example          # Vorlage für lokale Umgebungsvariablen
├── requirements.txt
└── README.md
```

Die Scripts bleiben schlank, die eigentliche Logik liegt in `src/`.

## Training und Experimente

Alle Befehle im Projektordner und mit aktivierter `.venv` ausführen:

```bash
# Ohne Datensatz: zwei CPU-Epochen auf kleinen, deterministischen Fixtures.
python -m scripts.train experiment=smoke
python -m scripts.train experiment=smoke model=dual_resolution

# Vergleich mit drei Seeds; Hydra erzeugt für jede Kombination einen eigenen Lauf.
python -m scripts.train --multirun experiment=smoke model=baseline,dual_resolution seed=17,29,43
```

Hydra komponiert `configs/config.yaml` mit den Gruppen `model`, `data`, `spectral`
und optional `experiment`. CLI-Overrides werden mitgespeichert. Neue Läufe erhalten
eigene Verzeichnisse unter `outputs/`, Sweeps unter `multirun/`. Ein vorhandenes
Experimentverzeichnis wird nicht überschrieben. `hydra.run.dir=outputs/mein-lauf`
legt einen bekannten Ausgabeort fest; beim nächsten Lauf einen neuen Namen verwenden.

| Einstellung | Verhalten |
| --- | --- |
| `model=baseline` | Drei einfache Conv/BatchNorm/ReLU-Stufen, Average Pooling und Klassifikationskopf; nur das gesamte Bild. |
| `model=dual_resolution` | Zwei unabhängige Encoder auf Basis der vorhandenen ResNet-Implementierung, lokale Mittelwertbildung, Feature-Konkatenation und Klassifikationsköpfe. Beide Encoder starten ohne Vortraining. |
| `model=dual_imagenet` | Gleiche Fusion, aber globaler torchvision ResNet-18 mit ImageNet-Gewichten; dieser Encoder bleibt zunächst eingefroren. Erstmaliges Training kann Gewichte herunterladen. |
| `spectral=none` | Rein überwachter Detektor ohne spektrales Referenzmodell. |
| `spectral=reconstruction` | Zusätzliche Fehlermerkmale eines separat vortrainierten Rekonstruktionsmodells. |
| `spectral=flow` | Zusätzliche Fehlermerkmale eines separat vortrainierten Conditional-Flow-Matching-Modells. |

Die Encoder verwenden explizite `nn.Module`-Klassen und den vorhandenen
Residualblock. Der globale View erhält ein Letterboxing auf 224×224; lokale
256×256-Patches werden ohne Skalierung ausgeschnitten. `views.patches=4` aktiviert
vier Patches. Training verwendet reproduzierbare Zufallspositionen, Evaluation
das Zentrum beziehungsweise die vier Ecken. Doppelte Positionen werden maskiert.
Ist ein Bild zu klein, verwendet der Detektor den mittrainierten globalen Kopf.
Die lokalen Encoder reduzieren die Auflösung nicht schon in der ersten Faltung.

### Spektrale Referenzen vortrainieren

Die Referenzen werden ausschließlich mit den realen Bildern des Trainingssplits
trainiert. Die Kondition enthält nur tiefe Frequenzen, das Ziel ausschließlich
den komplementären hochfrequenten Anteil. So kann das Modell das Ziel nicht aus
einer unverdeckten Kopie im Konditionspfad ablesen.

```bash
python -m scripts.pretrain experiment=smoke spectral=reconstruction hydra.run.dir=outputs/reconstruction-ref
python -m scripts.train experiment=smoke model=dual_resolution spectral=reconstruction pretrained_reference=outputs/reconstruction-ref/best.pt hydra.run.dir=outputs/reconstruction-detector

python -m scripts.pretrain experiment=smoke spectral=flow hydra.run.dir=outputs/flow-ref
python -m scripts.train experiment=smoke model=dual_resolution spectral=flow pretrained_reference=outputs/flow-ref/best.pt hydra.run.dir=outputs/flow-detector
```

Rekonstruktion minimiert den quadratischen Fehler des skalierten hochfrequenten
Residuals. Flow Matching interpoliert zwischen projiziertem Gaußrauschen und diesem
Residual und lernt das Geschwindigkeitsfeld `target - noise`. Zur Detektion werden
pro Farbkanal deterministische Fehlermerkmale mit festen Noise-Probes bei
`t = 0.25, 0.5, 0.75` berechnet. Das sind **keine Likelihoods**. Die Methode ist eine
eigene Forschungsvariante und keine Reproduktion von SPAI. `SpectralReference.sample`
bietet zusätzlich Euler-Sampling zur Inspektion; die Detektion braucht keine ODE-Lösung.

Im Detektor bleibt die gesamte spektrale Referenz eingefroren. Ihre Merkmale werden
mit den lokalen ResNet-Merkmalen kombiniert. Der globale und der lokale
Klassifikationskopf erhalten Hilfsverluste (`loss.global_aux=0.2`, `loss.local_aux=0.1`).
Die Referenz und der Detektor müssen dasselbe Dateninventar, dieselbe lokale
Patchgröße und dieselbe Spektralkonfiguration verwenden. Auch bei real-only
Vortraining deshalb dasselbe vollständige Manifest angeben. Dessen generierte
Bilder werden beim Referenztraining herausgefiltert.

### GenImage-Teilmenge (z. B. ADM) vorbereiten

GenImage speichert generierte Bilder als 256×256-PNG und reale ImageNet-Bilder als
JPEG unterschiedlicher Größe. Ein Detektor kann beide Klassen deshalb allein an
Dateiformat, JPEG-Artefakten, Größe oder Seitenverhältnis trennen. Der Builder
entfernt diese Abkürzungen: Reale Bilder werden einmal dekodiert und mittig auf
256×256 zugeschnitten (8×8-JPEG-Raster bleibt ausgerichtet; Graustufen- und zu
kleine Bilder werden übersprungen). Generierte Bilder werden einmal als JPEG mit
Qualität und Chroma-Subsampling aus der Verteilung der realen Bilder desselben
Splits komprimiert. Beide Klassen werden verlustfrei als PNG gespeichert.

Split-ZIPs (`.z01` … `.zip`) werden direkt gelesen, ohne sie zusammenzufügen.
`train` und `val_id` stammen disjunkt aus GenImage-`train/`, `test_id` aus GenImage-`val/`.

```bash
python -m scripts.prepare_genimage C:/data/ADM/imagenet_ai_0508_adm.zip data/genimage_adm --train 4000 --val 1000 --test 1000
python -m scripts.train experiment=adm model=baseline hydra.run.dir=outputs/adm-baseline
python -m scripts.eval experiment=adm checkpoint=outputs/adm-baseline/best.pt hydra.run.dir=outputs/adm-baseline-eval
```

Verbleibender Unterschied: Reale Bilder können mit anderen Quantisierungstabellen
als den IJG-Standardtabellen von Pillow kodiert sein.

### Echte Daten anschließen

`data/samples.jsonl` enthält einen JSON-Datensatz pro Zeile. Pfade beziehen sich auf
das Verzeichnis der Manifestdatei. Beispielzeilen (durch existierende Bilder ersetzen):

```jsonl
{"path":"raw/camera/image1.jpg","label":0,"split":"train","group_id":"photo-1","source_domain":"camera-collection-a"}
{"path":"raw/generated/image2.png","label":1,"split":"train","group_id":"generation-2","source_domain":"generated-collection-a","generator_family":"generator-a"}
{"path":"raw/camera/image3.jpg","label":0,"split":"val_id","group_id":"photo-3","source_domain":"camera-collection-b"}
{"path":"raw/generated/image4.png","label":1,"split":"val_id","group_id":"generation-4","source_domain":"generated-collection-b","generator_family":"generator-b"}
```

Labels sind `0=real`, `1=generated`. Training und Validierung sind erforderlich;
für eine abschließende Evaluation zusätzlich einen `test_id`-Split bereitstellen.
Zusammengehörende Bilder und Varianten müssen dieselbe `group_id` erhalten.
Die Pipeline prüft Gruppenüberschneidungen und identische Dateiinhalte zwischen
Splits. Sie berechnet SHA-256 für jede Quelldatei, prüft optional angegebene
`sha256`-Werte und protokolliert einen Inventar-Fingerprint. Das liest beim Start
alle kodierten Dateien einmal, lädt aber nicht alle dekodierten Bilder in RAM.
Near-Duplicate-Erkennung und die sachliche Prüfung der Herkunft bleiben Aufgaben
der Datensatzvorbereitung; Labels werden niemals aus Dateinamen abgeleitet.

```bash
python -m scripts.train model=baseline data.manifest=/data/samples.jsonl
python -m scripts.train model=dual_imagenet data.manifest=/data/samples.jsonl
python -m scripts.pretrain spectral=flow data.manifest=/data/samples.jsonl

# Für Generator-Generalisation einen wirklich getrennten Entwicklungssplit wählen.
python -m scripts.train model=dual_resolution data.manifest=/data/samples.jsonl data.val_split=val_ood data.test_split=test_ood
```

Die Bilder werden direkt in ihrer vorhandenen Auflösung gelesen, EXIF-orientiert
und nach RGB konvertiert; Transparenz wird auf Weiß gelegt. Bereits auf 128×128
verkleinerte Daten liefern keine nativen 256×256-Patches. Der ältere Tar-Konverter
unten bleibt für bestehende Anwendungen verfügbar; die neue Trainingspipeline
verwendet native Bilddateien und das explizite Manifest.

Detektortraining besucht jedes Bild einmal pro Epoche. Verlustgewichte gleichen Klassen,
Generatorfamilien beziehungsweise reale Quelldomänen und Bildgruppen aus.
Validierungsmetriken bleiben ungewichtet. Der Detektor wählt `best.pt` standardmäßig
nach mittlerer AUROC pro Generatorfamilie; frühes Stoppen erfolgt nach drei Epochen
ohne Verbesserung. Die spektrale Referenz wählt nach dem deterministischen
Validierungs-Probe-Fehler. Rekonstruktionsqualität muss anschließend anhand der
Detektorleistung überprüft werden. Alle Optimierer verwenden zunächst AdamW mit
konstanter Lernrate; Hyperparameter stehen in Hydra-Konfigurationen.

### MLflow und Reproduzierbarkeit

Training, Vortraining, Evaluation und Inferenz erzeugen MLflow-Runs. Standardmäßig
liegen SQLite-Trackingdaten und Artefakte lokal unter `logs/mlflow/`. Ein Server ist
mit `tracking.uri=http://localhost:5000` wählbar. Die lokale Oberfläche startet mit:

```bash
mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db
```

Jeder Trainingslauf speichert aufgelöste Konfiguration, Hydra-Overrides,
Paketversionen, Quellcode-Snapshot mit Hash, Git-Revision/Dirty-Status, Dateninventar,
Seed, Metriken sowie `best.pt` und `last.pt`. Checkpoints enthalten Modell,
Optimierer und RNG-Zustände von Python, NumPy und PyTorch. Spektrale Gewichte sind
im finalen Detektorcheckpoint enthalten; Evaluation und Inferenz laden keine
externen Modellgewichte nach.

```bash
# Dieselbe Konfiguration wie beim ersten Lauf; epochs ist die gesamte Zielzahl.
python -m scripts.train experiment=smoke epochs=4 resume=outputs/alter-lauf/last.pt
python -m scripts.eval checkpoint=outputs/flow-detector/best.pt
python -m scripts.inference checkpoint=outputs/flow-detector/best.pt input=/data/example.jpg
```

Fortsetzung erfolgt an Epochengrenzen in einem neuen, verknüpften MLflow-Run.
`best.pt` muss neben `last.pt` liegen. Änderungen an Daten, Modell, relevanten
Trainingsparametern oder am Quellcode werden bei exakter Fortsetzung abgewiesen.
Die Tests prüfen bitgleiche CPU-Modellzustände nach Fortsetzung gegenüber einem
ununterbrochenen Lauf. Gleiche Resultate über unterschiedliche Hardware- oder
Bibliotheksversionen hinweg sind damit nicht zugesichert.

Evaluation übernimmt Daten und Views aus dem Checkpoint. `eval_split=test_ood`
wählt einen anderen vorhandenen Split; `eval_manifest=/data/external.jsonl`
erlaubt ein externes Manifest, das nur den gewählten Split enthalten darf.
Gespeichert werden AUROC, Average Precision, Balanced Accuracy, FPR/TPR bei 0.5,
Coverage und Einzelvorhersagen. Inferenz akzeptiert eine Datei oder ein Verzeichnis
und schreibt `predictions.jsonl` mit Score, unkalibriertem Label, gewähltem Zweig
und Crop-Koordinaten. Fehlerhafte Dateien erhalten einen Fehlerdatensatz und führen
zu einem nicht erfolgreichen Exit-Code.

Noch nicht umgesetzt sind Wahrscheinlichkeitskalibrierung, automatische
Familien-Split-Erstellung, Bootstrap-Konfidenzintervalle, DDP und DINO/CLIP-Adapter.
Die getesteten ausführbaren Encoder sind das eigene CNN/ResNet und torchvision
ResNet-18. Der vollständige Forschungsplan steht in
[doc/implementation_proposal.md](doc/implementation_proposal.md).

## Bilder für PyTorch konvertieren

Setze in `.env` den Pfad zum Rohdatensatz (relative Pfade beziehen sich auf den Projektordner):

```dotenv
DATA_DIR="/path/to/raw/images"
```

```bash
python -m scripts.convert_images
```

Der Konverter durchsucht Unterordner und ZIP-Dateien. Alle von Pillow unterstützten
Bildtypen werden mit EXIF-Ausrichtung in **128×128 RGB PNG** konvertiert. Die Größe
wird ohne Zuschnitt angepasst; dabei kann sich das Seitenverhältnis verändern.
Transparenz erhält einen weißen Hintergrund, Animationen verwenden das erste Bild.
Die Quelldateien bleiben unverändert. ZIP-Dateien innerhalb anderer ZIP-Dateien
werden nicht rekursiv geöffnet.

Die Ausgabe liegt standardmäßig in `data/processed/`: WebDataset-kompatible
`images-000000.tar`-Shards mit jeweils bis zu 10.000 Bildern und JSON-Quellmetadaten,
`manifest.json` mit Zählwerten und `errors.jsonl` mit übersprungenen defekten Bildern
oder unvollständigen Archiven. Ein nichtleeres Ausgabeverzeichnis wird nicht
überschrieben. Wenn kein Bild konvertiert wurde, endet das Script mit Exit-Code 1.

```bash
python -m scripts.convert_images --output-dir data/processed-v2 --shard-size 1000
```

Split-ZIPs benötigen die vollständige `.zip` und alle gleichnamigen `.z01`, `.z02`,
… Dateien im selben Ordner. Pro Split-Archiv wird temporär ein
zusammengefügtes ZIP gespeichert; ausreichend freien Speicher einplanen und bei
Bedarf `--temp-dir /path/to/scratch` angeben. `.part`-Downloads und einzelne
`.z01` ohne finale `.zip` werden mit einer Warnung übersprungen.

Streaming für das Training, ohne den gesamten Datensatz in RAM zu laden:

```python
from torch.utils.data import DataLoader
from src.data.streaming import StreamingImageDataset

dataset = StreamingImageDataset("data/processed")
loader = DataLoader(dataset, batch_size=64, num_workers=2)
for images, metadata in loader:
    # Float32 [batch, 3, 128, 128], Wertebereich [0, 1]
    # metadata["source"] enthält die ursprünglichen Pfade.
    pass  # Hier Modelltraining und eigene Label-Zuordnung ergänzen.
```

Shards werden zwischen Workers und gegebenenfalls initialisierten PyTorch-DDP-Ranks
aufgeteilt. Für paralleles Laden genügend Shards erzeugen. Der Reader liest
sequenziell ohne Shuffle; Labels werden nicht aus Ordnernamen abgeleitet.
Für DDP vor dem Dataset die Prozessgruppe initialisieren und gleiche Trainingsschritte
pro Rank sicherstellen, da die Sample-Anzahl pro Rank unterschiedlich sein kann.
Optional nimmt `StreamingImageDataset(..., transform=...)` einen PIL-Transform an.

Tests: `python -m unittest discover -s tests -v`.

## Zusammenarbeit

- Neue Abhängigkeit installiert? Sie in `requirements.txt` eintragen und committen.
- Niemals committen: `.env`, `.venv/`, Datensätze, Modellgewichte (`*.pt`, `*.pth`, `*.ckpt`), `logs/`.
- Vor dem Arbeiten aktuellen Stand holen: `git pull`.
- Pro Aufgabe einen eigenen Branch anlegen und per Pull Request zusammenführen *(falls ihr so arbeitet)*.

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Befehl vom Projektroot starten, `PYTHONPATH=.` in `.env` prüfen und Scripts mit `python -m scripts.<name>` ausführen. |
| Pakete werden nicht gefunden | Prüfen, ob `.venv` aktiviert ist (Prompt zeigt `(.venv)`) und in VS Code der richtige Interpreter gewählt ist. |
| Installation einer ML-Bibliothek schlägt fehl | Python-Version prüfen, vermutlich unterstützt die Bibliothek sie noch nicht (siehe Voraussetzungen). |
