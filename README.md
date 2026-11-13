# ImageDetector

Kurze Beschreibung des Projekts: *(hier eintragen, z. B. Ziel, Datensatz, Modellart)*

## Voraussetzungen

- **Python 3.12 oder 3.13** *(Version in der Gruppe abstimmen und hier festhalten; sehr neue Versionen wie 3.14 werden von ML-Bibliotheken oft noch nicht unterstützt)*
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
*(Hier eintragen: Woher bekommt man die Daten? Link, Netzlaufwerk, Download-Anleitung.)*

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

## Benutzung

Alle Befehle im Projektordner und mit aktivierter `.venv` ausführen:

```bash
python -m scripts.train       # Training starten
python -m scripts.eval        # Modell evaluieren
python -m scripts.inference   # Vorhersagen auf neuen Bildern
```

Einstellungen (Lernrate, Batchgröße, Pfade usw.) werden in `configs/config.yaml` geändert, nicht im Code.

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
