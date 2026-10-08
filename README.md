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