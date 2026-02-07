# Email-Classification
# 📧 Classificatore Email Machine Learning (Ensemble)

## 📋 Project Overview
Questo progetto implementa un sistema di classificazione e-mail professionale che utilizza algoritmi di Machine Learning per distinguere tra diverse categorie di messaggi.

**Modelli utilizzati:**
* **SVC (Support Vector Classifier)**: per una classificazione precisa su base statistica.
* **Random Forest**: per un approccio robusto basato su alberi di decisione.
* **Ensemble Method**: un sistema che combina i due modelli precedenti per ridurre l'errore e aumentare l'affidabilità.

L'interfaccia utente è interamente sviluppata in **Streamlit**.

---

## ⚙️ Installation Instructions
Per eseguire questo progetto è necessario avere **Anaconda** installato. 
Apri il **Prompt di Anaconda** e installa le librerie mancanti con questo comando:

```bash
pip install streamlit scikit-learn pandas numpy
```
Usage Instructions (Come avviare il codice)
Per lanciare l'applicazione, segui questi passaggi nel Prompt di Anaconda:

Entra nella cartella del progetto:

```Bash
cd percorso/della/tua/cartella
```
Esegui il comando di avvio per Streamlit:
```Bash
streamlit run app.py
