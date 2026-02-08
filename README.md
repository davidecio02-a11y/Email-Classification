# Email-Classification
# 📧 Classificatore Email Machine Learning (Ensemble)

## 📋 Project Overview
Questo progetto implementa un sistema di classificazione e-mail che utilizza algoritmi di Machine Learning per distinguere automaticamente tra **Spam** (messaggi indesiderati) e **Ham** (messaggi autentici).

**Modelli utilizzati:**
* **SVC (Support Vector Classifier)**: per una classificazione precisa su base statistica.
* **Random Forest**: per un approccio robusto basato su alberi di decisione.
* **Naive Bayes**
* **Logistic Regression**
**Tecniche di Ensemble utilizzate:**
Per migliorare le performance e la robustezza della predizione, i modelli sopra citati vengono combinati attraverso:
1.  **Voting Classifier**: Una strategia che aggrega le previsioni di tutti i modelli per scegliere la classe più votata.
2.  **Stacking Classifier**: Un approccio meta-modello che impara a combinare in modo ottimale le previsioni dei singoli algoritmi.

L'interfaccia utente è interamente sviluppata in **Streamlit**.

---

## ⚙️ Installation Instructions
Per eseguire questo progetto è necessario avere **Anaconda** installato iserendo nel codice il percorso dove si trova il dataset spam. 
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
