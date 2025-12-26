# 💧 Water Potability: Predicting Safety with Random Forest

I built this project to see if Machine Learning can actually tell us if water is safe to drink based on chemical sensors. It's a tricky problem because water quality isn't just "black and white"—it's a complex mix of pH, minerals, and chemicals.

### 🚀 [Live Demo on Streamlit](https://github.com/Baljit26-cloud/Water_Quality_ML_Project)

---

## 🛠️ The "Human" Engineering Process

Most people just run a model and hope for the best. I spent most of my time on the **data prep**, which is where the real work happens.

### 1. Dealing with the "Noise" 🧹
* **Imputation:** The dataset had a lot of missing values in pH and Sulfate. Instead of deleting those rows, I used **Mean Imputation** to keep the dataset size large enough for the model to learn.
* **Outlier Cleanup:** I used the **Percentile Method** to chop off the top and bottom 1% of the pH and Sulfate data. Why? Because environmental sensors often give "garbage" readings that can confuse a model.

### 2. Picking the Right Features 🎯
I didn't use all the columns. I ran a **Feature Importance** check and found that 5 things really matter:
* **Sulfate, pH, Hardness, Chloramines, and Solids.**
By focusing on these, the model became faster and less prone to "distractions" from irrelevant data.

### 3. Balancing the Scales ⚖️
My data had way more "Safe" samples than "Unsafe" ones. If I didn't fix this, the model would just guess "Safe" every time to get a high score. I used `class_weight="balanced"` to force the model to take the "Unsafe" cases seriously.

---

## 📈 Why 70.43%? (The Reality Check)

You might see projects claiming 90%+ accuracy, but here’s why **70% is more honest** for this problem:

1.  **Complexity:** Water chemistry is non-linear. You can have "perfect" pH but high lead levels (which might not be in the dataset). 70% shows the model is learning the trends without being "cocky."
2.  **Overfitting vs. Generalization:** I could have pushed for 90%, but the model would have failed the moment it saw a real-world water sample. I chose **stability** over a fake high score.
3.  **Data Imbalance:** Because I used "balanced" weights, the accuracy is lower, but the **reliability** for detecting "Unsafe" water is much higher.

---

## 📊 Model Performance


The Confusion Matrix shows that the model is now actually identifying "Not Potable" samples instead of just ignoring them!

## 📂 What's in the Repo?
* `app.py`: The UI for the Streamlit app.
* `water_quality_final_model.py`: My training script.
* `random_forest_model.pkl`: The saved model weights.
* `requirements.txt`: The libraries you need to run this.

---

## 💡 Key Takeaway
This project taught me that **cleaning the data is 90% of the job.** Building the model is easy; making sure the model isn't lying to you about its accuracy is the hard part.
