# 🛡️ Social Engineering Email Detection System

> **Course**: CENG3544 - Computer and Network Security
> **Instructor**: Dr. Enis Karaarslan
> **Student**: Vuslat Sülbiye Türk

This project presents a machine learning-based approach to detect **social engineering tactics in email content**. Instead of identifying spam, this system targets manipulative techniques like **urgency, authority, impersonation, persuasion**, and more—offering a smarter, psychological filter.

## 📌 Project Overview
With the rising number of cyber-attacks leveraging social engineering techniques, especially via email, it's crucial to develop intelligent systems that can detect such threats before a human falls victim. This project builds a machine learning pipeline that classifies emails based on whether they exhibit the following characteristics:

Impersonation (e.g., pretending to be a boss or official)

Urgency (e.g., "You must act now!")

Authority (e.g., citing a legal or company authority to pressure)

Persuasion (e.g., manipulative language, incentives)

Safe Email (benign, harmless emails)

Unlike spam filters, this classifier attempts to analyze the psychological tactics embedded in the message — making it more targeted to social engineering detection.

## 🗃️ Dataset Description
The dataset used in this project contains 7600+ manually labeled emails. Each email is tagged with one or more labels corresponding to different social engineering traits. Some key aspects:

Emails may belong to multiple categories (multi-label classification).

Each label reflects a tactic, not a topic.

Additional legitimate emails were added to ensure balance and contrast.

💡 Example: An email could be simultaneously labeled as both impersonation and urgency.

Label distribution was analyzed visually to understand data imbalance. Special techniques were applied to reduce bias during training.

## ⚙️ Workflow

### 🔍 1. Data Preprocessing

* Cleaned the text: lowercased, removed punctuation/special characters.
* Applied **TF-IDF vectorization** to convert text into numerical values.
* Exploded label lists to understand frequency.
* Split into **training (80%) and testing (20%)** sets.

### 🧠 2. Model Training

| Model                       | Description                                                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Linear SVC**              | A fast and efficient classifier good for high-dimensional data like text. Performs well on small and large datasets. |
| **Random Forest**           | An ensemble of decision trees. It reduces overfitting and works well even with label imbalance.                      |
| **Logistic Regression**     | A baseline model that performs linear classification. Easy to interpret and fast to train.                           |
| **Multinomial Naive Bayes** | Often used in text classification. Assumes feature independence, which works well for some NLP tasks.                |

All models used the **One-vs-Rest** method to support multi-label output.

## 📈 Evaluation

We used the following metrics to evaluate each model's performance:

* **Precision**: How many predicted positives were actually correct?
* **Recall**: How many actual positives did the model catch?
* **F1-score**: A balance between precision and recall.
* **Accuracy per label**: How well each class is predicted, one by one.
* **Confusion Matrix**: A table showing correct vs incorrect predictions.
* **PR Curves** (Precision-Recall Curves): Help visualize trade-offs between catching all positives (recall) and making sure predictions are reliable (precision).
* **Cross-validation**: Evaluated model consistency by training/testing on different data splits.
* **Overfitting Plots**: Compared training vs testing accuracy to ensure models generalize.

## 🏆 Results Summary

* ✅ **Best performing models**: `Linear SVC` and `Random Forest`
* Naive Bayes was fast but struggled with label imbalance.
* Logistic Regression had stable results but lower recall on rare labels.
* Linear SVC gave strong precision and recall balance across labels.
* Random Forest was especially robust and adaptable to complex patterns.

Visualizations like PR curves, accuracy histograms, and confusion matrices helped interpret and compare model behavior clearly.

## 🔧 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Google Colab
* Git + GitHub
* 
## 🚀 How to Run

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Then, open the `security_proje_code.ipynb` notebook in [Google Colab](https://colab.research.google.com/) and run all cells in order.

---

## 🔮 Future Work

* Add more examples for underrepresented classes (like impersonation).
* Test with real-world email inboxes.
* Integrate sender metadata and subject line features.
* Experiment with **deep learning models** like BERT or LSTM.

---

## 🙏 Acknowledgments

I’d like to thank **Dr. Enis Karaarslan** for his guidance, and **ChatGPT** for supporting the coding, testing, and explanation process. I also acknowledge my own persistence in completing this project from start to finish.
