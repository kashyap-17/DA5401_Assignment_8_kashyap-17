### **Assignment: DA5401 A8: Ensemble Learning for Complex Regression**

**Student Information**

  * **Name:** Kashyap Shankar Iyer
  * **Roll Number:** DA25C012

-----

**Documentation**

This submission consists of a single Jupyter Notebook. Its purpose is to demonstrate a comprehensive understanding of ensemble learning techniques—**Bagging, Boosting, and Stacking**—applied to a real-world time-series regression problem: forecasting bike rental demand. The analysis focuses on establishing a strong baseline, implementing rigorous hyperparameter tuning, and explaining how each ensemble technique addresses the **bias-variance trade-off** to minimize the final prediction error, measured by **Root Mean Squared Error (RMSE)**. The notebook uses a meticulous, evidence-based approach to justify all modeling decisions, including feature engineering corrections.

-----

**Folder Structure**

```
.
└── Kashyap_DA25C012_A8_Ensemble_Learning.ipynb
```

-----

**Dataset Information**

The analysis utilizes the **Bike Sharing Demand Dataset (Hourly Data)** from the UCI Machine Learning Repository.

  * **Features:** Hourly variables including weather (temperature, humidity, windspeed), time factors (hour, month, weekday), and binary flags (`workingday`, `holiday`).
  * **Target:** The total count of rented bikes (`cnt`).
  * **Data Preparation:** The data is chronologically split (80% train / 20% test) to ensure a realistic time-series forecasting scenario.

-----

**Notebook Structure and Content**

The notebook is organized into the following four parts, detailing the complete modeling process:

  * **Part A: Data Preprocessing and Baseline**
    This section establishes the foundation for the assignment.

    1.  **Feature Engineering:** Implementation of **Cyclical Encoding** (`sin`/`cos` transformation) for time-based features (`hr`, `mnth`, `weekday`), which is crucial for capturing periodicity, correcting a methodological weakness of simple One-Hot Encoding.
    2.  **Split & Scale:** Chronological splitting of data and standard scaling of continuous features.
    3.  **Baseline Models:** Training and evaluating a **Linear Regression** model and a shallow **Decision Tree Regressor (max\_depth=6)** to set the performance benchmark (RMSE).

  * **Part B: Bagging and Variance Reduction**
    This section focuses on reducing model variance.

    1.  **Simple Bagging:** Implementation of a `BaggingRegressor` using the baseline Decision Tree, demonstrating an initial reduction in RMSE.
    2.  **Tuned Bagging (Model 4):** Implementation of `GridSearchCV` on the `BaggingRegressor` to find the optimal ensemble of deep, low-bias trees, establishing a strong, variance-reduced benchmark.

  * **Part C: Boosting and Bias Reduction**
    This section focuses on reducing model bias, aiming to beat the Bagging benchmark.

    1.  **Initial GBR:** Training a default `GradientBoostingRegressor`, used to highlight the need for parameter tuning.
    2.  **Deep Tuning (Model 7):** Rigorous application of `GridSearchCV` to find the optimal "slow and steady" parameters (`learning\_rate`, `n\_estimators`, `subsample`) for the GBR, resulting in the lowest RMSE for a single ensemble type.

  * **Part D: Stacking and Final Synthesis**
    This section combines the best models.

    1.  **Stacking Implementation:** Defining three diverse Base Learners (Tuned Bagging, Tuned GBR, and a tuned `KNeighborsRegressor`) and a `Ridge` Meta-Learner.
    2.  **Tuned Stacking:** Implementation of `GridSearchCV` on the `StackingRegressor` to find the optimal combination.
    3.  **Final Conclusion:** A summary table and detailed discussion identifying the **Tuned Gradient Boosting Regressor** as the champion model (lowest RMSE), with a final analysis explaining *why* the bias-reduction strategy (Boosting) ultimately outperformed the variance-reduction strategy (Bagging) and the composite Stacking model.