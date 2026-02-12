# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Student:** Harkirat Singh 

**Roll Number:** U20230082

**Course:** Reinforcement Learning Fundamentals  

---

## Problem Formulation

### Environment Structure
- **Contexts:** 3 unique user types (user_1, user_2, user_3)
- **Bandits/Arms:** 4 news categories per context (ENTERTAINMENT, EDUCATION, TECH, CRIME)
- **Total Arms:** 12 (3 contexts × 4 categories)
- **Horizon:** T = 10,000 steps per context (30,000 total)

### Arm Mapping
```
Arms 0-3:     ENTERTAINMENT, EDUCATION, TECH, CRIME     → user_1
Arms 4-7:     ENTERTAINMENT, EDUCATION, TECH, CRIME     → user_2
Arms 8-11:    ENTERTAINMENT, EDUCATION, TECH, CRIME     → user_3
```

---

## Methodology

### 1. Data Preprocessing
- **User Data:** Loaded `train_users.csv` containing 2000 users with 33 features
- **Test Data:** Loaded `test_users.csv` with 2000 unlabeled users
- **News Articles:** Loaded `news_articles.csv` with 209,527 articles across 42 categories
- **Missing Value Handling:** Applied median imputation (34.9% missing in 'age' column)
- **Feature Engineering:** 
  - Encoded categorical variables (browser_version, region_code, subscriber)
  - Standardized all numerical features using StandardScaler
  - 31 features used for classification after preprocessing

### 2. User Classification
- **Algorithm:** Random Forest Classifier (100 estimators, random_state=42)
- **Training:** 80% of `train_users.csv` (1600 samples)
- **Validation:** 20% of `train_users.csv` (400 samples)
- **Performance:** 
  - Training Accuracy: **97.44%**
  - Validation Accuracy: **90.00%**
  - Overfitting: 7.44%
  - This classifier serves as the **context detector** for the bandit system

**Classification Report (Validation Set):**
```
              precision    recall  f1-score   support
      user_1     0.8971    0.8592    0.8777       142
      user_2     0.9766    0.8803    0.9259       142
      user_3     0.8309    0.9741    0.8968       116

    accuracy                         0.9000       400
   macro avg     0.9015    0.9045    0.9001       400
weighted avg     0.9061    0.9000    0.9004       400
```

**Why I used Random Forest Classifier:**
- Robust to heterogeneous and noisy features; captures non-linear interactions
- Ensemble averaging reduces overfitting and yields stable performance
- `n_estimators=100` balances training speed and model accuracy
- Provides feature importance estimates for interpretability
- Best performer among three tested models (RF, Decision Tree, Logistic Regression)

**Model Comparison:**
```
              Model  Training Accuracy  Validation Accuracy  Overfitting
      Random Forest           0.9744                0.9000       0.0744
      Decision Tree           0.9225                0.8650       0.0575
Logistic Regression           0.8381                0.8200       0.0181
```

### 3. Contextual Bandit Algorithms

#### 3.1 Epsilon-Greedy Strategy
**Algorithm Description:**
- With probability ε: select random arm (exploration)
- With probability (1-ε): select best arm so far (exploitation)
- Maintains count and cumulative reward per arm
- Separate Q-tables maintained for each user context

**Hyperparameter Tuning:**
- Tested ε values: 0.01, 0.1, 0.3
- **Results:**
  - ε=0.01: Final Avg Reward = **5.3048** (Best for Epsilon-Greedy)
  - ε=0.1:  Final Avg Reward = 4.8287
  - ε=0.3:  Final Avg Reward = 3.7675

**Key Observations:**
- Lower ε values achieve better long-term rewards
- ε=0.01 balances exploration and exploitation effectively
- Higher ε (0.3) causes excessive random exploration, degrading performance significantly

#### 3.2 Upper Confidence Bound (UCB)
**Algorithm Description:**
- Maintains optimistic estimate: avg_reward + C×√(ln(t) / count)
- Exploration bonus decreases over time as uncertainty reduces
- Theoretically principled approach with logarithmic regret bounds
- Context-specific Q-value updates

**Hyperparameter Tuning:**
- Tested C values: 0.5, 1.0, 2.0
- **Results:**
  - C=2.0: Final Avg Reward = **5.3687** (Best Overall)
  - C=0.5: Final Avg Reward = 5.3565
  - C=1.0: Final Avg Reward = 5.3548

**Key Observations:**
- **UCB outperforms all other algorithms** across all hyperparameter settings
- All C values achieve similar high performance (5.35-5.37)
- Higher C values (2.0) provide slightly better exploration-exploitation balance
- Provides stable, smooth convergence trajectory

#### 3.3 SoftMax (Boltzmann Exploration)
**Algorithm Description:**
- Probabilistic arm selection using softmax distribution
- Selects arms proportional to their expected rewards
- Temperature τ controls exploration level
- Separate probability distributions per context

**Hyperparameter Setting:**
- Tested τ value: 1.0
- **Result:**
  - τ=1.0: Final Avg Reward = **5.2039**

**Key Observations:**
- Competitive performance, close to UCB and Epsilon-Greedy (ε=0.01)
- Provides smooth probabilistic exploration
- More nuanced than ε-greedy's hard exploration cutoff

### 4. RL Simulation Methodology (Context-Specific Tracking)
**Simulation Architecture:**
- **Temporal Horizon:** T = 10,000 steps **per context** (30,000 total steps)
- **Training Approach:** Each algorithm trained on all 3 contexts separately
- **Sampler:** Initialized with roll number 82 using `rlcmab_sampler` package
- **Per-Context Tracking:** Rewards accumulated separately for each of 3 user contexts

**Key Implementation Details:**
- For each algorithm variant, 3 separate bandits maintained (one per context)
- Training process:
  1. For each context (user_1, user_2, user_3):
     - Run 10,000 simulation steps
     - At each step: select arm → sample reward → update Q-values
     - Track cumulative rewards and Q-value evolution
  2. Calculate average reward across all 30,000 steps
- Context-specific Q-value tables learned independently
- Final Q-values show clear arm preferences per context

**Example Q-values (Epsilon-Greedy ε=0.01):**
```
Context 0 (user_1): [ 1.859, -5.684,  3.689, -7.390]
Context 1 (user_2): [ 1.914,  4.632, -1.698, -1.425]
Context 2 (user_3): [ 7.788, -0.934, -0.334, -0.575]
```

**Reason for context-specific tracking:**
- Ensures each user type's learning is independently analyzed
- Reveals context-dependent algorithm performance
- Enables identification of best algorithm-hyperparameter combination
- Mimics real-world scenario where different user segments have different preferences

### 5. Recommendation Engine
**End-to-End Pipeline:**
1. **Classify:** Predict user context using trained Random Forest (90% accuracy)
2. **Select:** Use trained bandit policy to select optimal category based on learned Q-values
3. **Recommend:** Randomly sample article from selected category (from 209K article dataset)
4. **Output:** Return user_id, context, category, expected reward, article headline

**Features:**
- Supports all three algorithm variants (Epsilon-Greedy, UCB, SoftMax)
- Handles feature scaling and missing value imputation automatically
- Returns complete recommendation with article metadata
- Expected reward calculated from learned Q-values

---

## Evaluation & Results

### 5.1 Classification Accuracy
**Random Forest Performance:**
- **Training Accuracy:** 97.44%
- **Validation Accuracy:** 90.00%
- **Per-Class Accuracy:**
  - user_1: 85.92%
  - user_2: 88.03%
  - user_3: 97.41%

**Label Distribution (Training Set):**
```
user_2: 712 samples (35.60%)
user_1: 707 samples (35.35%)
user_3: 581 samples (29.05%)
```

### 5.2 RL Simulation Results (Aggregated Performance)
**Simulation Parameters:** T = 30,000 total steps (10,000 per context), Roll Number = 82

**Overall Algorithm Performance:**
```
Algorithm              Hyperparameter    Average Reward    Std Dev    Final Cumulative Avg
─────────────────────────────────────────────────────────────────────────────────────────
UCB                    C=2.0          *** 5.3687          2.0718           5.3687 (BEST)
UCB                    C=0.5               5.3565          2.0772           5.3565
UCB                    C=1.0               5.3548          2.0690           5.3548
Epsilon-Greedy         ε=0.01              5.3048          2.1762           5.3048
SoftMax                τ=1.0               5.2039          2.2401           5.2039
Epsilon-Greedy         ε=0.1               4.8287          2.8530           4.8287
Epsilon-Greedy         ε=0.3               3.7675          3.7861           3.7675
```

### 5.3 Context-Specific Analysis
**Final Q-Values by Context (showing learned arm preferences):**

**Epsilon-Greedy (ε=0.01):**
```
Context 0 (user_1): [ 1.86, -5.68,  3.69, -7.39]  → Prefers Arm 2 (TECH)
Context 1 (user_2): [ 1.91,  4.63, -1.70, -1.42]  → Prefers Arm 1 (EDUCATION)
Context 2 (user_3): [ 7.79, -0.93, -0.33, -0.58]  → Prefers Arm 0 (ENTERTAINMENT)
```

**UCB (C=2.0 - Best Overall):**
```
Context 0 (user_1): [ 1.73, -6.40,  3.70, -7.51]  → Prefers Arm 2 (TECH)
Context 1 (user_2): [ 1.56,  4.63, -1.43, -2.78]  → Prefers Arm 1 (EDUCATION)
Context 2 (user_3): [ 7.78, -2.50, -1.00,  0.84]  → Prefers Arm 0 (ENTERTAINMENT)
```

**SoftMax (τ=1.0):**
```
Context 0 (user_1): [ 1.98, -5.73,  3.66, -8.41]  → Prefers Arm 2 (TECH)
Context 1 (user_2): [ 1.87,  4.64, -2.35, -3.15]  → Prefers Arm 1 (EDUCATION)
Context 2 (user_3): [ 7.77, -2.56, -0.08, -0.34]  → Prefers Arm 0 (ENTERTAINMENT)
```

**Key Insight:** All algorithms converged to the same arm preferences per context, validating the learned policies.

### 5.4 Test Set Recommendations
**Generated 2,000 recommendations for test users:**
- Test users classified: **user_2** (1,365 users), **user_1** (390 users), **user_3** (245 users)
- Recommendations: **EDUCATION** for user_2, **TECH** for user_1, **ENTERTAINMENT** for user_3
- Expected rewards: user_2 → **4.63**, user_1 → **3.70**, user_3 → **7.78**

**Sample Recommendations:**
```
User ID    Context    Category        Expected Reward    Article Headline
─────────────────────────────────────────────────────────────────────────────────────
U4058      user_2    EDUCATION           4.63          Do Grades Really Reflect Rigor?
U1118      user_3    ENTERTAINMENT       7.78          13 Times Taylor Swift Showed Her Way...
U6555      user_1    TECH                3.70          Tim Cook Says EU Ruling On Apple's Irish Tax...
```

### 5.5 Analysis Plots
**Generated Visualizations:**
1. **Cumulative Reward Curves** - Learning progression over 10,000 steps per context
2. **Hyperparameter Comparison** - Bar charts comparing algorithm variants
3. **Confusion Matrix** - Classification performance visualization

---

## Key Findings & Insights

### 1. Algorithm Performance Ranking
1. **UCB (C=2.0):** Best overall performance (5.3687 avg reward) with stable convergence
2. **UCB (C=0.5, 1.0):** Nearly identical performance (~5.35), showing robustness
3. **Epsilon-Greedy (ε=0.01):** Competitive performance (5.3048), efficient exploitation
4. **SoftMax (τ=1.0):** Strong performance (5.2039), smooth probabilistic exploration
5. **Epsilon-Greedy (ε=0.1):** Moderate performance (4.8287), excessive exploration
6. **Epsilon-Greedy (ε=0.3):** Poor performance (3.7675), too much random exploration

### 2. Hyperparameter Sensitivity
- **UCB:** Low sensitivity to C values (all perform well 5.35-5.37)
- **Epsilon-Greedy:** High sensitivity to ε (performance drops 44% from ε=0.01 to ε=0.3)
- **Lower exploration parameters consistently perform better**
- UCB's principled exploration strategy outperforms ε-greedy's random exploration

### 3. Context-Dependent Learning
- All algorithms learned distinct arm preferences per user context
- **user_1** context: Prefers Arm 2 / TECH (Q-value ~3.7)
- **user_2** context: Prefers Arm 1 / EDUCATION (Q-value ~4.6)
- **user_3** context: Prefers Arm 0 / ENTERTAINMENT (Q-value ~7.8)
- Consistency across algorithms validates the learned policies

### 4. Classification Quality Impact
- 90% classification accuracy enables effective context detection
- High user_3 recall (97.41%) ensures premium users are correctly identified
- Misclassifications could lead to suboptimal recommendations

### 5. Convergence Characteristics
- **UCB:** Smooth, principled convergence with minimal oscillation
- **SoftMax:** Stable probabilistic learning trajectory
- **ε-Greedy:** Fast initial gains, then plateaus based on ε value
- All algorithms show clear learning within first 2,000-3,000 steps

### 6. Practical Implications
- **UCB recommended for production:** Best performance with minimal tuning required
- **Low exploration rates critical:** High exploration significantly degrades performance
- **Context-specific policies essential:** Different user segments need different strategies
- **Classification accuracy matters:** Better user segmentation → better recommendations

---

## Conclusion

This lab successfully implemented a contextual multi-armed bandit system for personalized news recommendations. The **UCB algorithm with C=2.0** achieved the best performance (5.37 average reward), demonstrating the effectiveness of principled exploration strategies. The Random Forest classifier achieved 90% accuracy in user segmentation, enabling context-aware recommendations. The system learned distinct arm preferences for each user context, validating the contextual bandit approach.
