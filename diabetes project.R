# Only install if missing
packages <- c(
  "tidyverse", "tidymodels", "themis", "DALEX", "xgboost", 
  "lightgbm", "bonsai", "discrim", "klaR", "vip", "shapviz"
)

to_install <- setdiff(packages, rownames(installed.packages()))
if(length(to_install)) install.packages(to_install)

library(tidyverse)
library(tidymodels)
library(themis)     # for SMOTE
library(DALEX)      # for model explanations
library(xgboost)    # XGBoost model
library(lightgbm)   # LightGBM model
library(bonsai)     # LightGBM in tidymodels
library(discrim)    # Naive Bayes
library(klaR)       # engine for Naive Bayes
library(vip)        # feature importance visualization
library(shapviz)    # SHAP values for interpretability

set.seed(123)

# Load and check structure
diabetes <- read_csv("diabetes.csv")

glimpse(diabetes)

# Convert 0/1 to No/Yes for readability
diabetes <- diabetes %>%
  mutate(Outcome = factor(Outcome, levels = c(0, 1), labels = c("No", "Yes")))

# Columns that can't realistically be zero
na_cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

# Replace 0s with NA
diabetes <- diabetes %>%
  mutate(across(all_of(na_cols), ~ na_if(., 0)))

# Check how many NAs we introduced
diabetes %>%
  summarise(across(all_of(na_cols), ~ sum(is.na(.))))

diabetes
summary(diabetes)

# Count missing values per column
diabetes %>%
  summarise(across(all_of(na_cols), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "Feature", values_to = "Missing_Count")

diabetes %>%
  count(Outcome) %>%
  mutate(Percent = n / sum(n) * 100) %>%
  ggplot(aes(x = Outcome, y = n, fill = Outcome)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(round(Percent, 1), "%")), vjust = -0.5) +
  labs(title = "Outcome Distribution", y = "Count", x = "Diabetes Diagnosis") +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  theme_minimal()

# Plot histograms for all numeric predictors
diabetes %>%
  pivot_longer(-Outcome, names_to = "Feature", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~ Feature, scales = "free") +
  theme_minimal() +
  labs(title = "Distributions of All Predictors")

cor_data <- diabetes %>%
  dplyr::select(-Outcome) %>%
  drop_na()

# Compute correlations
cor_matrix <- cor(cor_data)

# Visualize as heatmap
cor_matrix %>%
  as_tibble(rownames = "Feature1") %>%
  pivot_longer(-Feature1, names_to = "Feature2", values_to = "Correlation") %>%
  ggplot(aes(Feature1, Feature2, fill = Correlation)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap of Numeric Features")

set.seed(123)  # For reproducibility

split <- initial_split(diabetes, prop = 0.7, strata = Outcome)

train_data <- training(split)
test_data  <- testing(split)

# Check proportions
train_data %>% count(Outcome)
test_data %>% count(Outcome)

# Define recipe
rec <- recipe(Outcome ~ ., data = train_data) %>%
  step_impute_median(all_predictors()) %>%      # Impute missing with median
  step_normalize(all_numeric_predictors()) %>%  # Scale & center
  step_smote(Outcome)                           # Balance outcome classes
# Prep the training recipe
rec_trained <- prep(rec)
train_prepped <- juice(rec_trained)

# Check if SMOTE worked
train_prepped %>% count(Outcome)

# Define a no-SMOTE recipe for test data
rec_test <- recipe(Outcome ~ ., data = train_data) %>%
  step_impute_median(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Prep and apply to test set
rec_test_trained <- prep(rec_test)
test_prepped <- bake(rec_test_trained, new_data = test_data)
#naive bayes
nb_spec <- naive_Bayes() %>%
  set_engine("klaR") %>%
  set_mode("classification")

nb_wf <- workflow() %>%
  add_model(nb_spec) %>%
  add_recipe(rec)

nb_fit <- fit(nb_wf, data = train_prepped)

#logistic regression
log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

log_wf <- workflow() %>%
  add_model(log_spec) %>%
  add_recipe(rec)

log_fit <- fit(log_wf, data = train_prepped)

#xgboost
xgb_spec <- boost_tree(
  trees = 200,
  tree_depth = 4,
  learn_rate = 0.05,
  mtry = 3,
  loss_reduction = 0.001,
  sample_size = 0.8
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(rec)

xgb_fit <- fit(xgb_wf, data = train_prepped)

#lightgbm
lgb_spec <- boost_tree(
  trees = 200,
  tree_depth = 4,
  learn_rate = 0.05,
  mtry = 3
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

lgb_wf <- workflow() %>%
  add_model(lgb_spec) %>%
  add_recipe(rec)

lgb_fit <- fit(lgb_wf, data = train_prepped)

library(yardstick)

get_perf <- function(fit, data, true) {
  probs <- predict(fit, data, type = "prob") %>% pull(.pred_Yes)
  preds <- predict(fit, data) %>% pull(.pred_class)
  
  tibble(
    Accuracy = accuracy_vec(true, preds),
    AUC      = roc_auc_vec(true, probs, event_level = "second")
  )
}

#evaluate each model
train_results <- bind_rows(
  get_perf(nb_fit,  train_prepped, train_prepped$Outcome) %>% mutate(Model = "Naive Bayes"),
  get_perf(log_fit, train_prepped, train_prepped$Outcome) %>% mutate(Model = "Logistic Regression"),
  get_perf(xgb_fit, train_prepped, train_prepped$Outcome) %>% mutate(Model = "XGBoost"),
  get_perf(lgb_fit, train_prepped, train_prepped$Outcome) %>% mutate(Model = "LightGBM")
)

test_results <- bind_rows(
  get_perf(nb_fit,  test_prepped, test_prepped$Outcome) %>% mutate(Model = "Naive Bayes"),
  get_perf(log_fit, test_prepped, test_prepped$Outcome) %>% mutate(Model = "Logistic Regression"),
  get_perf(xgb_fit, test_prepped, test_prepped$Outcome) %>% mutate(Model = "XGBoost"),
  get_perf(lgb_fit, test_prepped, test_prepped$Outcome) %>% mutate(Model = "LightGBM")
)

#view results
bind_rows(
  train_results %>% mutate(Set = "Train"),
  test_results %>% mutate(Set = "Test")
) %>%
  dplyr::select(Model, Set, Accuracy, AUC) %>%
  arrange(Model, desc(Set))

library(ggplot2)

# Combine predictions for ROC curves
make_roc_data <- function(fit, data, label) {
  predict(fit, data, type = "prob") %>%
    bind_cols(data) %>%
    roc_curve(truth = Outcome, .pred_Yes, event_level = "second") %>%
    mutate(Model = label)
}

roc_curves <- bind_rows(
  make_roc_data(nb_fit,  test_prepped, "Naive Bayes"),
  make_roc_data(log_fit, test_prepped, "Logistic Regression"),
  make_roc_data(xgb_fit, test_prepped, "XGBoost"),
  make_roc_data(lgb_fit, test_prepped, "LightGBM")
)

# Plot
ggplot(roc_curves, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed") +
  labs(title = "ROC Curves on Test Set", x = "1 - Specificity", y = "Sensitivity") +
  theme_minimal()

library(DALEX)

levels(y_test)
y_test_num <- ifelse(y_test == "Yes", 1, 0)
X_test_df <- as.data.frame(X_test)

y_test <- test_prepped$Outcome

X_test_matrix <- as.matrix(dplyr::select(test_prepped, -Outcome))

# Create explainer with numeric test data and outcome
explainer_xgb <- explain(
  model = pull_workflow_fit(xgb_fit)$fit,
  data = as.matrix(X_test_df),
  y = y_test_num,
  label = "XGBoost"
)

explainer_log <- explain(
  model = pull_workflow_fit(log_fit)$fit,
  data = X_test_df,      # logistic_reg can handle data.frame
  y = y_test_num,
  label = "Logistic Regression"
)


# LightGBM explainer
explainer_lgb <- explain(
  model = pull_workflow_fit(lgb_fit)$fit,
  data = X_test_matrix,
  y = y_test_num,
  label = "LightGBM"
)

# Naive Bayes explainer
explainer_nb <- explain(
  model = pull_workflow_fit(nb_fit)$fit,
  data = X_test_df,
  y = y_test_num,
  label = "Naive Bayes"
)


# Check variable importance plot
vi_xgb <- model_parts(explainer_xgb)
plot(vi_xgb) + ggtitle("XGBoost Variable Importance (DALEX)")

vi_log <- model_parts(explainer_log)
plot(vi_log) + ggtitle("Logistic Regression Variable Importance (DALEX)")

vi_lgb <- model_parts(explainer_lgb)
plot(vi_lgb) + ggtitle("LightGBM Variable Importance (DALEX)")

vi_nb <- model_parts(explainer_nb)
plot(vi_nb) + ggtitle("Naive Bayes Variable Importance (DALEX)")



# Explain prediction for a single patient from test set
new_patient <- X_test_matrix[1, , drop = FALSE]

# For XGBoost
local_explanation <- predict_parts(explainer_xgb, new_observation = new_patient)
plot(local_explanation) + ggtitle("XGBoost Explanation for Patient #1")


