import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")

    parser.add_argument("--test_end", type=str, required=False, default='2023-01-01', help="date to split train/test data")
    parser.add_argument("--epochs", required=False, default=1000, type=int)
    parser.add_argument("--n_samples", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    weekly_data = pd.read_json(args.data, orient="records")

    mlflow.log_metric("num_weeks", weekly_data.shape[0])

    import tempfile
    # Log first few rows as text artifact
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmpfile:
        tmpfile.write(weekly_data.head(5).to_string(index=False))
        tmpfile_path = tmpfile.name

    mlflow.log_artifact(tmpfile_path, artifact_path="data_preview")

    # Clean up temporary file
    os.remove(tmpfile_path)

    ####################
    #</prepare the data>
    ####################

    import logging
    import sys


    # # Configure logging to write to stdout (this ends up in std_log.txt in Azure ML)
    # logging.basicConfig(
    #     stream=sys.stdout,
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )
    # logger = logging.getLogger(__name__)

    weekly_data = weekly_data.set_index(pd.to_datetime(weekly_data['date']))
    weekly_data_train = weekly_data[weekly_data.index < pd.to_datetime(args.test_end)]
    weekly_data_test = weekly_data[weekly_data.index >= pd.to_datetime(args.test_end)]

    mlflow.log_metric("num_weeks_train", weekly_data_train.shape[0])
    mlflow.log_metric("num_weeks_test", weekly_data_test.shape[0])

    # logger.info(f"Columns in the dataset: {weekly_data.columns}")

    cols_fut_true = ['Butter_EEX_4_weeks_ahead', 'Butter_EEX_8_weeks_ahead', 'Butter_EEX_12_weeks_ahead', 'SMP_food_EEX_4_weeks_ahead', 'SMP_food_EEX_8_weeks_ahead', 'SMP_food_EEX_12_weeks_ahead']
    cols_fut_estimate = ['Fut_butter_4', 'Fut_butter_8', 'Fut_butter_12', 'Fut_smp_4', 'Fut_smp_8', 'Fut_smp_12']
    cols_x = ['Butter_EEX', 'SMP_food_EEX']

    cols_y = ['Gouda_EEX']
    cols_target = ['Gouda_EEX_4_weeks_ahead', 'Gouda_EEX_8_weeks_ahead', 'Gouda_EEX_12_weeks_ahead']

    fut_true_train = torch.tensor(weekly_data_train[cols_fut_true].values, dtype=torch.float32)
    fut_true_test = torch.tensor(weekly_data_test[cols_fut_true].values, dtype=torch.float32)

    fut_estimate_train = torch.tensor(weekly_data_train[cols_fut_estimate].values, dtype=torch.float32)
    fut_estimate_test = torch.tensor(weekly_data_test[cols_fut_estimate].values, dtype=torch.float32)

    x_train = torch.tensor(weekly_data_train[cols_x].values, dtype=torch.float32)
    x_test = torch.tensor(weekly_data_test[cols_x].values, dtype=torch.float32)
    y_train = torch.tensor(weekly_data_train[cols_y].values, dtype=torch.float32)
    y_test = torch.tensor(weekly_data_test[cols_y].values, dtype=torch.float32)

    fut_target_train = torch.tensor(weekly_data_train[cols_target].values, dtype=torch.float32)
    fut_target_test = torch.tensor(weekly_data_test[cols_target].values, dtype=torch.float32)

    dates_train  = weekly_data_train.index.values
    dates_test  = weekly_data_test.index.values

    ###################
    #</train the model>
    ###################
    class LearnableCovarianceFreeForm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            # Parameterize the Cholesky decomposition of the covariance matrix
            self.lower_triangular = nn.Parameter(torch.eye(dim))

        def forward(self):
            lower = torch.tril(self.lower_triangular)
            diag = torch.diag(lower)
            lower = lower + torch.diag(torch.abs(diag) + 1e-6)
            covariance_matrix = torch.matmul(lower, lower.T)
            return covariance_matrix

        def log_prob(self, mean, actual_values):
            covariance = self.forward()
            dist = MultivariateNormal(mean, covariance)
            return dist.log_prob(actual_values)

        def sample(self, num_samples, mean):
            covariance = self.forward()
            dist = MultivariateNormal(mean, covariance)
            return dist.sample((num_samples,))

        def get_learned_covariance(self):
            return self.forward().detach()


    class SimpleLRModel(nn.Module):
        def __init__(self, input_dim, future_dim):
            super(SimpleLRModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1, bias=False)
            self.covariance_learner = LearnableCovarianceFreeForm(future_dim)

        def forward(self, x):
            return self.linear(x)

        def combined_log_prob(self, x_future, x_future_actual, y_pred, y_true):
            # Calculate the log probability of the future estimates given the actual values
            log_prob_future = self.covariance_learner.log_prob(x_future, x_future_actual)
            # Calculate the log probability of the target variable given the future estimates
            log_prob_target = self._likelihood(y_pred, y_true)
            return log_prob_future + log_prob_target

        def _likelihood(self, y_pred, y_true):
            # Assuming a Gaussian likelihood for the target variable
            mse = torch.mean((y_pred - y_true) ** 2, dim=1)
            return - mse

        def get_learned_future_covariance(self):
            return self.covariance_learner.get_learned_covariance()

        def sample(self, num_samples, butter_fut, smp_fut, time_steps):
            pred_samples = torch.zeros(num_samples, time_steps)
            butter_samples = torch.zeros(num_samples, len(butter_fut))
            smp_samples = torch.zeros(num_samples, len(smp_fut))

            futs_mean = torch.cat((butter_fut, smp_fut), dim=0)

            for i in range(num_samples):
                fut_values = self.covariance_learner.sample(1, futs_mean).squeeze(0)

                butter_fut_sampled = fut_values[:butter_fut.shape[0]]  # Use input shape
                smp_fut_sampled = fut_values[butter_fut.shape[0]:]      # Use input shape

                butter_samples[i] = butter_fut_sampled
                smp_samples[i] = smp_fut_sampled

                single_pred = torch.zeros(time_steps)
                for t in range(time_steps):
                    combined_features = torch.cat((butter_fut_sampled[t].unsqueeze(0), smp_fut_sampled[t].unsqueeze(0)), dim=0)

                    single_pred[t] = self.linear(combined_features).squeeze(0) # Squeeze the output

                pred_samples[i] = single_pred

            return pred_samples, butter_samples, smp_samples

    num_fut_values = len(cols_fut_true)

    # Instantiate the model and optimizer
    model = SimpleLRModel(input_dim=len(cols_x), future_dim=num_fut_values)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x_train)

        loss = -torch.sum(model.combined_log_prob(fut_estimate_train, fut_true_train, pred, fut_target_train))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("MAE of index transformation: ", np.mean(np.abs(pred[:, 0].detach().numpy() - fut_target_train[:, 0].numpy())))

    print("Testting on test set")
    with torch.no_grad():
        model.eval()

        pred_test = model(x_test)
        test_loss = -torch.sum(model.combined_log_prob(fut_estimate_test, fut_true_test, pred_test, fut_target_test))
        print(f'Test Loss: {test_loss.item():.4f}')
        print("MAE of index transformation: ", np.mean(np.abs(pred_test[:, 0].detach().numpy() - fut_target_test[:, 0].numpy())))
        print("MSE of index transformation: ", np.mean((pred_test[:, 0].detach().numpy() - fut_target_test[:, 0].numpy())**2))
        print("RMSE of index transformation: ", np.sqrt(np.mean((pred_test[:, 0].detach().numpy() - fut_target_test[:, 0].numpy())**2)))


    mlflow.log_metric("MAE train", np.mean(np.abs(pred[:, 0].detach().numpy() - fut_target_train[:, 0].numpy())))
    mlflow.log_metric("MAE test", np.mean(np.abs(pred_test[:, 0].detach().numpy() - fut_target_test[:, 0].numpy())))
    mlflow.log_metric("RMSE test", np.sqrt(np.mean((pred_test[:, 0].detach().numpy() - fut_target_test[:, 0].numpy())**2)))
    mlflow.log_metric("MSE test", np.mean((pred_test[:, 0].detach().numpy() - fut_target_test[:, 0].numpy())**2))

    ##########################
    #<save and register model>
    ##########################
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.pytorch.log_model(
        model, 
        registered_model_name=args.registered_model_name,
        artifact_path="model_lr_generative",
    )


    ###########################
    #</save and register model>
    ###########################

    # Stop Logging
    mlflow.end_run()




if __name__ == "__main__":
    main()
