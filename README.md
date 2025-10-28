# TRIARD: A ThReshold-tuned strategy combining data Imputation And Rfdc Discovery
Is a research prototype designed to perform **automated data imputation** using **Relaxed Functional Dependencies (RFDcs)**.  
It integrates a **threshold tuning strategy** that iteratively adjusts similarity constraints to improve imputation quality while keeping computational costs manageable.

## ⚙️ Installation

Clone the repository and install the dependencies (Python ≥ 3.10 is recommended):

```bash
git clone https://github.com/DastLab/TRIARD.git
cd TRIARD
pip install -r requirements.txt
```

## Usage
python run.py <dataset_file_path> [options]

### Examples
python run.py datasets --iterations 7 --min_score_to_reach 0.5
python run.py datasets/d01_estonia-passenger-list.csv --iterations 7 --min_score_to_reach 0.5

## Input format

- Dataset: CSV file with ; (semicolon) separator.
- The dataset may include or omit a header row.

## Command-line Arguments

| Argument                                      | Type / Default            | Description                                                                     |
| --------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------- |
| `dataset_file_path`                           | *positional*              | Path to the CSV dataset file or folder.                                         |
| `--remove_log`                                | flag                      | Prevents saving intermediate logs and results.                                  |
| `--dataset_has_not_header`                    | flag                      | Specify if the dataset has **no header row**.                                   |
| `--dataset_null_char`                         | str, default=`?`          | Character used to represent missing values.                                     |
| `--output_folder`                             | str, default=`output`     | Folder where output files and logs will be stored.                              |
| `--increasing_steps`                          | list[int], default=`[]`   | List of threshold increment steps per attribute.                                |
| `--max_similarity_values`                     | list[float], default=`[]` | Maximum similarity values per attribute.                                        |
| `--iterations`                                | int, default=`7`          | Maximum number of tuning iterations per attribute (parameter `μ` in the paper). |
| `--min_score_to_reach`                        | float, default=`0.5`      | Minimum convergence score threshold (parameter `ϛ` in the paper).               |
| `--prevent_use_previous_results_at_each_step` | bool, default=`True`      | Prevents using dependencies from previous iterations.                           |
| `--missing_percentage_generator`              | float, default=`1`        | Percentage of artificially injected missing values for validation.              |
| `--prevent_run_domino`                        | flag                      | Prevents the comparative imputation using the DOMINO algorithm.                 |
| `--time_limit`                                | int, default=`None`       | Time limit for the entire process (in seconds).                                 |

## Output
All generated files and logs are stored in the folder specified by --output_folder.

## Convergence criteria test

python new_imputation_method.py datasets \
  --iterations 1 2 3 4 5 6 7 \
  --min_score_to_reach 0.1 0.2 0.3 0.4 0.5 0.6 

This command executes TRIARD on the all CSV in datasets folder, testing all combinations of:

μ ∈ {1, 2, 3, 4, 5, 6, 7}

ϛ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}


| Argument                                      | Type / Default                                   | Description                                                               |
| --------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| `dataset_file_path`                           | *positional*                                     | Path to a CSV dataset file or a folder containing multiple datasets.      |
| `--remove_log`                                | flag                                             | Disable saving of intermediate logs and results.                          |
| `--dataset_has_not_header`                    | flag                                             | Specify if the dataset has **no header row**.                             |
| `--dataset_null_char`                         | str, default=`?`                                 | Character representing missing values.                                    |
| `--output_folder`                             | str, default=`output`                            | Folder for storing all experiment results and logs.                       |
| `--increasing_steps`                          | list[int], default=`[]`                          | Incremental step size for each attribute’s threshold tuning.              |
| `--max_similarity_values`                     | list[float], default=`[]`                        | Maximum similarity value for each attribute.                              |
| `--iterations`                                | list[int], default=`[1,2,3,4,5,6,7]`             | Range of maximum iteration values (μ) to be tested.                       |
| `--min_score_to_reach`                        | list[float], default=`[0.1,0.2,0.3,0.4,0.5,0.6]` | Range of minimum convergence scores (ϛ) to be tested.                     |
| `--prevent_use_previous_results_at_each_step` | bool, default=`False`                            | If `True`, disables reuse of previously found dependencies during tuning. |
| `--missing_percentage_generator`              | float, default=`1`                               | Percentage of missing values artificially injected for evaluation.        |



