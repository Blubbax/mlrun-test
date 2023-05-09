import mlrun
mlrun.set_environment("http://localhost:8080", artifact_path="./")

training_job = mlrun.code_to_function(
    name="basic-training",
    filename="train.py",
    kind="job",
    image="mlrun/mlrun",
    handler="train"
)

run = training_job.run(
    inputs={}, 
    params = {"n_estimators": 100, "learning_rate": 1e-1, "max_depth": 3}
)