## jobs folder
* Most of the functionalities will be composed from just one job. And therefore the only think that this folder will usually have will be a name of this one and only job.

* But in case of a rare scenario it is allowed to be more than one job here.

### How to deploy a job
python scripts/prepare_job_json.py example_image_job $ENV --token $TOKEN
databricks jobs create --json-file job.json > job_created.json