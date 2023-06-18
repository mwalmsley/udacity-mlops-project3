# udacity-mlops-project3

Train census model locally. Deploy API via Render.

## Notes

Set wandb API key if not done already

    export WANDB_API_KEY=...

Clean data with data_cleaning.ipynb. 

    data/census.csv -> data//cleaned_data.csv

Train model, upload model/encoder/binarizer to wandb

    python train_model.py

Test locally (web app not required to be run separately). Artifacts pulled from wandb.

    pytest test_main.py

Push to github when happy. Action will lint and run pytest. If pytest passes, will send GET to `render` deploy hook, triggering app re-deploy. Or can deploy manually via `render` UI or with 

    python render_deploy_hook.py

Test deployed version

    pytest test_main_live.py


## Render details

Build command

    pip install -r requirements

Start command

    uvicorn main:app --host 0.0.0.0 --port 10000
