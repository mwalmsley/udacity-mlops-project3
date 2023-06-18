import requests
import os

# https://feedback.render.com/features/p/detect-githubgitlab-ci-test-status-to-trigger-auto-deploys
# https://render.com/docs/deploy-hooks

# added to environment on github
requests.get(os.environ['DEPLOY_HOOK_URL'])