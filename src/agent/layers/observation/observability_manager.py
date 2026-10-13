# ## Description
# #   This class facilitates the agent observability. Using the LGTM stack, 
# #   the activity of each layer of the harness can be monitored and tracked.
# #   As a result, it will be easier to debug, optimize and understand agent
# #   activity.
# #   S.F.B
# ##

# # Imports
import os, sys
import pdb
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# from guardrails.policy_enforcement_manager import PolicyEnforcement
import asyncio
# import argparse

# import asyncio

class ObservabilityManager:
    def __init__(self, *, evals, guardrails, obs_config):
        self.evals = evals
        self.guardrails = guardrails
        self.config = obs_config
        
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
async def main(
):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    