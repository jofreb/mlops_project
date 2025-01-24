import wandb

run = wandb.init()
artifact = run.use_artifact("jofreb-danmarks-tekniske-universitet-dtu/model-registry/NRMS_MLOPS:0", type="model")
artifact_dir = artifact.download("")
model = MyModel()
model.load_state_dict(torch.load("<artifact_dir>/model.ckpt"))
