import bentoml

# Save the model from file path 
bentoml.pytorch.save_model(
    "food_segmentation_model",
    model_file="",
    signatures={"__call__": {"batchable": False}},
    )
