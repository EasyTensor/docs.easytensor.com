# TensorFlow

TensorFlow models can be packaged to a directory which makes it fairly easy to export and run on EasyTensor.

TensorFlow exposes a `save_model` function for all Keras models. Once your model is trained, you can use this function to upload your model to EasyTensor.

```python
import os
import easytensor


export_path = os.path.join(os.getcwd(), "my_model")
print("export_path: {}".format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
model_id, access_token = easytensor.upload_model("My first model", export_path)
```
