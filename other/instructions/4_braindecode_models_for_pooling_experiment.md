I want to add the possibilities to use the model already implemented in braindecode in my pooling experiments. 

# Basic idea
I want you to add new module inside `src/jade/model`. The module should have name `braindecode_pooling_extension` it must explicitly focus on using models already implemented with Braindecode, but with different types of pooling layers.
E.g. if a model in braindecode use `AvgPool2d` this module should get the model from braindecode and substitute all the pooling layer with the layers I indicated.
At the moment the supported layer should be the one inside `src/jade/model/pooling_1d/` (extended for the 2d case when needed).

`braindecode_pooling_extension` should be a wrapper around the braindecode implementations. Just to give you and idea 

```python

class braindecode_wrapper(...) :

    def __init__(self, model_name : str, pooling_config, braindecode_config_dict) :
        ...

        model = (...)

        for layer in model :
            if layer is pooling : 
                layer = get_pooling_layer(pooling_config)

```

The code is not syntactically correct, but it serves only to give you a general idea. You can change the class name if you find a more appropriate one.

# Important Details
Some important notes on how the class should work:
- The pooling factor will remain the same. If a layer has a pooling factor of 2 also the new one will have a pooling factor of 2.  What change is the type of pooling, e.g. from AvgPool to sinc_pool.
- The class should check the expected input data shape. E.g. EEGNet expect input data in the shape `B x 1 x C x T` while ShallowFBCSPNet expect data in the shape `B x 1 x T x C` (with B batch size, C channels, T time samples). I do not know if other models expect different input shapes, but the class should be able to handle this. The class should also check if the pooling layer is 1D or 2D and apply the correct pooling layer accordingly.
- Regarding the pooling parameters. Since this is the main point of the module, you must use our pooling config.
- Regarding the model parameters. Since there are plenty of models in braindecode, it is not feasible to implement their config inside the config module. In this case, the class should accept a dictionary with the model configuration. The parameters check will be done by the braindecode model itself.
- I see that at least `ShallowFBCSPNet` has a layer called `pool_nonlin_exp`. I do not know if also other model have it. Check how it works and give me a list of possibilities on how to handle it. E.g. if we substitute the pooling layer with a different one, should we keep the `pool_nonlin_exp` or remove it? 

# Final remarks

The complete description of the codebase are inside the folder `llm_wiki_jade`. You can start from the file `0_wiki_quickstart.md`
VERY IMPORTANT. Update all the content inside `llm_wiki_jade` accordingly.
If you need more informations/instructions/clarifications let me know.
