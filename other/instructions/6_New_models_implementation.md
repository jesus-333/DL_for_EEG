I want to implement models for my analysis experiment. More precisely I want to implement the following models :
- ShallowFBCSPNet (Schirrmeister 2017)
- Deep4Net (Schirrmeister 2017)
- TIDNet (Kostas 2020).

# Source material

As source material for this task I have provide you the following material :
- `other/material/Schirrmeister_2017.pdf` : Paper that contains the description of ShallowFBCSPNet and Deep4Net. Note that inside the paper they have slightly different names. ShallowFBCSPNet is called Shallow ConvNet and Deep4Net is called Deep ConvNet. 
- `other/material/shallow_FBCSP_net.py`: Braindecode implementation of the ShallowFBCSPNet.
- `other/material/deep4net.py`: Braindecode implementation of Deep4Net.
- `other/material/Kostas_2020.pdf` : Paper with the description of TIDNet.
- `other/material/tidnet_original.py` and `other/material/tidnet_support_original.py`: Original implementation of the TIDNet model.
- `other/material/tidnet_braindecode.py`: Braindecode implementation of the TIDNet model.

You are free to use all the source material as reference and/or search other material online if you need it.

# Implementations details

Each model should be implemented inside `src/jade/model` in a module with their name.
Our implementation should support the custom pooling layer implemented in `src/jade/model/pooling_1d`.

As for other models inside `jade`, also these models should have their config dataclass. Implement the configsinside `src/jade/config/model/`

You could check the implementation of `EEGNet` if you want the example of another model implementation.

If you use the source code that I provided as a starting point do not simply copy and paste that code. 
Our implementation shoulb consistent with the rest of the package, from the internal class structure to the documentation, naming conventions etc.

You do not need to implement any training script. I will do myself in the future.

# Final remarks

The complete description of the codebase are inside the folder `llm_wiki_jade`. You can start from the file `0_wiki_quickstart.md` and then read the part you need for this task.
VERY IMPORTANT. Update all the content inside `llm_wiki_jade` accordingly when you finish this task.
If you need more informations/instructions/clarifications let me know.
