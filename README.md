# k-QSM

Source codes and trained networks described in the paper: Learn Less, Infer More: Learning in the Fourier Domain for Quantitative Susceptibility Mapping.

k-QSM was proposed by Junjie He and Dr. Lihui Wang. 
It reconstructs COSMOS-like QSM and preserves magnetic susceptibility anisotropy in white matter. 

###Environmental Requirements:
Python 3.7

torch 1.8.0


###Files descriptions:  
***k-QSM*** contains the following folders:

**data:** It provides samples of Kirby data.

**src:** It includes source codes.

###Usage

####Install requirements
pip install -r requirements.txt

####Test
You can run infer_script.py directly to reconstruct QSM on the provided data. The results will be saved in the sub-folder of **data**.    

For testing your own data, you can refer to the script and re-write it.

####Train
If you want to train k-QSM by yourself, refer to create_data.py and create your own training data.   
Run train.py to train k-QSM on your dataset.
