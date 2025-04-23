

NOTE: 
- we never ended up using: TripletDataset or DualEncoderDataset, dig into why. 



TO DO's TOMORROW: 
0) Go through v04 in detail -> note that we havent reviewed it. Review every line. (DONE)
1) An inference script (05_eval_dualen.py) to score or rank docs (DONE)
2) Dont forget to reimplement huggingface and wandb logging everywhere (DONE)
3) Set up v04 so that we can run both the simple model architecture as well as the RNN version tomorrow (DONE)

4) Maybe build a script that can manage v0-v05 in one loop very nicely, ensuring that our configs and parameters across the scripts are consistent. Also ensure that all the parameters are indeed at the top of each script (we define some in the middle sometimes)

5) Run with the first data-set a small subset. 
6) Test loading onto ChromaDB. 
7) Train on the full dataset. 