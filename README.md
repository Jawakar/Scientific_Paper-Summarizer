# Scientific_Paper-Summarizer
I achieved a higher Rouge-2 score by fine-tuning the BART-large-cnn model that is higher than scores given in the [ScisummNet paper](https://arxiv.org/pdf/1909.01716.pdf)
</br>
| Summarizer    | 2-R          |
| ------------- |:-------------:| 
| GCN Hybrid 2 w/ auth model from ScisummNet paper | 33.88 |
| Fine tuned BART-large-cnn    | 53.15  | 

# Dataset
 The CL-Scisumm project developed the first large-scale, human-annotated Scisumm dataset, ScisummNet. It provides over 1,000 papers in the ACL anthology network with their citation networks (e.g. citation sentences, citation counts) and their comprehensive, manual summaries. 
 </br></br>
 To know more about the dataset, [click here](https://www.kaggle.com/jawakar/scisummnet-corpus), this will take you to my Kaggle dataset page, where I have parsed and uploaded the XML format to CSV format.
# Tech Stack 
1. PyTorch
2. Transformers
3. BART-large-cnn(A Facebook AI's model)
4. Hugging Face
5. lxml
6. PyPDF2
# How to get this model
Use this [python script](https://github.com/Jawakar/Scientific_Paper-Summarizer/blob/main/Finetune%20BART.py) `Finetune BART.py` and train and validate with this [CSV file](https://github.com/Jawakar/Scientific_Paper-Summarizer/blob/main/scisumm.csv) `scisumm.csv`. The final model will be a PyTorch model, you can use this for scientific research-oriented summarization tasks. You can check this [notebook](https://github.com/Jawakar/Scientific_Paper-Summarizer/blob/main/Summarizer.ipynb) `Summarizer.ipynb` as a referral to summarize the research paper.
# Next Step
The attention module of the Transformer can only deal with fixed-length strings. As BART tokenizer from hugging face library uses a standard Transformer-based neural machine translation architecture, which provides support to 1024 tokens only. So creating a summary for a text of 30k-50k words like research papers increases the time complexity abruptly. 
</br></br>
I'm willing to generate summaries quickly and efficiently in future by utilizing [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451). The Reformer performs on par with Transformer models while being much more memory-efficient and much faster on longer sequences.
# Acknowledgement
I thank Hugging Face's [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) model, which is the current state of the art model when it comes to summarization tasks in NLP and [CL-ScisummNet](https://cs.stanford.edu/~myasu/projects/scisumm_net/), for creating a dataset combining 1000 scientific research papers.
