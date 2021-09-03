# Manual train-test split
scisum_df = pd.read_csv('/content/drive/MyDrive/Datasests/Scisummnet/scisumm.csv')

np.random.seed(10)
msk = np.random.rand(len(scisum_df)) < 0.8
scisum_train_df = scisum_df[msk]
scisum_test_df = scisum_df[~msk]
scisum_train_df.to_csv('/content/drive/MyDrive/Datasests/Scisummnet/scisumm_train.csv')
scisum_test_df.to_csv('/content/drive/MyDrive/Datasests/Scisummnet/scisumm_test.csv')


# hugging face script
!git clone https://github.com/huggingface/transformers.git
os.chdir("/content/transformers")
!pwd
!pip install -e .

!python '/content/drive/MyDrive/Datasests/Scisummnet/run_summarization.py' \
    --model_name_or_path facebook/bart-large-cnn \
    --do_train \
    --do_eval \
    --train_file '/content/drive/MyDrive/Datasests/Scisummnet/scisumm_train.csv' \
    --validation_file '/content/drive/MyDrive/Datasests/Scisummnet/scisumm_test.csv' \
    --text_column text \
    --summary_column summary \
    --source_prefix "summarize: " \
    --output_dir "/content/drive/MyDrive/Datasests/Scisummnet/Summarization Output/" \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --predict_with_generate