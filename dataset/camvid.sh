echo "{\"username\":\"<YOUR-USERNAME>\",\"key\":\"<YOUR-KAGGLE-KEY>\"}" > /root/.kaggle/kaggle.json
kaggle datasets download -d jcoral02/camvid
unzip -q ./camvid.zip
rm ./camvid.zip