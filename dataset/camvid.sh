mkdir /root/.kaggle
touch /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
echo "{\"username\":\"<YOUR-USERNAME>\",\"key\":\"<YOUR-KAGGLE-KEY>\"}" > /root/.kaggle/kaggle.json
kaggle datasets download -d jcoral02/camvid
unzip -q ./camvid.zip
rm ./camvid.zip