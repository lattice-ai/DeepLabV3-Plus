export KAGGLE_USERNAME=$1
export KAGGLE_KEY=$2
kaggle datasets download -d soumikrakshit/human-segmentation
unzip -q human-segmentation
rm human-segmentation.zip