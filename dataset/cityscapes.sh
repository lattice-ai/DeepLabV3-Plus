mkdir train_imgs train_labels val_imgs val_labels
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<YOUR_USER_NAME>&password=<YOUR_PASSWORD>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
rm index.html
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<YOUR_USER_NAME>&password=<YOUR_PASSWORD>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
rm index.html
unzip -q gtFine_trainvaltest.zip
rm README
rm license.txt
unzip -q leftImg8bit_trainvaltest.zip
mv ./leftImg8bit/train/**/*leftImg8bit.png ./train_imgs
mv ./leftImg8bit/val/**/*leftImg8bit.png ./val_imgs
mv ./gtFine/train/**/*labelIds.png ./train_labels
mv ./gtFine/val/**/*labelIds.png ./val_labels