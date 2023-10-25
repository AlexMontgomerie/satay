# Create the folder and download YOLOv3 models
mkdir yolov3
cd yolov3
wget https://drive.google.com/uc?id=1XzfKGgpwcQp_ATdt93zx9ZTJXZNW2oib&export=download
wget https://drive.google.com/uc?id=1lC_LOwVz-WlfL5PG6fQVib8_JqTBZmUE&export=download
wget https://drive.google.com/uc?id=14lWTELBWpRZYqlNycMJntITYHmb3un7C&export=download
wget https://drive.google.com/uc?id=1mRkym2dVluYSdV-s_YG4Bw72F6tScZ5I&export=download
wget https://drive.google.com/uc?id=1XYg-THg9t2-Ulo1d2j13Rt6k3KeUMpK8&export=download

sleep 10

mv uc?id=1XzfKGgpwcQp_ATdt93zx9ZTJXZNW2oib yolov3-tiny_imgsz416_fp16.onnx
mv uc?id=1lC_LOwVz-WlfL5PG6fQVib8_JqTBZmUE yolov3-tiny_imgsz320_fp16.onnx
mv uc?id=14lWTELBWpRZYqlNycMJntITYHmb3un7C yolov3-tiny_imgsz640_fp16.onnx
mv uc?id=1mRkym2dVluYSdV-s_YG4Bw72F6tScZ5I yolov3_imgsz320_fp16.onnx
mv uc?id=1XYg-THg9t2-Ulo1d2j13Rt6k3KeUMpK8 yolov3_imgsz320_fp16.onnx

# Create the folder and download YOLOv5 models
cd ..
mkdir yolov5
cd yolov5

wget https://drive.google.com/uc?id=1cANwTxFDhM-pyU6utr-IDhaji-ZDblgm&export=download
wget https://drive.google.com/uc?id=1UGC5ctDWDA3nW6t--N_74SrggG0gYCO0&export=download
wget https://drive.google.com/uc?id=1NU543JeKPXixMI2KZ53SfDQ531CsrDW7&export=download
wget https://drive.google.com/uc?id=1NBAQcs0BWX0Fd6DI1nY7Xy14Ldj15nEa&export=download
wget https://drive.google.com/uc?id=1MiZOTlkB2hvKpT5cLlFlHMMo3hh7sCGs&export=download
wget https://drive.google.com/uc?id=1GtPrI62Z9tpC3GF1RgqpVq1xc-cDuWPI&export=download
wget https://drive.google.com/uc?id=165GuQaihvGwPv8-EaP-zrin-Zn2kH0Yt&export=download
wget https://drive.google.com/uc?id=15F7DLG_pdd8b8zKEgJ75h8bEkaZWr8dB&export=download
wget https://drive.google.com/uc?id=13yTG1Bp8Vk7umxp53UzT9UR3utV0gEmv&export=download

sleep 10

mv uc?id=1cANwTxFDhM-pyU6utr-IDhaji-ZDblgm yolov5s_imgsz640_fp16.onnx
mv uc?id=1UGC5ctDWDA3nW6t--N_74SrggG0gYCO0 yolov5s_imgsz320_fp16.onnx
mv uc?id=1NU543JeKPXixMI2KZ53SfDQ531CsrDW7 yolov5m_imgsz640_fp16.onnx
mv uc?id=1NBAQcs0BWX0Fd6DI1nY7Xy14Ldj15nEa yolov5m6_imgsz1280_fp16.onnx
mv uc?id=1MiZOTlkB2hvKpT5cLlFlHMMo3hh7sCGs yolov5n_imgsz320_fp16.onnx
mv uc?id=1GtPrI62Z9tpC3GF1RgqpVq1xc-cDuWPI yolov5m_imgsz320_fp16.onnx
mv uc?id=165GuQaihvGwPv8-EaP-zrin-Zn2kH0Yt yolov5n_imgsz640_fp16.onnx
mv uc?id=15F7DLG_pdd8b8zKEgJ75h8bEkaZWr8dB yolov5n6_imgsz1280_fp16.onnx
mv uc?id=13yTG1Bp8Vk7umxp53UzT9UR3utV0gEmv yolov5s6_imgsz1280_fp16.onnx

# Create the folder and download YOLOv8 models
cd ..
mkdir yolov8
cd yolov8
wget https://drive.google.com/uc?id=1vSrSJfH8dgzY5LONas4zcOsRO1Oljfko&export=download
wget https://drive.google.com/uc?id=1rHQG_s5p7-389HkW3nwzmEFlI-dwdOcM&export=download
wget https://drive.google.com/uc?id=1pf2SEn1zC7K7TgRosG8mttC0E9S6onWa&export=download
wget https://drive.google.com/uc?id=1gCaYiQvn44wGXfZactSXYNLhOi4i2oL8&export=download
wget https://drive.google.com/uc?id=1CAg3yARf-B1egI6e8FQOpyPwO8YoCzma&export=download
wget https://drive.google.com/uc?id=17L7D19kdEut3KVwp-MkgCa0Rmtf1bDXy&export=download

sleep 10

mv uc?id=1vSrSJfH8dgzY5LONas4zcOsRO1Oljfko yolov8n_imgsz320_fp16.onnx
mv uc?id=1rHQG_s5p7-389HkW3nwzmEFlI-dwdOcM yolov8m_imgsz640_fp16.onnx
mv uc?id=1pf2SEn1zC7K7TgRosG8mttC0E9S6onWa yolov8m_imgsz320_fp16.onnx
mv uc?id=1gCaYiQvn44wGXfZactSXYNLhOi4i2oL8 yolov8n_imgsz640_fp16.onnx
mv uc?id=1CAg3yARf-B1egI6e8FQOpyPwO8YoCzma yolov8s_imgsz640_fp16.onnx
mv uc?id=17L7D19kdEut3KVwp-MkgCa0Rmtf1bDXy yolov8s_imgsz320_fp16.onnx

# Go back to the main folder
cd ../..
