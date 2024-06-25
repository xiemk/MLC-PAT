# Getting the Data

## VOC2007

1. Navigate to the VOC2007 data directory:
```
cd ./voc2007
```
2. Download the data:
```
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar --output voc_raw.tar
```
3. Extract the data:
```
tar -xf voc_raw.tar
```
4. Format the data (If the `formatted_xxx_xxx.npy` files already exist, this step can be skipped.):

```
python format_voc2007.py
```
5. Clean up:
```
rm voc_raw.tar
```


## COCO2014

1. Navigate to the COCO2014 data directory:
```
cd ./coco2014
```
2. Download the data:
```
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
```
3. Extract the data:
```
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
```
4. Format the data (If the `formatted_xxx_xxx.npy` files already exist, this step can be skipped.):
```
python format_coco2014.py
```
5. Clean up:
```
rm coco_annotations.zip
rm coco_train_raw.zip
rm coco_val_raw.zip
```


## VG256

1. Navigate to the VG256 data directory:
```
cd ./vg256
```
2. Download the data:
```
curl https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip --output images.zip
curl https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip --output images2.zip
curl https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1_2.json.zip --output objects.zip
```
3. Extract and merge the data:
```
unzip -q images.zip
unzip -q images2.zip
unzip -q objects.zip
mv ./VG_100K_2/* ./VG_100K/
```
4. Format the data (If the `formatted_xxx_xxx.npy` files already exist, this step can be skipped.):
```
python format_vg256.py
```
5. Clean up:
```
rm images.zip
rm images2.zip
rm objects.zip
rm -rf VG_100K_2
```

# A Uniform Format for Data

The `format_xxx.py` can be used to produce uniformly formatted image lists and labels for the framework.
