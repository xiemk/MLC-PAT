import json
import os
import numpy as np
import argparse

pp = argparse.ArgumentParser(description='Format VG256 metadata.')
pp.add_argument('--load-path', type=str, default='.', help='Path to a directory containing a copy of the VG256 dataset.')
pp.add_argument('--save-path', type=str, default='.', help='Path to output directory.')
args = pp.parse_args()

common_attributes = set(['white','black','blue','green','red','brown','yellow',
'small','large','silver','wooden','orange','gray','grey','metal','pink','tall',
'long','dark'])

def clean_string(string):
    string = string.lower().strip()
    if len(string) >= 1 and string[-1] == '.':
        return string[:-1].strip()
    return string

def clean_objects(string, common_attributes):
    # Return object and attribute lists
    string = clean_string(string)
    words = string.split()
    if len(words) > 1:
        prefix_words_are_adj = True
        for att in words[:-1]:
            if not att in common_attributes:
                prefix_words_are_adj = False
        if prefix_words_are_adj:
            return words[-1:],words[:-1]
        else:
            return [string],[]
    else:
        return [string],[]
    
def generate_split(num_ex, frac, rng):
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

def sort_by_image_id(a, b):
    # sort a and b together by the image id of a
    zipped = zip(a, b)
    sort_zipped = sorted(zipped, key=lambda x:(int(x[0].split('.')[0])))
    result = zip(*sort_zipped)
    a_sorted, b_sorted = [list(x) for x in result]
    return a_sorted, b_sorted

category_to_index = {}
objects_to_index = {}
id2obj={}

# encode object to idx
category_to_objects = json.load(open(os.path.join(args.load_path, 'vg256.json')))
for idx, (category, objects) in enumerate(category_to_objects.items()):
    id2obj[idx]=category
    category_to_index[category] = idx
    for o in objects:
        objects_to_index[o] = idx
    

print('Wait a minute...')
n_classes = len(category_to_index)

images_list = []
labels_list = []

objs_data = json.load(open(os.path.join(args.load_path, 'objects.json')))

for img_attrs in objs_data:
    image = str(img_attrs['image_id']) + '.jpg'
    label = np.zeros(n_classes)
    for obj in img_attrs['objects']:
        clean_obj, _ = clean_objects(obj['names'][0], common_attributes)
        try:
            label[objects_to_index[clean_obj[0]]] = 1
        except:
            pass
    if label.sum() != 0:
        images_list.append(image)
        labels_list.append(label)

# Sort
images_list, labels_list = sort_by_image_id(images_list, labels_list)

# Split
ss_rng = np.random.RandomState(1)
split_idx = {}
(split_idx['train'], split_idx['val']) = generate_split( len(images_list), 0.3, ss_rng)

# Add the complete path to images
images_list = ['VG_100K/' + s for s in images_list]

# Save
images_arr = np.array(images_list)
labels_arr = np.array(labels_list)


obj_count = {}
for idx in range(labels_arr.shape[1]):
    obj_count[id2obj[idx]]=labels_arr[:,idx].sum()

print(sorted(obj_count.items(), key = lambda kv:(kv[1], kv[0])))


for phase in ['train', 'val']:
    np.save(os.path.join(args.save_path, f'formatted_{phase}_images.npy'), images_arr[split_idx[phase]])
    np.save(os.path.join(args.save_path, f'formatted_{phase}_labels.npy'), labels_arr[split_idx[phase]])
