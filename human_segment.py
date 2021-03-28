import os 
import glob 
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

data_dir = './dataset/'

dst_dir = 'segmented_data' + data_dir[data_dir.rfind('/'):]
os.makedirs(dst_dir)

# Read DetectoRS config file
config = 'mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py'

# DetectoRS pretrained on COCO dataset
checkpoint = 'checkpoints/detectors_htc_r50_1x_coco-329b1453.pth'

# initialize the detector
model = init_detector(config, checkpoint, device='cuda')

person_idx = model.CLASSES.index('person')

for img_path in glob.glob(f'{data_dir}/*/*'):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    result = inference_detector(detectorRSModel, img_path)
    mask = result[1][person_idx][0]
    
    cropped_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    cv2.imwrite(f'{dst_dir}/{img_path[img_path.rfind('/')+1:]}', cropped_img)
    