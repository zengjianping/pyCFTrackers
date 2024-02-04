import glob, os, sys, cv2
import pandas as pd
import argparse, yaml
import numpy as np
from PIL import Image
import argparse, time

from lib.utils import get_img_list,get_ground_truthes,APCE,PSR
from cftracker.mosse import MOSSE
from cftracker.csk import CSK
from cftracker.kcf import KCF
from cftracker.cn import CN
from cftracker.dsst import DSST
from cftracker.staple import Staple
from cftracker.dat import DAT
from cftracker.eco import ECO
from cftracker.bacf import BACF
from cftracker.csrdcf import CSRDCF
from cftracker.samf import SAMF
from cftracker.ldes import LDES
from cftracker.mkcfup import MKCFup
from cftracker.strcf import STRCF
from cftracker.mccth_staple import MCCTHStaple
from lib.eco.config import otb_deep_config,otb_hc_config
from cftracker.config import staple_config,ldes_config,dsst_config,csrdcf_config,mkcf_up_config,mccth_staple_config


def get_frames(video_name, time=None, size=None):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            fts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            ret, frame = cap.read()
            if ret:
                if isinstance(time, list):
                    if time[1] > 0 and fts > time[1]:
                        break
                    if time[0] > 0 and fts < time[0]:
                        continue
                if isinstance(size, list) and all(size):
                    frame = cv2.resize(frame, size)
                yield frame
            else:
                break
    else:
        images = glob.glob(os.path.join(video_name, 'img/*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

class PyTracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        self.tracker = None
    
    def _create_tracker(self, is_color=True):
        if self.tracker_type == 'MOSSE':
            self.tracker=MOSSE()
        elif self.tracker_type=='CSK':
            self.tracker=CSK()
        elif self.tracker_type=='CN':
            self.tracker=CN()
        elif self.tracker_type=='DSST':
            self.tracker=DSST(dsst_config.DSSTConfig())
        elif self.tracker_type=='Staple':
            self.tracker=Staple(config=staple_config.StapleConfig())
        elif self.tracker_type=='Staple-CA':
            self.tracker=Staple(config=staple_config.StapleCAConfig())
        elif self.tracker_type=='KCF_CN':
            self.tracker=KCF(features='cn',kernel='gaussian')
        elif self.tracker_type=='KCF_GRAY':
            self.tracker=KCF(features='gray',kernel='gaussian')
        elif self.tracker_type=='KCF_HOG':
            self.tracker=KCF(features='hog',kernel='gaussian')
        elif self.tracker_type=='DCF_GRAY':
            self.tracker=KCF(features='gray',kernel='linear')
        elif self.tracker_type=='DCF_HOG':
            self.tracker=KCF(features='hog',kernel='linear')
        elif self.tracker_type=='DAT':
            self.tracker=DAT()
        elif self.tracker_type=='ECO-HC':
            self.tracker=ECO(config=otb_hc_config.OTBHCConfig())
        elif self.tracker_type=='ECO':
            self.tracker=ECO(config=otb_deep_config.OTBDeepConfig())
        elif self.tracker_type=='BACF':
            self.tracker=BACF()
        elif self.tracker_type=='CSRDCF':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFConfig())
        elif self.tracker_type=='CSRDCF-LP':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFLPConfig())
        elif self.tracker_type=='SAMF':
            self.tracker=SAMF()
        elif self.tracker_type=='LDES':
            self.tracker=LDES(ldes_config.LDESDemoLinearConfig())
        elif self.tracker_type=='DSST-LP':
            self.tracker=DSST(dsst_config.DSSTLPConfig())
        elif self.tracker_type=='MKCFup':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupConfig())
        elif self.tracker_type=='MKCFup-LP':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupLPConfig())
        elif self.tracker_type=='STRCF':
            self.tracker=STRCF()
        elif self.tracker_type=='MCCTH-Staple':
            self.tracker=MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())
        elif self.tracker_type=='MCCTH':
            self.tracker=MCCTH(config=mccth_config.MCCTHConfig())
        else:
            raise NotImplementedError

    def process(self, video_name, save_result=False, select_roi=True, expand_roi=False, verbose=True):
        gt_bboxes = None
        height, width = 0, 0
        is_color = True
        process_option = {
            'time_range': [0,0],
            'zoom_size': [0,0]
        }
        video_writer = None

        if os.path.isdir(video_name):
            filenames = sorted(glob.glob(os.path.join(video_name, "img/*.jpg")),
                key=lambda x: int(os.path.basename(x).split('.')[0]))
            num_frames = len(filenames)
            gt_bboxes = pd.read_csv(os.path.join(video_name, "groundtruth_rect.txt"), sep='\t|,| ',
                    header=None, names=['xmin', 'ymin', 'width', 'height'],
                    engine='python')
        
        elif os.path.isfile(video_name):
            option_file = os.path.splitext(video_name)[0] + '.yaml'
            if os.path.isfile(option_file):
                data = open(option_file, 'r', encoding='utf-8').read()
                process_option.update(yaml.safe_load(data))
        
        time_range = process_option['time_range']
        zoom_size = process_option['zoom_size']
        init_bbox = process_option.get('init_bbox', None)
        init_time = 0.0
        update_time = 0.0
        update_frames = 0

        idx = -1
        for idx, frame in enumerate(get_frames(video_name, time_range, zoom_size)):
            if self.tracker is None:
                height, width = frame.shape[:2]
                if len(frame.shape) == 3:
                    is_color = True
                else:
                    is_color = False
                    frame = frame[:, :, np.newaxis]
            
                if gt_bboxes is not None:
                    bbox = gt_bboxes.iloc[0].values
                elif init_bbox is not None and not select_roi:
                    x, y, w, h = init_bbox
                    if expand_roi:
                        bbox = [int(x-w/4), int(y-h/4), int(w*3/2), int(h*3/2)]
                    else:
                        bbox = [x, y, w, h]
                else:
                    try:
                        bbox = cv2.selectROI('init', frame, False, False)
                        process_option['init_bbox'] = bbox
                        if os.path.isfile(video_name):
                            option_file = os.path.splitext(video_name)[0] + '.yaml'
                            fil = open(option_file, 'w', encoding='utf-8')
                            yaml.safe_dump(process_option, fil)
                            fil.close()
                    except:
                        cv2.waitKey(0)
                        exit()

                # starting tracking
                self._create_tracker(is_color)
                time_s = time.time()
                self.tracker.init(frame, bbox)
                init_time = time.time() - time_s
                bbox = (bbox[0]-1, bbox[1]-1,
                        bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
                rc_color = (255,0,0)

                if save_result and video_writer is None :
                    bbox_name = 'bbox_e' if expand_roi else 'bbox_n'
                    result_dir = os.path.join(os.path.dirname(video_name), 'outputs', bbox_name, self.tracker_type)
                    result_file = os.path.join(result_dir, os.path.basename(video_name))
                    os.makedirs(result_dir, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(result_file, fourcc, 25, (width,height), True)

            else:
                time_s = time.time()
                bbox = self.tracker.update(frame, vis=verbose)
                update_time += time.time() - time_s
                update_frames += 1
                rc_color = (0,255,0)

            if verbose:
                x1,y1,w,h = bbox
                if len(frame.shape)==2:
                    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                if hasattr(self.tracker, 'score'):
                    score = self.tracker.score
                    #apce = APCE(score)
                    #psr = PSR(score)
                    #F_max = np.max(score)
                    size=self.tracker.crop_size
                    score = cv2.resize(score, size)
                    score -= score.min()
                    score =score/ score.max()
                    score = (score * 255).astype(np.uint8)
                    # score = 255 - score
                    score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
                    center = (int(x1+w/2),int(y1+h/2))
                    x0,y0=center
                    x0=np.clip(x0,0,width-1)
                    y0=np.clip(y0,0,height-1)
                    center=(x0,y0)
                    xmin = int(center[0]) - size[0] // 2
                    xmax = int(center[0]) + size[0] // 2 + size[0] % 2
                    ymin = int(center[1]) - size[1] // 2
                    ymax = int(center[1]) + size[1] // 2 + size[1] % 2
                    left = abs(xmin) if xmin < 0 else 0
                    xmin = 0 if xmin < 0 else xmin
                    right = width - xmax
                    xmax = width if right < 0 else xmax
                    right = size[0] + right if right < 0 else size[0]
                    top = abs(ymin) if ymin < 0 else 0
                    ymin = 0 if ymin < 0 else ymin
                    down = height - ymax
                    ymax = height if down < 0 else ymax
                    down = size[1] + down if down < 0 else size[1]
                    score = score[top:down, left:right]
                    crop_img = frame[ymin:ymax, xmin:xmax]
                    score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
                    #frame[ymin:ymax, xmin:xmax] = score_map

                show_frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), rc_color, 1)
                cv2.putText(show_frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                #cv2.putText(show_frame, 'APCE:' + str(apce)[:5], (5, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                #cv2.putText(show_frame, 'PSR:' + str(psr)[:5], (5, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
                #cv2.putText(show_frame, 'Fmax:' + str(F_max)[:5], (5, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

                if video_writer is not None:
                    video_writer.write(show_frame)
                cv2.imshow('demo', show_frame)

                key = cv2.waitKey(5)
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self.tracker = None
            
        if video_writer is not None:
            video_writer.release()
        
        update_time /= update_frames
        print('init time: {}, update time: {}'.format(init_time, update_time))

        return True


def main(args):
    tracker = PyTracker(args.tracker_type)
    tracker.process(args.video_name, args.save_result, args.select_roi, args.expand_roi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_type', type=str, default='ECO-HC')
    parser.add_argument('--video_name', type=str, default='datas/sequences/Crossing/')
    parser.add_argument('--save_result', action='store_true', default=False)
    parser.add_argument('--select_roi', action='store_true', default=False)
    parser.add_argument('--expand_roi', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

