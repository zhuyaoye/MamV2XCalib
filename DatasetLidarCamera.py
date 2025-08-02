import csv
import os
from math import radians
import cv2

import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from torch.utils.data import Dataset
from torchvision import transforms
from utils import invert_pose, rotate_forward, quaternion_from_matrix,point_in_box
import os.path as osp
import json
from pypcd import pypcd
from abc import ABC, abstractmethod
import mmcv
from scipy.spatial.transform import Rotation 
import glob
import sys


def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    time = None
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"]) / 256.0
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points, time

def read_jpg(jpg_path):
    image = mmcv.imread(jpg_path)
    return image



def get_annos(path, prefix, single_frame, sensortype="camera"):
    img_path = path + prefix + single_frame["image_path"]
    trans0_path = ""
    if "calib_lidar_to_camera_path" in single_frame.keys():
        trans0_path = single_frame["calib_lidar_to_camera_path"]
    else:
        trans0_path = single_frame["calib_virtuallidar_to_camera_path"]
    trans1_path = single_frame["calib_camera_intrinsic_path"]
    trans0, rot0 = get_trans(load_json(osp.join(path, prefix, trans0_path)))
    lidar2camera = {}
    lidar2camera.update(
        {
            "translation": trans0,
            "rotation": rot0,
        }
    )
    # trans0, rot0 = lidar2camera["translation"], lidar2camera["rotation"]
    camera2image = load_json(osp.join(path, prefix, trans1_path))["cam_K"]

    annFile = {}
    img_ann = {}
    calib = {}
    calib.update(
        {
            "cam_intrinsic": camera2image,
            "Tr_velo_to_cam": lidar2camera,
        }
    )

    img_ann.update({"file_name": img_path, "calib": calib})
    imglist = []
    imglist.append(img_ann)
    annFile.update({"images": imglist})
    if not osp.exists(osp.join(path, prefix, "annos")):
        os.mkdir(osp.join(path, prefix, "annos"))
    ann_path_o = osp.join(path, prefix, "annos", single_frame["image_path"].split("/")[-1].split(".")[0] + ".json")
    with open(ann_path_o, "w") as f:
        json.dump(annFile, f)


def build_path_to_info(prefix, data, sensortype="lidar"):
    path2info = {}
    if sensortype == "lidar":
        for elem in data:
            if elem["pointcloud_path"] == "":
                continue
            path = osp.join(prefix, elem["pointcloud_path"])
            path2info[path] = elem
    elif sensortype == "camera":
        for elem in data:
            if elem["image_path"] == "":
                continue
            path = osp.join(prefix, elem["image_path"])
            path2info[path] = elem
    return path2info

def build_frame_to_info(data):
    frame2info = {}
    for elem in data:
        if elem["frame_id"] == "":
            continue
        frame2info[elem["frame_id"]] = elem
    return frame2info


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def get_trans(info):
    return info["translation"], info["rotation"]




class Coord_transformation(object):
    """
    coord_list=['Infrastructure_image','Infrastructure_camera','Infrastructure_lidar',
                        'world', 'Vehicle_image','Vehicle_camera','Vehicle_lidar',
                        'Vehicle_novatel']

    'Infrastructure_image' ->'Infrastructure_camera'->'Infrastructure_lidar'->'world'
                                                                                   ^
                                                                                   |
                          Vehicle_image'->'Vehicle_camera'->'Vehicle_lidar'->'Vehicle_novatel'

           Transformation                                   Function name
    infrastructure-lidar to world          ->      Coord_Infrastructure_lidar2world()
    vehicle-lidar to world                 ->      Coord_Vehicle_lidar2world()
    infrastructure-lidar to vehicle-lidar  ->      Coord_Infrastructure_lidar2Vehicle_lidar()
    world to vehicle-lidar                 ->      Coord_world2vehicel_lidar()


    Transformation equation
        a^p=a^R_b*P_b+a^P_b0
        reverse:  P_b=vers(a^R_b)a^p-vers(a^R_b)(a^P_b0)
    """

    def __init__(self, from_coord, to_coord, path_root, infra_name, veh_name,sys_delta_x,sys_delta_y):
        # self.transformer = Transformation()
        self.from_coord = from_coord
        self.to_coord = to_coord
        self.path_root = path_root
        self.infra_name = infra_name
        self.veh_name = veh_name

        # Hard code for time-compensation late fusion
        #self.delta_x = None
        #self.delta_y = None
        self.delta_x = sys_delta_x
        self.delta_y=sys_delta_y
    def __call__(self, point):

        path_all = {
            "path_root": self.path_root,
            "path_lidar2world": "infrastructure-side/calib/virtuallidar_to_world/" + self.infra_name + ".json",
            "path_lidar2novatel": "vehicle-side/calib/lidar_to_novatel/" + self.veh_name + ".json",
            "path_novatel2world": "vehicle-side/calib/novatel_to_world/" + self.veh_name + ".json",
            "i_lidar2i_camera":"infrastructure-side/calib/virtuallidar_to_camera/"+ self.infra_name + ".json",
            "v_lidar2v_camera":"vehicle-side/calib/lidar_to_camera/"+ self.veh_name + ".json"
        }

        rotation, translation = self.forward(self.from_coord, self.to_coord, path_all)
        return self.point_transformation(point, rotation, translation)

    def forward(self, from_coord, to_coord, path_all):
        coord_list = ["Infrastructure_lidar", "World", "Vehicle_lidar","Infrastructure_camera","Vehicle_camera"]
        if (from_coord in coord_list) and (to_coord in coord_list):
            if from_coord == "Infrastructure_lidar" and to_coord == "World":
                rotation, translation = self.Coord_Infrastructure_lidar2world(path_all)
                return rotation, translation
            if from_coord == "Vehicle_lidar" and to_coord == "World":
                rotation, translation = self.Coord_Vehicle_lidar2world(path_all)
                return rotation, translation
            if from_coord == "Infrastructure_lidar" and to_coord == "Vehicle_lidar":
                rotation, translation = self.Coord_Infrastructure_lidar2Vehicle_lidar(path_all)
                return rotation, translation
            if from_coord == "World" and to_coord == "Vehicle_lidar":
                rotation, translation = self.Coord_world2vehicel_lidar(path_all)
                return rotation, translation
            if from_coord == "Vehicle_lidar" and to_coord == "Infrastructure_camera":
                rotation, translation = self.Vehicle_lidar2iCamera(path_all)
                K=self.get_K(path_all)
                return rotation, translation, K
            if from_coord == "Infrastructure_lidar" and to_coord == "Infrastructure_camera":
                rotation, translation = self.ilidar2iCamera(path_all)
                
                K=self.get_K(path_all)
                return rotation, translation, K
            if from_coord == "Vehicle_lidar" and to_coord == "Vehicle_camera":
                rotation, translation = self.vlidar2vCamera(path_all)
                
                K=self.get_vK(path_all)
                return rotation, translation, K
            
        else:
            raise ("error: wrong coordinate name")
        
    def get_rot_trans(self):
        path_all = {
            "path_root": self.path_root,
            "path_lidar2world": "infrastructure-side/calib/virtuallidar_to_world/" + self.infra_name + ".json",
            "path_lidar2novatel": "vehicle-side/calib/lidar_to_novatel/" + self.veh_name + ".json",
            "path_novatel2world": "vehicle-side/calib/novatel_to_world/" + self.veh_name + ".json",
            "i_lidar2i_camera": "infrastructure-side/calib/virtuallidar_to_camera/"+ self.infra_name + ".json",
            "path_K":"infrastructure-side/calib/camera_intrinsic/"+ self.infra_name + ".json",
            "vpath_K":"vehicle-side/calib/camera_intrinsic/"+ self.veh_name + ".json",
            "v_lidar2v_camera":"vehicle-side/calib/lidar_to_camera/"+ self.veh_name + ".json"
        }
        
        return self.forward(self.from_coord, self.to_coord, path_all)

    def rev_matrix(self, R):
        R = np.matrix(R)
        rev_R = R.I
        rev_R = np.array(rev_R)
        return rev_R

    def muilt_coord(self, rotationA2B, translationA2B, rotationB2C, translationB2C):
        rotationA2B = np.array(rotationA2B).reshape(3, 3)
        rotationB2C = np.array(rotationB2C).reshape(3, 3)
        rotation = np.dot(rotationB2C, rotationA2B)
        translationA2B = np.array(translationA2B).reshape(3, 1)
        translationB2C = np.array(translationB2C).reshape(3, 1)
        translation = np.dot(rotationB2C, translationA2B) + translationB2C
        return rotation, translation

    def reverse(self, rotation, translation):
        rev_rotation = self.rev_matrix(rotation)
        rev_translation = -np.dot(rev_rotation, translation)
        return rev_rotation, rev_translation

    def trans(self, input_point, translation, rotation):
        translation = np.array(translation).reshape(3, 1)
        rotation = np.array(rotation).reshape(3, 3)
        for point in input_point:
            output_point = np.dot(rotation, input_point.reshape(3, 1)).reshape(3) + np.array(translation).reshape(3)
        return np.array(output_point)

    def get_lidar2novatel(self, path_lidar2novatel):  # vehicle side
        lidar2novatel = self.read_json(path_lidar2novatel)
        rotation = lidar2novatel["transform"]["rotation"]
        translation = lidar2novatel["transform"]["translation"]
        return rotation, translation

    def get_novatel2world(self, path_novatel2world):  # vehicle side
        novatel2world = self.read_json(path_novatel2world)
        rotation = novatel2world["rotation"]
        translation = novatel2world["translation"]
        return rotation, translation

    def get_lidar2world(self, path_lidar2world):  # Infrastructure side, lidar to word
        lidar2world = self.read_json(path_lidar2world)
        rotation = lidar2world["rotation"]
        translation = lidar2world["translation"]
        #delta_x = lidar2world["relative_error"]["delta_x"]
        #delta_y = lidar2world["relative_error"]["delta_y"]
        #if delta_x == "":
        #    delta_x = 0
        #if delta_y == "":
            #delta_y = 0
            
        delta_x = 0
        delta_y = 0
        return rotation, translation, delta_x, delta_y
    
    def get_i_lidar2i_camera(self, path_i_lidar2i_camera):  # vehicle side
        i_lidar2i_camera = self.read_json(path_i_lidar2i_camera)
        rotation = i_lidar2i_camera["rotation"]
        translation = i_lidar2i_camera["translation"]
        #rotation, translation = self.reverse(rotation, translation)
        return rotation, translation
    
    def get_v_lidar2v_camera(self, path_v_lidar2v_camera):  # vehicle side
        v_lidar2v_camera = self.read_json(path_v_lidar2v_camera)
        rotation = v_lidar2v_camera["rotation"]
        translation = v_lidar2v_camera["translation"]
        return rotation, translation

    def get_K(self, path_all):  # vehicle side
        K_dic = self.read_json(os.path.join(path_all["path_root"], path_all["path_K"]))
        K = K_dic["cam_K"]

        return np.array(K).reshape([3, 3])
    
    
    def get_vK(self, path_all):  # vehicle side
        K_dic = self.read_json(os.path.join(path_all["path_root"], path_all["vpath_K"]))
        K = K_dic["cam_K"]

        return np.array(K).reshape([3, 3])
    
    
    def read_json(self, path_json):
        with open(path_json, "r") as load_f:
            my_json = json.load(load_f)
        return my_json

    def Coord_Infrastructure_lidar2world(self, path_all):
        rotation, translation, delta_x, delta_y = self.get_lidar2world(
            os.path.join(path_all["path_root"], path_all["path_lidar2world"])
        )
        return rotation, translation

    def Coord_world2vehicel_lidar(self, path_all):
        # world to novatel
        rotation, translation = self.get_novatel2world(
            os.path.join(path_all["path_root"], path_all["path_novatel2world"])
        )
        rotationA2B, translationA2B = self.reverse(rotation, translation)
        # novatel to lidar
        rotation, translation = self.get_lidar2novatel(
            os.path.join(path_all["path_root"], path_all["path_lidar2novatel"])
        )
        rotationB2C, translationB2C = self.reverse(rotation, translation)
        new_rotationA2C, new_translationA2C = self.muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)
        return new_rotationA2C, new_translationA2C

    def Coord_Vehicle_lidar2world(self, path_all):
        rotationA2B, translationA2B = self.get_lidar2novatel(
            os.path.join(path_all["path_root"], path_all["path_lidar2novatel"])
        )
        rotationB2C, translationB2C = self.get_novatel2world(
            os.path.join(path_all["path_root"], path_all["path_novatel2world"])
        )
        new_rotationA2C, new_translationA2C = self.muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)

        return new_rotationA2C, new_translationA2C

    def Coord_Infrastructure_lidar2Vehicle_lidar(self, path_all):
        rotationA2B, translationA2B, delta_x, delta_y = self.get_lidar2world(
            os.path.join(path_all["path_root"], path_all["path_lidar2world"])
        )

        delta_x = self.delta_x
        delta_y = self.delta_y

        translationA2B = translationA2B + np.array([delta_x, delta_y, 0]).reshape(3, 1)
        rotationB2C, translationB2C = self.Coord_world2vehicel_lidar(path_all)
        new_rotationA2C, new_translationA2C = self.muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)

        return new_rotationA2C, new_translationA2C
    
    def Vehicle_lidar2iCamera(self, path_all):
        rotation, translation=self.Coord_Infrastructure_lidar2Vehicle_lidar(path_all)
        rotationA2B,translationA2B=self.reverse(rotation, translation)
        rotationB2C, translationB2C = self.get_i_lidar2i_camera(os.path.join(path_all["path_root"], path_all["i_lidar2i_camera"]))
        new_rotationA2C, new_translationA2C = self.muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)

        return new_rotationA2C, new_translationA2C
    
    def ilidar2iCamera(self, path_all):
        
        rotation, translation = self.get_i_lidar2i_camera(os.path.join(path_all["path_root"], path_all["i_lidar2i_camera"]))
        rotation=np.array(rotation).reshape(3, 3)
        translation = np.array(translation).reshape(3, 1)
        return rotation, translation
    
    def vlidar2vCamera(self, path_all):
        
        rotation, translation = self.get_v_lidar2v_camera(os.path.join(path_all["path_root"], path_all["v_lidar2v_camera"]))
        rotation=np.array(rotation).reshape(3, 3)
        translation = np.array(translation).reshape(3, 1)
        return rotation, translation

    def point_transformation(self, input_box, rotation, translation):
        translation = np.array(translation).reshape(3, 1)
        rotation = np.array(rotation).reshape(3, 3)
        output = []
        for box in input_box:
            if len(box) == 3:
                output.append(np.dot(rotation, box.reshape(3, 1)).reshape(3) + np.array(translation).reshape(3))
                continue
            output_point = []
            for point in box:
                output_point.append(np.dot(rotation, point.reshape(3, 1)).reshape(3) + np.array(translation).reshape(3))
            output.append(output_point)

        return np.array(output)

    def single_point_transformation(self, input_point):
        path_all = {
            "path_root": self.path_root,
            "path_lidar2world": "infrastructure-side/calib/virtuallidar_to_world/" + self.infra_name + ".json",
            "path_lidar2novatel": "vehicle-side/calib/lidar_to_novatel/" + self.veh_name + ".json",
            "path_novatel2world": "vehicle-side/calib/novatel_to_world/" + self.veh_name + ".json",
        }

        rotation, translation = self.forward(self.from_coord, self.to_coord, path_all)
        input_point = np.array(input_point).reshape(3, 1)
        translation = np.array(translation).reshape(3, 1)
        rotation = np.array(rotation).reshape(3, 3)
        output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)

        return output_point

class Frame(dict, ABC):
    def __init__(self, path, info_dict):
        self.path = path
        for key in info_dict:
            self.__setitem__(key, info_dict[key])

    @abstractmethod
    def point_cloud(self, **args):
        raise NotImplementedError

    @abstractmethod
    def image(self, **args):
        raise NotImplementedError

class InfFrame(Frame):
    def __init__(self, path, inf_dict, tmp_key="tmps"):
        super().__init__(path, inf_dict)
        self.id = {}
        self.id["lidar"] = inf_dict["pointcloud_path"][-10:-4]
        self.id["camera"] = inf_dict["image_path"][-10:-4]
        self.tmp = "../cache/" + tmp_key + "/tmp_i_" + self.id["lidar"] + ".bin"
        if not osp.exists("../cache/" + tmp_key):
            os.system("mkdir ../cache/" + tmp_key)

    def point_cloud(self, data_format="array"):
        points, _ = read_pcd(osp.join(self.path, self.get("pointcloud_path")))
        if data_format == "array":
            return points, _
        elif data_format == "file":
            if not osp.exists(self.tmp):
                points.tofile(self.tmp)
            return self.tmp
        elif data_format == "tensor":
            return torch.tensor(points)

    def image(self, data_format="rgb"):
        image_array = read_jpg(osp.join(self.path, self.get("image_path")))
        #print(osp.join(self.path, self.get("image_path")))
        if data_format == "array":
            return image_array
        elif data_format == "file":
            if not osp.exists(self.tmp):
                image_array.copy(self.tmp)
            return self.tmp
        elif data_format == "tensor":
            return torch.tensor(image_array)

    def transform(self, from_coord="", to_coord=""):
        """
        This function serves to calculate the Transformation Matrix from 'from_coord' to 'to_coord'
        coord_list=['Infrastructure_image','Infrastructure_camera','Infrastructure_lidar',
                       'world', 'Vehicle_image','Vehicle_camera','Vehicle_lidar',
                       'Vehicle_novatel']
        Args:
            from_coord(str): element in the coord_list
            to_coord(str):  element in coord_list
        Return:
            Transformation_Matrix: Transformation Matrix from 'from_coord' to 'to_coord'
        """
        infra_name = self.id["camera"]
        trans = Coord_transformation(from_coord, to_coord, "/".join(self.path.split("/")[:-2]), infra_name, "")
        return trans
    
class VehFrame(Frame):
    def __init__(self, path, veh_dict, tmp_key="tmps"):
        super().__init__(path, veh_dict)
        self.id = {}
        self.id["lidar"] = veh_dict["pointcloud_path"][-10:-4]
        self.id["camera"] = veh_dict["image_path"][-10:-4]
        self.tmp = "../cache/" + tmp_key + "/tmp_v_" + self.id["lidar"] + ".bin"
        if not osp.exists("../cache/" + tmp_key):
            os.system("mkdir -p ../cache/" + tmp_key)

    def point_cloud(self, data_format="array"):
        points, _ = read_pcd(osp.join(self.path, self.get("pointcloud_path")))
        #print(osp.join(self.path, self.get("pointcloud_path")))
        if data_format == "array":
            return points, _
        elif data_format == "file":
            if not osp.exists(self.tmp):
                points.tofile(self.tmp)
            return self.tmp
        elif data_format == "tensor":
            return torch.tensor(points)

    def image(self, data_format="rgb"):
        image_array = read_jpg(osp.join(self.path, self.get("image_path")))
        if data_format == "array":
            return image_array
        elif data_format == "file":
            if not osp.exists(self.tmp):
                image_array.tofile(self.tmp)
            return self.tmp
        elif data_format == "tensor":
            return torch.tensor(image_array)


class VICFrame(Frame):
    def __init__(self, path, info_dict, veh_frame, inf_frame, time_diff, delta_x,delta_y,offset=None):
        # TODO: build vehicle frame and infrastructure frame
        super().__init__(path, info_dict)
        self.veh_frame = veh_frame
        self.inf_frame = inf_frame
        self.time_diff = time_diff
        self.transformation = None
        self.delta_x=delta_x
        self.delta_y=delta_y


    def vehicle_frame(self):
        return self.veh_frame

    def infrastructure_frame(self):
        return self.inf_frame


    def transform(self, from_coord="", to_coord=""):
        """
        This function serves to calculate the Transformation Matrix from 'from_coord' to 'to_coord'
        coord_list=['Infrastructure_image','Infrastructure_camera','Infrastructure_lidar',
                       'world', 'Vehicle_image','Vehicle_camera','Vehicle_lidar',
                       'Vehicle_novatel']
        Args:
            from_coord(str): element in the coord_list
            to_coord(str):  element in coord_list
        Return:
            Transformation_Matrix: Transformation Matrix from 'from_coord' to 'to_coord'
        """
        veh_name = self.veh_frame["image_path"][-10:-4]
        infra_name = self.inf_frame["image_path"][-10:-4]
        sys_delta_x=self.delta_x
        sys_delta_y=self.delta_y
        trans = Coord_transformation(from_coord, to_coord, self.path, infra_name, veh_name,sys_delta_x,sys_delta_y)
        return trans



class DatasetI2V_DAIR_V2X_Camera_Sequence(Dataset):

    def __init__(self, path, split_path,
                 max_t=1.5, max_r=20., split='val',
                 sensortype="lidar", Time_sequence=5):
        super(DatasetI2V_DAIR_V2X_Camera_Sequence, self).__init__()

        self.path = path
        self.split = split
        self.max_r = max_r  # Max rotation in degrees
        self.max_t = max_t  # Max translation in meters
        self.Time_sequence = Time_sequence  # Temporal sequence length
        self.idexes = 0  # Index counter for tracking group-wise random params

        # === Load infrastructure and vehicle-side metadata info ===
        self.inf_path2info = build_path_to_info(
            "infrastructure-side",
            load_json(osp.join(path, "infrastructure-side/data_info.json")),
            sensortype,
        )
        self.veh_path2info = build_path_to_info(
            "vehicle-side",
            load_json(osp.join(path, "vehicle-side/data_info.json")),
            sensortype,
        )

        # === Load cooperative frame pairs (annotations for both sides) ===
        frame_pairs = load_json(osp.join(path, "cooperative/data_info.json"))
        frame_pairs = self.get_split(split_path, split, frame_pairs)

        # === Initialize container for processed data ===
        self.data = []
        self.inf_frames = {}
        self.veh_frames = {}
        self.random_params = {}
        self.val_RT = []  # Pre-generated rotation/translation for val/test

        # === Filter valid samples by distance ===
        for elem in frame_pairs:
            if sensortype == "lidar":
                inf_frame = self.inf_path2info[elem["infrastructure_pointcloud_path"]]
                veh_frame = self.veh_path2info[elem["vehicle_pointcloud_path"]]
            elif sensortype == "camera":
                inf_frame = self.inf_path2info[elem["infrastructure_image_path"]]
                veh_frame = self.veh_path2info[elem["vehicle_image_path"]]
                get_annos(path, "infrastructure-side", inf_frame, "camera")
                get_annos(path, "vehicle-side", veh_frame, "camera")

            delta_x = elem['system_error_offset']['delta_x']
            delta_y = elem['system_error_offset']['delta_y']

            inf_frame = InfFrame(path + "/infrastructure-side/", inf_frame)
            veh_frame = VehFrame(path + "/vehicle-side/", veh_frame)
            vic_frame = VICFrame(path, elem, veh_frame, inf_frame, 0, delta_x, delta_y)

            # Compute ground-truth transformation
            trans = vic_frame.transform('Vehicle_lidar', 'Infrastructure_camera')
            rotation, translation, K = trans.get_rot_trans()
            translation = translation.flatten()
            distance = np.linalg.norm(translation)

            if split in ['val', 'test'] and distance <= 10:
                self.data.append(vic_frame)
            elif split == 'train' and distance <= 50:
                self.data.append(vic_frame)

        # Trim data to be divisible by Time_sequence
        new_length = (len(self.data) // self.Time_sequence) * self.Time_sequence
        self.data = self.data[:new_length]

        # === Load or generate fixed val/test RT errors ===
        if split in ['val', 'test']:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            val_RT_file = os.path.join(current_dir, f'val_RT_left_seq10_{max_r:.2f}_{max_t:.2f}.csv')

            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])

                for i in range(0, len(self.data), Time_sequence):
                    # Generate one shared RT noise for each sequence
                    rotz = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)

                    for j in range(Time_sequence):
                        index = i + j
                        if index < len(self.data):
                            val_RT_file.writerow([index, transl_x, transl_y, transl_z, rotx, roty, rotz])
                            self.val_RT.append([float(index), transl_x, transl_y, transl_z, rotx, roty, rotz])

            assert len(self.val_RT) == len(self.data), "Mismatch in RT size and data"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # === Sample a new RT noise ===
        if self.split == 'train':
            # Group-wise noise: apply same RT to each sequence
            group_idx = self.idexes % self.Time_sequence
            self.idexes += 1
            if group_idx == self.Time_sequence - 1:
                self.idexes = 0

            if group_idx == 0:
                max_angle = self.max_r
                rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
                roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
                rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
                transl_x = np.random.uniform(-self.max_t, self.max_t)
                transl_y = np.random.uniform(-self.max_t, self.max_t)
                transl_z = np.random.uniform(-self.max_t, self.max_t)

                self.random_params = {
                    'rotz': rotz, 'roty': roty, 'rotx': rotx,
                    'transl_x': transl_x, 'transl_y': transl_y, 'transl_z': transl_z
                }
            else:
                rotz = self.random_params['rotz']
                roty = self.random_params['roty']
                rotx = self.random_params['rotx']
                transl_x = self.random_params['transl_x']
                transl_y = self.random_params['transl_y']
                transl_z = self.random_params['transl_z']
        else:
            # For val/test, use pre-generated RT
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # === Convert to rotation + translation matrix ===
        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))
        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        img = self.data[idx].inf_frame.image(data_format="tensor")
        points = self.data[idx].veh_frame.point_cloud(data_format="tensor")

        trans = self.data[idx].transform('Vehicle_lidar', 'Infrastructure_camera')
        rotation, translation, K = trans.get_rot_trans()
        translation = translation.flatten()
        rotated_pc = np.dot(points[:, :3].numpy(), rotation.T)
        transformed_pc = rotated_pc + translation
        pc_org = torch.from_numpy(transformed_pc).T  # (3, N)

        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_org, homogeneous), 0).float()  # (4, N)

        
        if self.split == 'test':
            sample = {
                'rgb': img, 'point_cloud': pc_in, 'calib': K,
                'tr_error': T, 'rot_error': R,
                'initial_RT': initial_RT
            }
        else:
            sample = {
                'rgb': img, 'point_cloud': pc_in, 'calib': K,
                'tr_error': T, 'rot_error': R
            }

        return sample

    def get_split(self, split_path, split, frame_pairs):
        """Return subset of frame_pairs belonging to given split (train/val/test)"""
        if osp.exists(split_path):
            split_data = load_json(split_path)
        else:
            print("Split File Doesn't Exists!")
            raise Exception

        if split in ["train", "val", "test"]:
            split_data = split_data["cooperative_split"][split]
        else:
            print("Split Method Doesn't Exists!")
            raise Exception

        frame_pairs_split = []
        for frame_pair in frame_pairs:
            veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
            if veh_frame_idx in split_data:
                frame_pairs_split.append(frame_pair)

        return frame_pairs_split


  
class DatasetI2V_DAIR_V2X_Camera(Dataset):
    def __init__(self, path, split_path,
                 max_t=1.5, max_r=20., split='val',
                 sensortype="lidar"):
        super(DatasetI2V_DAIR_V2X_Camera, self).__init__()
        self.path = path
        self.split = split

        # Build mappings from file paths to frame info dicts
        self.inf_path2info = build_path_to_info(
            "infrastructure-side",
            load_json(osp.join(path, "infrastructure-side/data_info.json")),
            sensortype,
        )
        self.veh_path2info = build_path_to_info(
            "vehicle-side",
            load_json(osp.join(path, "vehicle-side/data_info.json")),
            sensortype,
        )

        # Load and filter frame pairs based on split
        frame_pairs = load_json(osp.join(path, "cooperative/data_info.json"))
        frame_pairs = self.get_split(split_path, split, frame_pairs)

        self.data = []
        self.inf_frames = {}
        self.veh_frames = {}
        self.val_RT = []  # For storing predefined calibration errors in validation/test
        self.max_r = max_r  # Maximum rotation perturbation in degrees
        self.max_t = max_t  # Maximum translation perturbation in meters

        for elem in frame_pairs:
            # Choose lidar or camera frame info
            if sensortype == "lidar":
                inf_frame = self.inf_path2info[elem["infrastructure_pointcloud_path"]]
                veh_frame = self.veh_path2info[elem["vehicle_pointcloud_path"]]
            elif sensortype == "camera":
                inf_frame = self.inf_path2info[elem["infrastructure_image_path"]]
                veh_frame = self.veh_path2info[elem["vehicle_image_path"]]
                get_annos(path, "infrastructure-side", inf_frame, "camera")
                get_annos(path, "vehicle-side", veh_frame, "camera")

            # System offset error for ego-motion compensation
            delta_x = elem['system_error_offset']['delta_x']
            delta_y = elem['system_error_offset']['delta_y']

            # Create frame wrappers
            inf_frame = InfFrame(path + "/infrastructure-side/", inf_frame)
            veh_frame = VehFrame(path + "/vehicle-side/", veh_frame)

            # Cooperative frame object, includes both vehicle and infrastructure frames
            vic_frame = VICFrame(path, elem, veh_frame, inf_frame, 0, delta_x, delta_y)

            # Get transform between vehicle lidar and infrastructure camera
            trans = vic_frame.transform('Vehicle_lidar','Infrastructure_camera')
            rotation, translation, K = trans.get_rot_trans()
            translation = translation.flatten()
            distance = np.linalg.norm(translation)

            # Optional filtering based on distance
            # if distance < 70:
            #     self.data.append(vic_frame)
            self.data.append(vic_frame)

        # If validation or test split, load or create fixed perturbation values
        if split == 'val' or split == 'test':
            print(max_r)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            val_RT_file = os.path.join(current_dir, f'val_RT_left_seq10_{max_r:.2f}_{max_t:.2f}.csv')

            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                # Generate new calibration perturbation values and save them
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.data)):
                    rotz = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                        rotx, roty, rotz])

            assert len(self.val_RT) == len(self.data), "Mismatch in test RT entries"

    def __len__(self):
        # Return number of frame pairs
        return len(self.data)

    def __getitem__(self, idx):
        # Generate random transformation perturbation during training
        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
        else:
            # Use predefined perturbation for val/test
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # Convert to rotation and translation using Blender-style mathutils
        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        # Invert the pose to simulate calibration error
        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        img = self.data[idx].inf_frame.image(data_format="tensor")

        points = self.data[idx].veh_frame.point_cloud(data_format="tensor")
        trans = self.data[idx].transform('Vehicle_lidar','Infrastructure_camera')
        rotation, translation, K = trans.get_rot_trans()
        translation = translation.flatten()
        rotated_point_cloud = np.dot(points[:, :3].numpy(), rotation.T)
        transformed_point_cloud = rotated_point_cloud + translation
        pc_org = torch.from_numpy(transformed_point_cloud).T

        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_org, homogeneous), 0).float()

        if self.split == 'test':
            sample = {
                'rgb': img,
                'point_cloud': pc_in,
                'calib': K,
                'tr_error': T,
                'rot_error': R,
                'initial_RT': initial_RT
            }
        else:
            sample = {
                'rgb': img,
                'point_cloud': pc_in,
                'calib': K,
                'tr_error': T,
                'rot_error': R
            }

        return sample

    def get_split(self, split_path, split, frame_pairs):
        """
        Load the frame pair list for a given data split.
        """
        if osp.exists(split_path):
            split_data = load_json(split_path)
        else:
            print("Split File Doesn't Exist!")
            raise Exception

        if split in ["train", "val", "test"]:
            split_data = split_data["cooperative_split"][split]
        else:
            print("Split Type Not Supported!")
            raise Exception

        # Filter frame pairs based on vehicle frame filename
        frame_pairs_split = []
        for frame_pair in frame_pairs:
            veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
            if veh_frame_idx in split_data:
                frame_pairs_split.append(frame_pair)

        return frame_pairs_split
    


class DatasetI2V_tumtraf_Sequence(Dataset):

    def __init__(self, path,
                 max_t=1.5, max_r=20., split='val',
                 Time_sequence=5):
        super(DatasetI2V_tumtraf_Sequence, self).__init__()
        
        
        
        input_folder_path_images1 = osp.join(path, 'images/s110_camera_basler_south1_8mm')
        input_folder_path_images2 = osp.join(path, 'images/s110_camera_basler_south2_8mm')
        input_folder_path_images3 = osp.join(path, 'images/s110_camera_basler_north_8mm')
        
        
        
        input_folder_path_point_clouds = osp.join(path, 'point_clouds/vehicle_lidar_robosense')
            
        input_folder_path_v2i_transformation_matrices =  osp.join(path, 'labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
            
        self.camera_id = 's110_camera_basler_south1_8mm'
        self.lidar_id = 's110_lidar_ouster_south'
        

        types = ("*.jpg", "*.png")  # the tuple of image file types
        input_image_file_paths1 = []
        input_image_file_paths2 = []
        input_image_file_paths3 = []
        input_image_file_paths4 = []
        for files in types:
            input_image_file_paths1.extend(sorted(glob.glob(input_folder_path_images1 + "/" + files)))
        print("Found {} images in {}".format(len(input_image_file_paths1), input_folder_path_images1))
        print(glob.glob(input_folder_path_images1 + "/" + files))
        
        
        for files in types:
            input_image_file_paths2.extend(sorted(glob.glob(input_folder_path_images2 + "/" + files)))
        print("Found {} images in {}".format(len(input_image_file_paths2), input_folder_path_images2))
        print(glob.glob(input_folder_path_images2 + "/" + files))
        
        for files in types:
            input_image_file_paths3.extend(sorted(glob.glob(input_folder_path_images3 + "/" + files)))
        print("Found {} images in {}".format(len(input_image_file_paths3), input_folder_path_images3))
        print(glob.glob(input_folder_path_images3 + "/" + files))
        
        
        
        
        if input_folder_path_point_clouds != "":
            point_cloud_file_paths = sorted(glob.glob(input_folder_path_point_clouds + "/*.pcd"))
            print("Found {} point cloud files.".format(len(point_cloud_file_paths)))
        else:
            point_cloud_file_paths = [""] * len(input_image_file_paths1)

        

        if input_folder_path_v2i_transformation_matrices is not None:
            v2i_transformation_matrices_file_paths = sorted(
                glob.glob(input_folder_path_v2i_transformation_matrices + "/*.json"))
            print("Found {} detection files.".format(len(v2i_transformation_matrices_file_paths)))
        else:
            v2i_transformation_matrices_file_paths = [""] * len(input_image_file_paths1)

        
        
        filtered_image_file_paths1 = []
        filtered_image_file_paths2 = []
        filtered_image_file_paths3 = []
        
        filtered_point_cloud_file_paths = []
        filtered_v2i_transformation_matrices_file_paths = []
        for image_file_path1,image_file_path2,image_file_path3, point_cloud_file_path, v2i_transformation_matrices_file_path in zip(
                input_image_file_paths1,
                input_image_file_paths2,
                input_image_file_paths3,
                
                point_cloud_file_paths,
                v2i_transformation_matrices_file_paths
            ):
            
            v2i_transformation_matrices_json = json.load(open(v2i_transformation_matrices_file_path))
            for frame_id, frame_obj in v2i_transformation_matrices_json["openlabel"]["frames"].items():
                if "vehicle_lidar_robosense_to_s110_lidar_ouster_south" in frame_obj["frame_properties"]["transforms"]:
                    v2i_matrix = np.array(
                        frame_obj["frame_properties"]["transforms"][
                            "vehicle_lidar_robosense_to_s110_lidar_ouster_south"][
                            "transform_src_to_dst"]["matrix4x4"])
            
                
               
            translation_vector = v2i_matrix[:3, 3]
            horizontal_distance = np.linalg.norm(translation_vector)  
            
                
            if horizontal_distance <= 50:
                filtered_image_file_paths1.append(image_file_path1)
                filtered_image_file_paths2.append(image_file_path2)
                filtered_image_file_paths3.append(image_file_path3)
                
                filtered_point_cloud_file_paths.append(point_cloud_file_path)
                filtered_v2i_transformation_matrices_file_paths.append(v2i_transformation_matrices_file_path)

            
        self.input_image_file_paths1 = filtered_image_file_paths1
        self.input_image_file_paths2 = filtered_image_file_paths2
        self.input_image_file_paths3 = filtered_image_file_paths3
        
        
        self.point_cloud_file_paths = filtered_point_cloud_file_paths
        self.v2i_transformation_matrices_file_paths = filtered_v2i_transformation_matrices_file_paths
            

        self.path = path
        self.split=split
        
        self.data = []
        self.inf_frames = {}
        self.veh_frames = {}
        self.random_params={}
        self.val_RT = []
        self.max_r = max_r
        self.max_t = max_t
        self.Time_sequence=Time_sequence
        self.idexes=0
        
            
        self.new_length = (len(self.input_image_file_paths1)*3 // self.Time_sequence) * self.Time_sequence
        
        if split == 'val' or split == 'test':
            print(max_r)
            
            val_RT_file =os.path.join('/home/JJ_Group/zhuyy2402/MambaV2XCalib', 
                                       f'val_RT_left_seq7_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(0, self.new_length, Time_sequence):  # 以步长为10遍历all_files
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    for j in range(Time_sequence):  
                        index = i + j
                        if index < self.new_length:  
                            val_RT_file.writerow([index, transl_x, transl_y, transl_z, rotx, roty, rotz])
                            self.val_RT.append([float(index), float(transl_x), float(transl_y), float(transl_z), float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == self.new_length, "Something wrong with test RTs"
            
    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        camera_idx=idx%3
        idx2=idx//3
        if self.split == 'train':
           
            group_idx = self.idexes %5
            self.idexes+=1
            if group_idx==4:
                self.idexes=0
            
            if group_idx==0:
                max_angle = self.max_r
                rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
                roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
                rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
                transl_x = np.random.uniform(-self.max_t, self.max_t)
                transl_y = np.random.uniform(-self.max_t, self.max_t)
                transl_z = np.random.uniform(-self.max_t, self.max_t)
               
                self.random_params = {
                    'rotz': rotz,
                    'roty': roty,
                    'rotx': rotx,
                    'transl_x': transl_x,
                    'transl_y': transl_y,
                    'transl_z': transl_z
                }
            else:
                rotz = self.random_params['rotz']
                roty = self.random_params['roty']
                rotx = self.random_params['rotx']
                transl_x = self.random_params['transl_x']
                transl_y = self.random_params['transl_y']
                transl_z = self.random_params['transl_z']
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)
        
        v2i_transformation_matrices_json = json.load(open(self.v2i_transformation_matrices_file_paths[idx2]))
        for frame_id, frame_obj in v2i_transformation_matrices_json["openlabel"]["frames"].items():
            if "vehicle_lidar_robosense_to_s110_lidar_ouster_south" in frame_obj["frame_properties"]["transforms"]:
                v2i_matrix = np.array(
                    frame_obj["frame_properties"]["transforms"][
                            "vehicle_lidar_robosense_to_s110_lidar_ouster_south"][
                            "transform_src_to_dst"]["matrix4x4"])
        
        
        
        if camera_idx==0:
            img = cv2.imread(self.input_image_file_paths1[idx2], cv2.IMREAD_UNCHANGED)
            K = np.array([[-1301.42, 0, 940.389], [0, -1299.94, 674.417], [0, 0, 1]])
            projection_matrix = np.array(
                    [
                        [-0.41205, 0.910783, -0.0262516, 15.0787],
                        [0.453777, 0.230108, 0.860893, 2.52926],
                        [0.790127, 0.342818, -0.508109, 3.67868],
                    ]
                )
        elif camera_idx==1:
            img = cv2.imread(self.input_image_file_paths2[idx2], cv2.IMREAD_UNCHANGED)
            K = np.array(
                [[1315.56, 0, 969.353], [0, 1368.35, 579.071], [0, 0, 1]]
            )
            # manual calibration, optimizing intrinsics and extrinsics
            transformation_matrix_lidar_to_base = np.array(
                [
                    [0.247006, -0.955779, -0.15961, -16.8017],
                    [0.912112, 0.173713, 0.371316, 4.66979],
                    [-0.327169, -0.237299, 0.914685, 6.4602],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            transformation_matrix_base_to_camera = np.array(
                [
                    [0.8924758822566284, 0.45096261644035174, -0.01093243630327495, 14.921784677055939],
                    [0.29913535165414396, -0.6097951995429897, -0.7339399539506467, 13.668310799382738],
                    [-0.3376460291207414, 0.6517534297474759, -0.679126369559744, -5.630430017833277],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            projection_matrix = np.matmul(
                transformation_matrix_base_to_camera, transformation_matrix_lidar_to_base
            )
            projection_matrix=projection_matrix[:-1, :]
            
            
        elif camera_idx==2:
            img = cv2.imread(self.input_image_file_paths3[idx2], cv2.IMREAD_UNCHANGED)    
            K = np.array([[1360.68, 0, 849.369], [0, 1470.71, 632.174], [0, 0, 1]])
            projection_matrix = np.array(
                [
                    [-0.564602, -0.824833, -0.0295815, -12.9358],
                    [-0.458346, 0.343143, -0.819861, 7.22666],
                    [0.686399, -0.449337, -0.571798, -6.75018],
                ],
            )
         
        img=torch.from_numpy(img)
        points,_= read_pcd(self.point_cloud_file_paths[idx2])
        
        points_3d_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        points_3d = np.matmul(v2i_matrix,points_3d_homogeneous.T).T
                    
        
        # remove rows having all zeros (131k points -> 59973 points)
        points_3d = points_3d[~np.all(points_3d == 0, axis=1)]

        # crop point cloud to 120 m range
        distances = np.array([np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2) for point in points_3d])
        points_3d = points_3d[distances < 120.0]

        points_3d = np.transpose(points_3d)
        points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)
        distances = []
        indices_to_keep = []
        for i in range(len(points_3d[0, :])):
            point = points_3d[:, i]
            distance = np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2))
            if distance > 2:
                distances.append(distance)
                indices_to_keep.append(i)

        points_3d = points_3d[:, indices_to_keep]
      
        points = np.matmul(projection_matrix, points_3d[:4, :])
        
        
        points=torch.from_numpy(points)
        
        if points.shape[0] == 3:
            homogeneous = torch.ones(points.shape[1]).unsqueeze(0)
            points = torch.cat((points, homogeneous), 0).float()
        
        
        
        
        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': points, 'calib':  K,
                      'tr_error': T, 'rot_error': R, 
                      'initial_RT': initial_RT}
        else:
            sample = {'rgb': img, 'point_cloud': points, 'calib': K,'tr_error': T, 'rot_error': R}
        return sample
    
    def get_split(self, split_path, split, frame_pairs):
        if osp.exists(split_path):
            split_data = load_json(split_path)
        else:
            print("Split File Doesn't Exists!")
            raise Exception

        if split in ["train", "val", "test"]:
            split_data = split_data["cooperative_split"][split]
        else:
            print("Split Method Doesn't Exists!")
            raise Exception

        frame_pairs_split = []
        for frame_pair in frame_pairs:
            veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
            if veh_frame_idx in split_data:
                frame_pairs_split.append(frame_pair)
        return frame_pairs_split
   
    

class DatasetI2V_tumtraf(Dataset):

    def __init__(self, path,
                 max_t=1.5, max_r=20., split='val',
                 Time_sequence=5):
        super(DatasetI2V_tumtraf, self).__init__()
        input_folder_path_images1 = osp.join(path, 'images/s110_camera_basler_south1_8mm')
        input_folder_path_images2 = osp.join(path, 'images/s110_camera_basler_south2_8mm')
        input_folder_path_images3 = osp.join(path, 'images/s110_camera_basler_north_8mm')
        
        
        
        input_folder_path_point_clouds = osp.join(path, 'point_clouds/vehicle_lidar_robosense')
            
        input_folder_path_v2i_transformation_matrices =  osp.join(path, 'labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
            
        self.camera_id = 's110_camera_basler_south1_8mm'
        self.lidar_id = 's110_lidar_ouster_south'
        
    
        types = ("*.jpg", "*.png")  # the tuple of image file types
        input_image_file_paths1 = []
        input_image_file_paths2 = []
        input_image_file_paths3 = []
        input_image_file_paths4 = []
        for files in types:
            input_image_file_paths1.extend(sorted(glob.glob(input_folder_path_images1 + "/" + files)))
        print("Found {} images in {}".format(len(input_image_file_paths1), input_folder_path_images1))
        print(glob.glob(input_folder_path_images1 + "/" + files))
        
        
        for files in types:
            input_image_file_paths2.extend(sorted(glob.glob(input_folder_path_images2 + "/" + files)))
        print("Found {} images in {}".format(len(input_image_file_paths2), input_folder_path_images2))
        print(glob.glob(input_folder_path_images2 + "/" + files))
        
        
        
        for files in types:
            input_image_file_paths3.extend(sorted(glob.glob(input_folder_path_images3 + "/" + files)))
        print("Found {} images in {}".format(len(input_image_file_paths3), input_folder_path_images3))
        print(glob.glob(input_folder_path_images3 + "/" + files))
        
        
        
        
        if input_folder_path_point_clouds != "":
            point_cloud_file_paths = sorted(glob.glob(input_folder_path_point_clouds + "/*.pcd"))
            print("Found {} point cloud files.".format(len(point_cloud_file_paths)))
        else:
            point_cloud_file_paths = [""] * len(input_image_file_paths1)

        

        if input_folder_path_v2i_transformation_matrices is not None:
            v2i_transformation_matrices_file_paths = sorted(
                glob.glob(input_folder_path_v2i_transformation_matrices + "/*.json"))
            print("Found {} detection files.".format(len(v2i_transformation_matrices_file_paths)))
        else:
            v2i_transformation_matrices_file_paths = [""] * len(input_image_file_paths1)

        
        
        filtered_image_file_paths1 = []
        filtered_image_file_paths2 = []
        filtered_image_file_paths3 = []
        
        filtered_point_cloud_file_paths = []
        filtered_v2i_transformation_matrices_file_paths = []
        for image_file_path1,image_file_path2,image_file_path3, point_cloud_file_path, v2i_transformation_matrices_file_path in zip(
                input_image_file_paths1,
                input_image_file_paths2,
                input_image_file_paths3,
                
                point_cloud_file_paths,
                v2i_transformation_matrices_file_paths
            ):
            
            v2i_transformation_matrices_json = json.load(open(v2i_transformation_matrices_file_path))
            for frame_id, frame_obj in v2i_transformation_matrices_json["openlabel"]["frames"].items():
                if "vehicle_lidar_robosense_to_s110_lidar_ouster_south" in frame_obj["frame_properties"]["transforms"]:
                    v2i_matrix = np.array(
                        frame_obj["frame_properties"]["transforms"][
                            "vehicle_lidar_robosense_to_s110_lidar_ouster_south"][
                            "transform_src_to_dst"]["matrix4x4"])
            
               
            translation_vector = v2i_matrix[:3, 3]
            horizontal_distance = np.linalg.norm(translation_vector)  
            
            if horizontal_distance <= 1000:
                filtered_image_file_paths1.append(image_file_path1)
                filtered_image_file_paths2.append(image_file_path2)
                filtered_image_file_paths3.append(image_file_path3)
                
                filtered_point_cloud_file_paths.append(point_cloud_file_path)
                filtered_v2i_transformation_matrices_file_paths.append(v2i_transformation_matrices_file_path)

           
        self.input_image_file_paths1 = filtered_image_file_paths1
        self.input_image_file_paths2 = filtered_image_file_paths2
        self.input_image_file_paths3 = filtered_image_file_paths3
        
        
        self.point_cloud_file_paths = filtered_point_cloud_file_paths
        self.v2i_transformation_matrices_file_paths = filtered_v2i_transformation_matrices_file_paths
            
        
        self.path = path
        self.split=split
        
        
        
        self.random_params={}
        self.val_RT = []
        self.max_r = max_r
        self.max_t = max_t
        self.Time_sequence=Time_sequence
        self.idexes=0
        
            
        self.new_length = len(self.input_image_file_paths1)*3 
        if split == 'val' or split == 'test':
            print(max_r)
            #val_RT_file = os.path.join('/home/JJ_Group/zhuyy2402/I2V_RAFT/0.1_1.csv')
            val_RT_file =os.path.join('/home/JJ_Group/zhuyy2402/I2V_RAFT', 
                                       f'val_RT_left_seq_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(self.new_length):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == self.new_length, "Something wrong with test RTs"
            
    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        camera_idx=idx%3
        idx2=idx//3
        
        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)
       
        v2i_transformation_matrices_json = json.load(open(self.v2i_transformation_matrices_file_paths[idx2]))
        for frame_id, frame_obj in v2i_transformation_matrices_json["openlabel"]["frames"].items():
            if "vehicle_lidar_robosense_to_s110_lidar_ouster_south" in frame_obj["frame_properties"]["transforms"]:
                v2i_matrix = np.array(
                    frame_obj["frame_properties"]["transforms"][
                            "vehicle_lidar_robosense_to_s110_lidar_ouster_south"][
                            "transform_src_to_dst"]["matrix4x4"])
        
        
        if camera_idx==0:
            img = cv2.imread(self.input_image_file_paths1[idx2], cv2.IMREAD_UNCHANGED)
            K = np.array([[-1301.42, 0, 940.389], [0, -1299.94, 674.417], [0, 0, 1]])
            projection_matrix = np.array(
                    [
                        [-0.41205, 0.910783, -0.0262516, 15.0787],
                        [0.453777, 0.230108, 0.860893, 2.52926],
                        [0.790127, 0.342818, -0.508109, 3.67868],
                    ]
                )
        elif camera_idx==1:
            img = cv2.imread(self.input_image_file_paths2[idx2], cv2.IMREAD_UNCHANGED)
            K = np.array(
                [[1315.56, 0, 969.353], [0, 1368.35, 579.071], [0, 0, 1]]
            )
            # manual calibration, optimizing intrinsics and extrinsics
            transformation_matrix_lidar_to_base = np.array(
                [
                    [0.247006, -0.955779, -0.15961, -16.8017],
                    [0.912112, 0.173713, 0.371316, 4.66979],
                    [-0.327169, -0.237299, 0.914685, 6.4602],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            # extrinsic base to south2 camera
            transformation_matrix_base_to_camera = np.array(
                [
                    [0.8924758822566284, 0.45096261644035174, -0.01093243630327495, 14.921784677055939],
                    [0.29913535165414396, -0.6097951995429897, -0.7339399539506467, 13.668310799382738],
                    [-0.3376460291207414, 0.6517534297474759, -0.679126369559744, -5.630430017833277],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            projection_matrix = np.matmul(
                transformation_matrix_base_to_camera, transformation_matrix_lidar_to_base
            )
            projection_matrix=projection_matrix[:-1, :]
            
            
        elif camera_idx==2:
            img = cv2.imread(self.input_image_file_paths3[idx2], cv2.IMREAD_UNCHANGED)    
            K = np.array([[1360.68, 0, 849.369], [0, 1470.71, 632.174], [0, 0, 1]])
            projection_matrix = np.array(
                [
                    [-0.564602, -0.824833, -0.0295815, -12.9358],
                    [-0.458346, 0.343143, -0.819861, 7.22666],
                    [0.686399, -0.449337, -0.571798, -6.75018],
                ],
            )
                     
        img=torch.from_numpy(img)
        points,_= read_pcd(self.point_cloud_file_paths[idx2])
        
        points_3d_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        points_3d = np.matmul(v2i_matrix,points_3d_homogeneous.T).T
                    
        
        # remove rows having all zeros (131k points -> 59973 points)
        points_3d = points_3d[~np.all(points_3d == 0, axis=1)]

        # crop point cloud to 120 m range
        distances = np.array([np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2) for point in points_3d])
        points_3d = points_3d[distances < 120.0]

        points_3d = np.transpose(points_3d)
        points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)
        distances = []
        indices_to_keep = []
        for i in range(len(points_3d[0, :])):
            point = points_3d[:, i]
            distance = np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2))
            if distance > 2:
                distances.append(distance)
                indices_to_keep.append(i)

        points_3d = points_3d[:, indices_to_keep]
        
    
        points = np.matmul(projection_matrix, points_3d[:4, :])
        
        
        points=torch.from_numpy(points)
        
        if points.shape[0] == 3:
            homogeneous = torch.ones(points.shape[1]).unsqueeze(0)
            points = torch.cat((points, homogeneous), 0).float()
        

        
        if self.split == 'test' :
            sample = {'rgb': img, 'point_cloud': points, 'calib':  K,
                      'tr_error': T, 'rot_error': R, 
                      'initial_RT': initial_RT}
        else:
            sample = {'rgb': img, 'point_cloud': points, 'calib': K,'tr_error': T, 'rot_error': R}
        return sample
    
    def get_split(self, split_path, split, frame_pairs):
        if osp.exists(split_path):
            split_data = load_json(split_path)
        else:
            print("Split File Doesn't Exists!")
            raise Exception

        if split in ["train", "val", "test"]:
            split_data = split_data["cooperative_split"][split]
        else:
            print("Split Method Doesn't Exists!")
            raise Exception

        frame_pairs_split = []
        for frame_pair in frame_pairs:
            veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
            if veh_frame_idx in split_data:
                frame_pairs_split.append(frame_pair)
        return frame_pairs_split
    
    
