# -*- coding: utf-8 -*-
import os.path
import folder_paths
from nodes import interrupt_processing
import json
import time
import numpy as np
import torch
import cv2
import math
from PIL import Image
from server import PromptServer
from aiohttp import web

# ====================================================================================================
# API 路由处理 - 安全的初始化方式
# ====================================================================================================
def init_server_routes():
    """安全地初始化服务器路由"""
    server_instance = PromptServer.instance
    if not hasattr(server_instance, '_openpose_routes_initialized'):
        # 避免重复初始化
        routes = server_instance.routes
        
        # 全局变量初始化
        if not hasattr(server_instance, 'PAUSED_NODES'):
            setattr(server_instance, 'PAUSED_NODES', {})
        if not hasattr(server_instance, 'POSE_3D_IMAGES'):
            setattr(server_instance, 'POSE_3D_IMAGES', {})
        if not hasattr(server_instance, 'CURRENT_POSTURE_TEXTS'):
            setattr(server_instance, 'CURRENT_POSTURE_TEXTS', {})
        
        # 获取全局变量
        PAUSED_NODES = getattr(server_instance, 'PAUSED_NODES')
        CURRENT_POSTURE_TEXTS = getattr(server_instance, 'CURRENT_POSTURE_TEXTS')
        
        # 定义API路由
        @routes.post('/openpose/update_pose')
        async def openpose_update_pose(request):
            try:
                json_data = await request.json()
                node_id = str(json_data.get('node_id'))
                pose_data = json_data.get('pose_data')
                
                if node_id in PAUSED_NODES:
                    PAUSED_NODES[node_id] = {'status': 'resume', 'data': pose_data}
                    return web.json_response({"status": "success"})
                else:
                    return web.json_response({"status": "error", "message": "Node not paused"}, status=400)
            except Exception as e:
                return web.json_response({"status": "error", "message": str(e)}, status=500)

        @routes.post('/openpose/cancel')
        async def openpose_cancel(request):
            try:
                json_data = await request.json()
                node_id = str(json_data.get('node_id'))
                
                if node_id in PAUSED_NODES:
                    PAUSED_NODES[node_id] = {'status': 'cancel'}
                    return web.json_response({"status": "success"})
                else:
                    return web.json_response({"status": "error", "message": "Node not paused"}, status=400)
            except Exception as e:
                return web.json_response({"status": "error", "message": str(e)}, status=500)

        @routes.post('/openpose/set_current_posture_text')
        async def openpose_set_current_posture_text(request):
            try:
                json_data = await request.json()
                node_id = str(json_data.get('node_id'))
                text = json_data.get('text', '')
                CURRENT_POSTURE_TEXTS[node_id] = text
                return web.json_response({"status": "success"})
            except Exception as e:
                return web.json_response({"status": "error", "message": str(e)}, status=500)

        @routes.get('/openpose/get_current_posture_text')
        async def openpose_get_current_posture_text(request):
            try:
                node_id = str(request.query.get('node_id', ''))
                text = CURRENT_POSTURE_TEXTS.get(node_id, '')
                return web.json_response({"status": "success", "text": text})
            except Exception as e:
                return web.json_response({"status": "error", "message": str(e)}, status=500)
        
        # 标记已初始化
        setattr(server_instance, '_openpose_routes_initialized', True)

# 初始化服务器路由
try:
    init_server_routes()
except Exception as e:
    print(f"API路由初始化失败: {e}")

# 获取全局变量引用
server_instance = PromptServer.instance
PAUSED_NODES = getattr(server_instance, 'PAUSED_NODES', {})
if not PAUSED_NODES:
    PAUSED_NODES = {}
setattr(server_instance, 'PAUSED_NODES', PAUSED_NODES)

CURRENT_POSTURE_TEXTS = getattr(server_instance, 'CURRENT_POSTURE_TEXTS', {})
if not CURRENT_POSTURE_TEXTS:
    CURRENT_POSTURE_TEXTS = {}
setattr(server_instance, 'CURRENT_POSTURE_TEXTS', CURRENT_POSTURE_TEXTS)


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

    def __eq__(self, __value: object) -> bool:
        return True

    def __str__(self):
        return "*"


any_type = AnyType("*")


# ====================================================================================================
# 常量定义
# ====================================================================================================
LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], 
            [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


# ====================================================================================================
# 姿态分析工具类 - 保持原始获取姿势代码不变
# ====================================================================================================
class PoseAnalyzer:
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """计算三点形成的角度 - 适配新版本numpy"""
        if p1 is None or p2 is None or p3 is None:
            return None
        
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)
        
        dot_product = np.dot(v1, v2)
        norms_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms_product == 0:
            return None
        
        cos_angle = np.clip(dot_product / norms_product, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        return float(angle)

    @staticmethod
    def get_joint_position(keypoints, idx):
        """获取指定关节的位置，带置信度检查"""
        if idx < len(keypoints):
            kp = keypoints[idx]
            if kp is not None and len(kp) >= 2:
                # 如果是包含置信度的格式 [x, y, confidence]，则检查置信度
                if len(kp) == 3 and kp[2] < 0.1:  # 置信度阈值
                    return None
                return (kp[0], kp[1]) if isinstance(kp, (list, tuple)) else kp
        return None

    @staticmethod
    def analyze_posture(keypoints):
        """姿态分析 - 保持原有逻辑不变，仅修复numpy兼容性"""
        if not keypoints or len(keypoints) < 18:
            return "无有效姿势数据"
        
        # 关键点索引：0-17 对应 COCO 18个关键点
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle, 17: neck
        
        # 提取关键点位置
        nose = PoseAnalyzer.get_joint_position(keypoints, 0)
        neck = PoseAnalyzer.get_joint_position(keypoints, 17)
        left_shoulder = PoseAnalyzer.get_joint_position(keypoints, 5)
        right_shoulder = PoseAnalyzer.get_joint_position(keypoints, 6)
        left_elbow = PoseAnalyzer.get_joint_position(keypoints, 7)
        right_elbow = PoseAnalyzer.get_joint_position(keypoints, 8)
        left_wrist = PoseAnalyzer.get_joint_position(keypoints, 9)
        right_wrist = PoseAnalyzer.get_joint_position(keypoints, 10)
        left_hip = PoseAnalyzer.get_joint_position(keypoints, 11)
        right_hip = PoseAnalyzer.get_joint_position(keypoints, 12)
        left_knee = PoseAnalyzer.get_joint_position(keypoints, 13)
        right_knee = PoseAnalyzer.get_joint_position(keypoints, 14)
        left_ankle = PoseAnalyzer.get_joint_position(keypoints, 15)
        right_ankle = PoseAnalyzer.get_joint_position(keypoints, 16)
        
        # 基本姿态判断
        posture_parts = []
        
        # 1. 判断整体姿态（站立/坐/躺/跪）
        if nose and left_hip and right_hip and left_ankle and right_ankle:
            # 计算身体主要轴线的方向
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
            head_y = nose[1] if nose else (neck[1] if neck else float('inf'))
            
            # 通过头部和髋部的垂直距离判断姿态
            vertical_distance = abs(head_y - hip_center_y)
            leg_length = abs(hip_center_y - ankle_center_y)
            
            if leg_length > 0:
                ratio = vertical_distance / leg_length
                if ratio > 0.8:  # 站立
                    posture_parts.append("站立")
                elif ratio < 0.5:  # 坐着
                    posture_parts.append("坐着")
                else:
                    posture_parts.append("半蹲")
            else:
                posture_parts.append("站立")
        else:
            posture_parts.append("站立")  # 默认
        
        # 2. 手臂姿态
        arms_description = []
        
        # 左手臂
        if left_shoulder and left_elbow and left_wrist:
            shoulder_elbow_angle = PoseAnalyzer.calculate_angle(left_shoulder, left_elbow, left_wrist)
            if shoulder_elbow_angle is not None:
                if shoulder_elbow_angle < 90:
                    arms_description.append("左臂弯曲")
                elif shoulder_elbow_angle > 160:
                    arms_description.append("左臂伸直")
        
        # 右手臂
        if right_shoulder and right_elbow and right_wrist:
            shoulder_elbow_angle = PoseAnalyzer.calculate_angle(right_shoulder, right_elbow, right_wrist)
            if shoulder_elbow_angle is not None:
                if shoulder_elbow_angle < 90:
                    arms_description.append("右臂弯曲")
                elif shoulder_elbow_angle > 160:
                    arms_description.append("右臂伸直")
        
        # 检查手臂是否举起
        if left_shoulder and left_elbow:
            if left_elbow[1] < left_shoulder[1]:
                arms_description.append("左臂上举")
        if right_shoulder and right_elbow:
            if right_elbow[1] < right_shoulder[1]:
                arms_description.append("右臂上举")
        
        if left_shoulder and left_wrist:
            if left_wrist[1] < left_shoulder[1]:
                arms_description.append("左臂高举")
        if right_shoulder and right_wrist:
            if right_wrist[1] < right_shoulder[1]:
                arms_description.append("右臂高举")
        
        if arms_description:
            posture_parts.extend(arms_description)
        
        # 3. 腿部姿态
        legs_description = []
        
        # 左腿
        if left_hip and left_knee and left_ankle:
            hip_knee_angle = PoseAnalyzer.calculate_angle(left_hip, left_knee, left_ankle)
            if hip_knee_angle is not None:
                if hip_knee_angle < 120:
                    legs_description.append("左膝弯曲")
                elif hip_knee_angle > 160:
                    legs_description.append("左腿伸直")
        
        # 右腿
        if right_hip and right_knee and right_ankle:
            hip_knee_angle = PoseAnalyzer.calculate_angle(right_hip, right_knee, right_ankle)
            if hip_knee_angle is not None:
                if hip_knee_angle < 120:
                    legs_description.append("右膝弯曲")
                elif hip_knee_angle > 160:
                    legs_description.append("右腿伸直")
        
        if legs_description:
            posture_parts.extend(legs_description)
        
        # 4. 身体方向
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_line_angle = math.degrees(math.atan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ))
            hip_line_angle = math.degrees(math.atan2(
                right_hip[1] - left_hip[1],
                right_hip[0] - left_hip[0]
            ))
            
            # 如果肩膀和臀部连线角度差异很大，可能是在转身
            angle_diff = abs(shoulder_line_angle - hip_line_angle)
            if angle_diff > 30:
                posture_parts.append("转身")
        
        # 组合最终描述
        if not posture_parts:
            return "人体姿态"
        else:
            return "、".join(posture_parts) if posture_parts else "人体姿态"

    @staticmethod
    def analyze_pose_from_json(pose_json):
        """从JSON格式的姿势数据分析姿态 - 保持原有逻辑不变"""
        if not pose_json:
            return "无姿势数据"
        
        try:
            data = json.loads(pose_json)
            people = data.get('people', [])
            
            if not people:
                return "无检测到的人体"
            
            all_descriptions = []
            for person_idx, person in enumerate(people):
                kp_flat = person.get('pose_keypoints_2d', [])
                keypoints = []
                
                # 将平面数组转换为坐标对
                for i in range(0, len(kp_flat), 3):
                    if i + 2 < len(kp_flat):
                        x = kp_flat[i]
                        y = kp_flat[i + 1]
                        conf = kp_flat[i + 2]
                        keypoints.append([x, y, conf])
                    else:
                        keypoints.append(None)
                
                # 分析每个人的姿态
                description = PoseAnalyzer.analyze_posture(keypoints)
                all_descriptions.append(f"人物{person_idx+1}: {description}")
            
            return "; ".join(all_descriptions)
        
        except Exception as e:
            return f"姿态分析出错: {str(e)}"


# ====================================================================================================
# OpenPose Editor 节点
# ====================================================================================================
class OpenPoseEditor:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {"default": ""}),
            },
            "optional": {
                "pose_image": ("IMAGE",),
                "pose_point": ("POSE_KEYPOINT",),
                "prev_image": ("IMAGE",),
                "bridge_anything": (any_type,),
                "output_width_for_dwpose": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "output_height_for_dwpose": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "scale_for_xinsr_for_dwpose": ("BOOLEAN", {"default": True}),
                "stop_for_edit": ("BOOLEAN", {"default": False, "label_on": "Pause for Edit", "label_off": "No Pause"}),
            },
            "hidden": {
                "backgroundImage": ("STRING", {"multiline": False}),
                "poses_datas": ("STRING", {"multiline": True}),
                "unique_id": "UNIQUE_ID",
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 每次都返回不同的值，强制重新执行
        return time.time()
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("dw_pose_image", "dw_comb_image", "pose_3d_image", "pose_2d_image", 
                    "dw_pose_image_width", "dw_pose_image_height", "posture_text")
    FUNCTION = "get_images"
    CATEGORY = "image"
    
    def pose_point_to_json(self, pose_point, image_tensor):
        if not pose_point or not isinstance(pose_point, list) or image_tensor.shape[0] == 0:
            return ""
        
        image_h, image_w = image_tensor.shape[1], image_tensor.shape[2]
        processed_people = []
        
        for result_dict in pose_point:
            people = result_dict.get("people", []) if isinstance(result_dict, dict) else []
            for person in people:
                original_kp = person.get("pose_keypoints_2d", [])
                body_kp = [0.0] * 54
                num = min(18, len(original_kp) // 3)
                for i in range(num):
                    base = i * 3
                    if base + 2 < len(original_kp) and original_kp[base + 2] > 0:
                        body_kp[base] = original_kp[base] * image_w
                        body_kp[base + 1] = original_kp[base + 1] * image_h
                        body_kp[base + 2] = original_kp[base + 2]
                processed_people.append({"pose_keypoints_2d": body_kp})
        
        return json.dumps({"width": int(image_w), "height": int(image_h), "people": processed_people}, indent=4)
    
    def render_dw_pose(self, pose_json, width, height, scale_for_xinsr):
        if not pose_json or not pose_json.strip():
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        try:
            data = json.loads(pose_json)
        except:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        target_w, target_h = width, height
        orig_w = data.get('width', target_w)
        orig_h = data.get('height', target_h)
        
        if orig_w > 0 and orig_h > 0:
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
        else:
            scale_x = scale_y = 1.0
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        people = data.get('people', [])
        if not people:
            return canvas
        
        target_max = max(target_w, target_h)
        base_size = max(2, int(target_max / 256))
        joint_radius = max(2, base_size)
        stick_width = max(1, base_size)
        
        if scale_for_xinsr and target_max >= 500:
            stick_width = min(stick_width * 2, 8)
        
        for person in people:
            kp_flat = person.get('pose_keypoints_2d', [])
            keypoints = []
            for i in range(0, len(kp_flat), 3):
                if i + 2 < len(kp_flat) and kp_flat[i + 2] > 0:
                    x = int(kp_flat[i] * scale_x)
                    y = int(kp_flat[i + 1] * scale_y)
                    x = max(0, min(x, target_w - 1))
                    y = max(0, min(y, target_h - 1))
                    keypoints.append((x, y))
                else:
                    keypoints.append(None)
            
            for (idx1, idx2), color in zip(LIMB_SEQ, COLORS):
                p1 = keypoints[idx1 - 1] if idx1 - 1 < len(keypoints) else None
                p2 = keypoints[idx2 - 1] if idx2 - 1 < len(keypoints) else None
                if p1 is None or p2 is None:
                    continue
                # 使用cv2.LINE_AA作为抗锯齿参数（兼容新版OpenCV）
                cv2.line(canvas, p1, p2, color, stick_width, lineType=cv2.LINE_AA)
            
            for i, kp in enumerate(keypoints):
                if kp is not None and i < len(COLORS):
                    cv2.circle(canvas, kp, joint_radius, COLORS[i], -1, lineType=cv2.LINE_AA)
        
        try:
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        except:
            return canvas
    
    def get_images(self, image, output_width_for_dwpose, output_height_for_dwpose, 
                   scale_for_xinsr_for_dwpose, stop_for_edit, backgroundImage, poses_datas,
                   bridge_anything=None, prev_image=None, pose_image=None, pose_point=None,
                   unique_id=None):
        
        # 参数默认值
        if backgroundImage is None:
            backgroundImage = ""
        if poses_datas is None:
            poses_datas = ""
        
        node_str_id = str(unique_id) if unique_id else ""
        
        # 从 pose_point 生成 poses_datas
        if pose_image is not None and pose_point is not None:  # 修复参数顺序
            temp_poses_datas = self.pose_point_to_json(pose_point, pose_image)
            if temp_poses_datas:
                poses_datas = temp_poses_datas
        
        # 更新尺寸
        if poses_datas:
            try:
                pd = json.loads(poses_datas)
                output_width_for_dwpose = pd.get("width", output_width_for_dwpose)
                output_height_for_dwpose = pd.get("height", output_height_for_dwpose)
            except:
                pass
        
        # 暂停逻辑（保留）
        if stop_for_edit and unique_id:
            node_str_id = str(unique_id)
            PAUSED_NODES[node_str_id] = {'status': 'waiting', 'initial_data': poses_datas}
            PromptServer.instance.send_sync("openpose_node_pause", {
                "node_id": node_str_id,
                "current_pose": poses_datas,
                "current_background_image": backgroundImage
            })
            
            while True:
                if node_str_id not in PAUSED_NODES:
                    break
                state = PAUSED_NODES[node_str_id]
                status = state.get('status')
                
                if status == 'resume':
                    new_pose_data = state.get('data')
                    if new_pose_data:
                        poses_datas = new_pose_data
                        try:
                            pd = json.loads(new_pose_data)
                            output_width_for_dwpose = pd.get("width", output_width_for_dwpose)
                            output_height_for_dwpose = pd.get("height", output_height_for_dwpose)
                        except:
                            pass
                    del PAUSED_NODES[node_str_id]
                    break
                elif status == 'cancel':
                    del PAUSED_NODES[node_str_id]
                    interrupt_processing()
                    # 返回正确的张量形状
                    empty_tensor = torch.zeros((1, output_height_for_dwpose, output_width_for_dwpose, 3), dtype=torch.float32)
                    return (empty_tensor, empty_tensor, empty_tensor, empty_tensor, 
                           output_width_for_dwpose, output_height_for_dwpose, "已取消")
                time.sleep(0.1)
        
        # 渲染 DWPose
        dw_pose_np = self.render_dw_pose(poses_datas, output_width_for_dwpose, 
                                          output_height_for_dwpose, scale_for_xinsr_for_dwpose)
        dw_pose_image = torch.from_numpy(dw_pose_np.astype(np.float32) / 255.0).unsqueeze(0)
        
        # 合成图
        dw_combined_image = dw_pose_image.clone()
        if backgroundImage and backgroundImage.strip():
            bg_path = folder_paths.get_annotated_filepath(backgroundImage)
            if os.path.exists(bg_path):
                try:
                    bg_pil = Image.open(bg_path).convert("RGB")
                    bg_np = np.array(bg_pil)
                    bg_resized = cv2.resize(bg_np, (output_width_for_dwpose, output_height_for_dwpose))
                    dw_gray = cv2.cvtColor(dw_pose_np, cv2.COLOR_RGB2GRAY)
                    _, mask = cv2.threshold(dw_gray, 1, 255, cv2.THRESH_BINARY)
                    combined = bg_resized.copy()
                    combined[mask != 0] = dw_pose_np[mask != 0]
                    dw_combined_image = torch.from_numpy(combined.astype(np.float32) / 255.0).unsqueeze(0)
                except:
                    pass
        
        # 3D 和 2D 图
        pose_3d_image = dw_pose_image.clone()
        pose_2d_image = dw_pose_image.clone()
        
        # ========== 修复的 posture_text 逻辑 ==========
        # 使用姿态分析器生成实际的姿态描述
        posture_text = PoseAnalyzer.analyze_pose_from_json(poses_datas)
        
        # 保存到全局字典
        if node_str_id:
            CURRENT_POSTURE_TEXTS[node_str_id] = posture_text
        
        # 构建 UI 数据
        ui_data = {
            "poses_datas": [poses_datas],
            "editdPose": [""],
            "inputPose": [""],
            "backgroundImage": [backgroundImage],
            "refresh_trigger": [str(int(time.time() * 1000))],
            "dw_pose_shape": [list(dw_pose_image.shape)],
            "combined_shape": [list(dw_combined_image.shape)],
            "dw_pose_width": [output_width_for_dwpose],
            "dw_pose_height": [output_height_for_dwpose]
        }
        
        return {
            "ui": ui_data,
            "result": (dw_pose_image, dw_combined_image, pose_3d_image, pose_2d_image,
                       output_width_for_dwpose, output_height_for_dwpose, posture_text)
        }


# ====================================================================================================
# 节点注册
# ====================================================================================================
NODE_CLASS_MAPPINGS = {
    "Nui.OpenPoseEditor": OpenPoseEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Nui.OpenPoseEditor": "whk_3DPose_edit_2D_DocKr",
}