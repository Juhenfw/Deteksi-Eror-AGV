import cv2
import numpy as np
from ultralytics import YOLO
import time
import pytz
import logging
import threading
import sys
import http.client
import json
from datetime import datetime
import os
import mysql.connector
from mysql.connector import pooling
import requests  # Added for WhatsApp API integration
from urllib.parse import quote
import torch
import queue

# ===== FIX UNICODE LOGGING ERROR =====
class UnicodeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        logging.StreamHandler.__init__(self, stream)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            msg_safe = msg.replace('🔴', '[MERAH]').replace('🟢', '[HIJAU]').replace('⚠️', '[PERINGATAN]').replace('🔄', '[REFRESH]')
            stream.write(msg_safe + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


# Konfigurasi logging dengan handler khusus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("robot_detection_detailed.log", encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)
logger = logging.getLogger("RobotDetection")

# ===== GPU CUDA Check =====
if torch.cuda.is_available():
    device = torch.device('cuda')  # Menggunakan GPU jika tersedia
    logger.info("CUDA tersedia, menggunakan GPU.")
else:
    device = torch.device('cpu')  # Jika CUDA tidak tersedia, gunakan CPU
    logger.warning("CUDA tidak tersedia, menggunakan CPU.")

# ===== WHATSAPP CONFIGURATION =====
whatsapp_config = {
    'api_url': "https://api.whatspie.com/messages",
    'device': "no_device_pengirim",  # Replace with your WhatsApp device number
    
    # Contoh Penulisan nomor
    # 'receiver': "6289988774321",  # Replace with recipient's WhatsApp number
    'receivers': [
        # "no_device_pengirim",  # First recipient
        "no_device_pengirim",  # Second recipient
        # "no_device_pengirim",   # Third recipient
        # "no_device_pengirim"
        # Add more recipients as needed
    ],
    'token': "UArENT9e23uTreMutyGJBoidkns",  # Replace with your actual WhatsPie API token
    'enabled': True  # Set to False to disable WhatsApp notifications
}


# ===== DATABASE CONFIGURATION =====
db_config = {
    'host': 'host',
    'user': 'user',
    'password': 'password',
    'database': 'database',
    'table': 'table'
}

reconnect_delay = 10  # seconds to wait before reconnecting to DB

class MySQLDatabasePool:
    def __init__(self, config):
        self.config = config
        self.pool = None
        self.create_pool()
        
    def create_pool(self):
        """Menciptakan koneksi pool MySQL"""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="robot_tracker_pool",
                pool_size=5,
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                autocommit=True,
                connect_timeout=10
            )
            logger.info("Database connection pool created successfully")
        except mysql.connector.Error as err:
            logger.error(f"Error creating connection pool: {err}")
            self.pool = None
            # Schedule reconnection
            threading.Timer(reconnect_delay, self.create_pool).start()
    
    def get_connection(self):
        """Mendapatkan koneksi dari pool dengan penanganan error yang lebih baik"""
        if not self.pool:
            self.create_pool()
            if not self.pool:
                return None
                
        try:
            return self.pool.get_connection()
        except mysql.connector.Error as err:
            logger.error(f"Failed to get connection from pool: {err}")
            # Recreate pool on failure
            self.pool = None
            self.create_pool()
            return None
    
    def insert(self, data):
        """Memasukkan data ke dalam database tanpa station field"""
        conn = self.get_connection()
        if not conn:
            logger.error("Failed to insert data - no connection available")
            return None
            
        try:
            cursor = conn.cursor()
            query = f"""
                INSERT INTO {self.config['table']} 
                (location, status, timestamp)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, data)
            conn.commit()
            last_id = cursor.lastrowid
            cursor.close()
            return last_id
        except mysql.connector.Error as err:
            logger.error(f"Database insert error: {err}")
            return None
        finally:
            conn.close()

    def close(self):
        """Menutup koneksi pool"""
        if self.pool:
            logger.info("Closing database connection pool")
            self.pool = None

# Global database connection pool
db_pool = None

# Queue untuk komunikasi antar thread
q = queue.Queue()

# Membagi area menjadi 2 plot
Area_1 = np.array([[234, 379],
                    [582, 356],
                    [700, 500],
                    [636, 1000],
                    [310, 1000]])


Area = np.array([[1329, 729],
                    [1650, 500],
                    [1920, 294],
                    [1920, 1080],
                    [1635, 1080]])


# Waktu deteksi berhenti
sec_stop = 15  # deteksi berhenti setelah 5 detik
stop_detection = {}  # Menyimpan waktu deteksi berhenti untuk setiap objek
robot_positions = {}  # Tracking posisi robot untuk deteksi gerakan kecil
robot_areas = {}     # Tracking area tempat robot berada


# Tracking robot yang pernah terdeteksi di area tertentu
robots_in_areas = {}  # {robot_id: {area_name: bool}}

# Tracking robot yang masuk Area_0 dan mengirim notifikasi
robots_at_area_0 = {}  # {robot_id: timestamp}
area_0_notifications_sent = {}  # {robot_id: bool} to track if notification sent
area_0_message_displayed = {}  # {robot_id: bool} to track if message displayed

# Toleransi gerakan yang dianggap "diam"
MOVEMENT_TOLERANCE = 4 # dalam pixel - dikurangi untuk lebih sensitif
# Toleransi waktu (dalam detik) untuk dianggap berhenti
TIME_TOLERANCE = 1.0  # akan dihitung sebagai berhenti jika lebih dari 1 detik

# Memuat model YOLO yang sudah dilatih
model_path = 'path/to/your_model.pt'
model = YOLO(model_path)

if torch.cuda.is_available():
    model = model.to('cuda')
    logger.info("Using CUDA for inference")
else:
    model = model.to('cpu')
    logger.info("Using CPU for inference")

# URL RTSP stream CCTV
username = "username"
password = "password"
ip = "ip_CCTV"
port = "port_CCTV"

use_rtsp = True

if use_rtsp:
    url = f"rtsp://{username}:{quote(password)}@{ip}:{port}"
    logger.info(f"Using RTSP stream: {ip}:{port}")
else:
    url = "path/to/video.ts"
    logger.info(f"Using video file: {url}")

# Variabel untuk tracking robot yang sedang error
robots_with_errors = set()  # Simpan robot_id yang sedang error (tidak bergerak)
error_durations = {}  # Simpan durasi error untuk robot (robot_id: duration)
whatsapp_notified_errors = set()  # Track which errors have been notified via WhatsApp


# Tambahan variabel untuk tracking point-based error detection
robot_point_positions = {}  # {robot_id: {point_index: [last_positions], timestamps: [timestamps]}}
error_points = {}  # {robot_id: {point_index: bool}} - tracks which corner points are in error state
point_stop_durations = {}  # {robot_id: {point_index: duration}}


def send_area_0_notification(robot_id):
    """Send WhatsApp notification when a robot arrives at Home Base"""
    if not whatsapp_config['enabled']:
        logger.info("WhatsApp notifications are disabled")
        return False
    
    notification_key = f"{robot_id}_area_0_{datetime.now().strftime('%Y%m%d')}"
    
    # Check if we've already sent a notification for this robot today
    if notification_key in whatsapp_notified_errors:
        logger.info(f"Home Base notification for {notification_key} already sent today")
        return False
    
    message = (
        f"✅ *AGV INFORMATION* ✅\n\n"
        # f"*Robot ID:* {robot_id}\n"
        f"*Location:* Home Base\n"
        f"*Time:* {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
        f"*Status:* Robot Telah Sampai"
    )

    successful_sends = 0
    
    # Send to each receiver in the list
    for receiver in whatsapp_config['receivers']:
        try:
            payload = json.dumps({
                "device": whatsapp_config['device'],
                "receiver": receiver,
                "type": "chat",
                "message": message,
                "simulate_typing": 1
            })
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {whatsapp_config["token"]}'
            }
            
            response = requests.request(
                "POST", 
                whatsapp_config['api_url'], 
                headers=headers, 
                data=payload
            )
            
            if response.status_code == 200:
                logger.info(f"Home Base notification sent successfully to {receiver} for {robot_id}")
                successful_sends += 1
            else:
                logger.error(f"Failed to send Home Base notification to {receiver}: {response.text}")
            
        except Exception as e:
            logger.error(f"Error sending Home Base notification to {receiver}: {str(e)}")
    
    # Mark as notified if at least one message was sent successfully
    if successful_sends > 0:
        whatsapp_notified_errors.add(notification_key)
        return True
    
    return False


def send_whatsapp_notification(robot_id, area_name, duration, error_points_str=""):
    """Send WhatsApp notification to all configured receivers when a robot has an error"""
    if not whatsapp_config['enabled']:
        logger.info("WhatsApp notifications are disabled")
        return False
    
    notification_key = f"{robot_id}_{area_name}_{datetime.now().strftime('%Y%m%d')}"
    
    # Check if we've already sent a notification for this error today
    if notification_key in whatsapp_notified_errors:
        logger.info(f"WhatsApp notification for {notification_key} already sent today")
        return False
    
    error_location = f"{error_points_str}" if error_points_str else ""
    message = (
        f"⚠️ *AGV INFORMATION ERROR* ⚠️\n\n"
        # f"*Robot ID:* {robot_id}\n"
        f"*Location:* {area_name}\n"
        # f"*Error:* Robot stopped moving{error_location}\n"
        # f"*Duration:* {duration:.1f} seconds\n"
        f"*Time:* {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n"
        f"Segera Lakukan Pengecekan"
    )

    successful_sends = 0
    
    # Send to each receiver in the list
    for receiver in whatsapp_config['receivers']:
        try:
            payload = json.dumps({
                "device": whatsapp_config['device'],
                "receiver": receiver,
                "type": "chat",
                "message": message,
                "simulate_typing": 1
            })
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {whatsapp_config["token"]}'
            }
            
            response = requests.request(
                "POST", 
                whatsapp_config['api_url'], 
                headers=headers, 
                data=payload
            )
            
            if response.status_code == 200:
                logger.info(f"WhatsApp notification sent successfully to {receiver} for {robot_id} in {area_name}")
                successful_sends += 1
            else:
                logger.error(f"Failed to send WhatsApp notification to {receiver}: {response.text}")
            
        except Exception as e:
            logger.error(f"Error sending WhatsApp notification to {receiver}: {str(e)}")
    
    # Mark as notified if at least one message was sent successfully
    if successful_sends > 0:
        whatsapp_notified_errors.add(notification_key)
        return True
    
    return False


def get_corner_points(x1, y1, x2, y2):
    """Mendapatkan 4 titik sudut dari bounding box"""
    top_left = (int(x1), int(y1))
    top_right = (int(x2), int(y1))
    bottom_left = (int(x1), int(y2))
    bottom_right = (int(x2), int(y2))
    return [top_left, top_right, bottom_left, bottom_right]


def calculate_distance(pos1, pos2):
    """Menghitung jarak antara dua posisi"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def is_in_area(point, area):
    """Memeriksa apakah titik berada dalam area polygon"""
    return cv2.pointPolygonTest(area, (point[0], point[1]), False) >= 0


def check_area_for_point(point):
    """Menentukan area untuk satu titik"""
    if is_in_area(point, Area_1):
        return "Area_1"
    elif is_in_area(point, Area_0):
        return "Area_0"
    return None


def check_area(corner_points):
    """Menentukan area berdasarkan 4 titik sudut bounding box"""
    for point in corner_points:
        area = check_area_for_point(point)
        if area:
            return area
    return None


def log_to_database(robot_id, area_name):
    """Log robot area entry to database with IN status only, omitting station field"""
    global db_pool
    
    if db_pool is None:
        logger.warning("Cannot log to database - connection pool not initialized")
        return False
    
    try:
        # Prepare data for insert - always use "IN" status only, without station field
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = (
            area_name,              # location field (using area_name here instead)
            "IN",                   # status field - always IN
            timestamp               # timestamp field
        )
        
        # Insert data into database
        record_id = db_pool.insert(data)
        
        if record_id:
            logger.info(f"Logged robot {robot_id} entry to {area_name} in database with ID {record_id}")
            return True
        else:
            logger.warning(f"Failed to log robot {robot_id} entry to database")
            return False
    
    except Exception as e:
        logger.error(f"Error logging to database: {str(e)}")
        return False


def print_robot_entry(robot_id, area_name, corner_points):
    """Mencetak informasi ketika robot memasuki area baru"""
    logger.info(f"DETEKSI ROBOT: {robot_id} terdeteksi di {area_name}")
    
    points_in_area_1 = [i for i, p in enumerate(corner_points) if is_in_area(p, Area_1)]
    points_in_area_0 = [i for i, p in enumerate(corner_points) if is_in_area(p, Area_0)]
    
    if points_in_area_0:
        logger.info(f"Titik yang masuk Area_0: {points_in_area_0}")
    if points_in_area_1:
        logger.info(f"Titik yang masuk Area_1: {points_in_area_1}")
    
    # When robot enters Area_0, log special message and track it
    if area_name == "Area_0":
        logger.info(f"Robot Telah Sampai")
        robots_at_area_0[robot_id] = datetime.now()
        area_0_message_displayed[robot_id] = False
        
        # If this robot was previously in error state, remove it
        if robot_id in robots_with_errors:
            logger.info(f"Robot {robot_id} moved to Home Base - clearing error state")
            robots_with_errors.remove(robot_id)
            if robot_id in error_durations:
                del error_durations[robot_id]
        
        # Check if we need to send notification
        if robot_id not in area_0_notifications_sent or not area_0_notifications_sent[robot_id]:
            if send_area_0_notification(robot_id):
                area_0_notifications_sent[robot_id] = True
    
    logger.info(f"Waktu: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    
    # Log entry to the database with "IN" status only
    log_to_database(robot_id, area_name)


def log_error_to_database(robot_id, area_name, duration):
    """Log robot error to database with IN status only, omitting station field"""
    global db_pool
    
    if db_pool is None:
        logger.warning("Cannot log error to database - connection pool not initialized")
        return False
    
    try:
        # Prepare data for insert - use "IN" status for all entries, without station field
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = (
            area_name,              # location field (using area_name instead)
            "IN",                   # status field - always IN
            timestamp               # timestamp field
        )
        
        # Insert data into database
        record_id = db_pool.insert(data)
        
        if record_id:
            logger.info(f"Logged robot {robot_id} error in {area_name} to database with ID {record_id}")
            return True
        else:
            logger.warning(f"Failed to log robot {robot_id} error to database")
            return False
    
    except Exception as e:
        logger.error(f"Error logging error to database: {str(e)}")
        return False


def print_robot_error(robot_id, area_name, duration, error_point_indices=None):
    """Mencetak pesan error ketika robot berhenti terlalu lama"""
    # Skip error reporting for robots in Area_0
    if area_name == "Area_0":
        return
        
    robots_with_errors.add(robot_id)
    error_durations[robot_id] = duration
    
    error_points_str = ""
    if error_point_indices:
        error_points_str = f" pada titik sudut {error_point_indices}"
    
    error_msg = (
        f"ERROR ROBOT: {robot_id} BERHENTI di {area_name}{error_points_str}\n"
        f"Durasi berhenti: {duration:.1f} detik (melebihi batas {sec_stop} detik)\n"
        f"Waktu: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
        f"TINDAKAN: Mohon periksa robot segera!"
    )
    logger.error(error_msg)
    
    # Log error to the database with "IN" status only
    log_error_to_database(robot_id, area_name, duration)
    
    # Send WhatsApp notification for the error
    send_whatsapp_notification(robot_id, area_name, duration, error_points_str)


def print_robot_moving_again(robot_id, area_name, error_point_indices=None):
    """Mencetak informasi ketika robot bergerak kembali setelah error"""
    if robot_id in robots_with_errors:
        duration = error_durations.get(robot_id, 0)
        
        error_points_str = ""
        if error_point_indices:
            error_points_str = f" pada titik sudut {error_point_indices}"
        
        recovery_msg = (
            f"PEMULIHAN: {robot_id} di {area_name}{error_points_str} mulai bergerak kembali\n"
            f"Setelah berhenti selama {duration:.1f} detik\n"
            f"Waktu: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )
        logger.info(recovery_msg)
        
        # We don't log recovery to database since we only track IN status
        
        robots_with_errors.remove(robot_id)
        if robot_id in error_durations:
            del error_durations[robot_id]


def check_point_movement(robot_id, point_index, current_point, current_time, point_area):
    """
    Check if a specific corner point of a robot is stationary
    Returns: (is_stationary, duration)
    """
    tracking_key = f"{robot_id}_{point_index}"
    
    # Initialize tracking for this point if not exists
    if robot_id not in robot_point_positions:
        robot_point_positions[robot_id] = {
            "positions": {point_index: current_point},
            "timestamps": {point_index: current_time}
        }
        return False, 0
    
    if point_index not in robot_point_positions[robot_id]["positions"]:
        robot_point_positions[robot_id]["positions"][point_index] = current_point
        robot_point_positions[robot_id]["timestamps"][point_index] = current_time
        return False, 0
    
    # Get previous position and time
    last_pos = robot_point_positions[robot_id]["positions"][point_index]
    last_time = robot_point_positions[robot_id]["timestamps"][point_index]
    
    # Calculate distance moved
    distance = calculate_distance(current_point, last_pos)
    duration = current_time - last_time
    
    # Update position regardless of movement
    robot_point_positions[robot_id]["positions"][point_index] = current_point
    
    # If point moved beyond tolerance, reset timestamp
    if distance > MOVEMENT_TOLERANCE:
        robot_point_positions[robot_id]["timestamps"][point_index] = current_time
        
        # If this point was in error, log recovery
        if robot_id in error_points and point_index in error_points.get(robot_id, {}):
            if error_points[robot_id][point_index]:
                error_points[robot_id][point_index] = False
                # Only print recovery if all points are no longer in error
                if not any(error_points[robot_id].values()):
                    error_point_indices = [idx for idx, is_error in error_points[robot_id].items() if is_error]
                    if not error_point_indices and robot_id in robots_with_errors:
                        print_robot_moving_again(robot_id, point_area, error_point_indices)
        
        return False, 0
    
    # Point is stationary
    return True, duration

# Fungsi untuk menerima stream dan memasukkan frame ke dalam queue
frame_count = 0  # Deklarasikan frame_count di luar fungsi

def Receive():
    """Thread untuk menerima frame dari RTSP stream"""
    global frame_count  # Menandakan bahwa kita menggunakan variabel global frame_count
    logger.info("Start receiving video stream")
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Gagal menerima frame")
            break
        if frame_count % 2 == 0:  # Proses hanya setiap 3 frame
            q.put(frame)  # Masukkan frame ke dalam queue
        frame_count += 1  # Increment frame_count setiap kali frame diterima
    cap.release()

# Fungsi untuk menampilkan frame dari queue
def Display():
    """Thread untuk menampilkan frame dari queue"""
    logger.info("Start displaying video stream")
    while True:
        if not q.empty():  # Jika ada frame dalam queue
            frame = q.get(timeout=1)
            frame = process_frame(frame)  # Proses frame (deteksi robot dan lain-lain)
            frame = cv2.resize(frame, (480, 360))
            cv2.imshow("IP: 10.109.62.20 | Area_1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def process_frame(frame):
    results = model(frame, verbose=False)
    current_robots = set()
    current_time = time.time()

    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].cpu().numpy()
        cls = result.cls[0].cpu().numpy()

        cls_id = int(cls)

        if conf > 0.7 and cls_id == 0:
            corner_points = get_corner_points(x1, y1, x2, y2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
            for i, point in enumerate(corner_points):
                cv2.circle(frame, point, 5, colors[i], -1)
                cv2.putText(frame, str(i), (point[0]+5, point[1]+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            
            robot_id = f"Robot_{int(cls)}"
            current_robots.add(robot_id)
            
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            current_area = check_area(corner_points)
            
            if current_area:
                # Track robot area
                if robot_id not in robot_areas:
                    robot_areas[robot_id] = current_area
                    print_robot_entry(robot_id, current_area, corner_points)
                elif robot_areas[robot_id] != current_area:
                    robot_areas[robot_id] = current_area
                    print_robot_entry(robot_id, current_area, corner_points)
                
                # Display "Robot Telah Sampai" for robots in Area_0
                if current_area == "Area_0":
                    # Update tracking
                    if robot_id not in robots_at_area_0:
                        robots_at_area_0[robot_id] = datetime.now()
                        area_0_message_displayed[robot_id] = False
                        
                    # Display message if not done yet or periodically
                    if not area_0_message_displayed.get(robot_id, False):
                        # Get the text size
                        message = "Robot Telah Sampai"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.2
                        thickness = 3
                        (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)
                        
                        # Calculate text position to center it above the robot
                        text_x = center_point[0] - (text_width // 2)
                        text_y = center_point[1] - 150  # Position above the robot
                        
                        # Draw background rectangle for better visibility
                        cv2.rectangle(frame, 
                                     (text_x - 10, text_y - text_height - 10),
                                     (text_x + text_width + 10, text_y + 10),
                                     (0, 0, 0), -1)
                        
                        # Draw the message with green color
                        cv2.putText(frame, message, 
                                   (text_x, text_y), 
                                   font, font_scale, (0, 255, 0), thickness)
                        
                        # Mark as displayed
                        area_0_message_displayed[robot_id] = True
                    
                    # Remove from error tracking if in Area_0
                    if robot_id in robots_with_errors:
                        robots_with_errors.remove(robot_id)
                        if robot_id in error_durations:
                            del error_durations[robot_id]
                
                # Initialize error_points tracking for this robot if not exists
                if robot_id not in error_points:
                    error_points[robot_id] = {}
                
                # Check movement for each corner point
                corner_errors = []
                for i, point in enumerate(corner_points):
                    point_area = check_area_for_point(point)
                    if point_area:  # Only track points in monitored areas
                        is_stationary, duration = check_point_movement(robot_id, i, point, current_time, point_area)
                        
                        # Mark this point as error if stationary too long and not in Area_0
                        if is_stationary and duration > sec_stop and current_area != "Area_0":
                            error_points[robot_id][i] = True
                            corner_errors.append(i)
                            
                            # Display error indicators for this point
                            cv2.circle(frame, point, 7, (0, 0, 255), -1)
                            cv2.putText(frame, f"{duration:.1f}s", 
                                       (point[0], point[1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        elif is_stationary:
                            # Show duration for stationary points - green in Area_0, yellow elsewhere
                            color = (0, 255, 0) if current_area == "Area_0" else (0, 255, 255)
                            cv2.putText(frame, f"{duration:.1f}s", 
                                       (point[0], point[1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # If any corner points are in error, mark the whole robot as in error
                # But skip this for Area_0
                if corner_errors and current_area != "Area_0":
                    max_duration = 0
                    for idx in corner_errors:
                        duration = current_time - robot_point_positions[robot_id]["timestamps"].get(idx, current_time)
                        max_duration = max(max_duration, duration)
                    
                    # Get the text size
                    text = "AGV Berhenti!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 3
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Calculate text position to center it
                    text_x = center_point[0] - (text_width // 2)
                    text_y = center_point[1] - (text_height // 2) - 200

                    # Draw the centered text
                    cv2.putText(frame, text, 
                            (text_x, text_y + text_height), # Add text_height because putText uses bottom-left corner
                            font, font_scale, (0, 0, 255), thickness)

                    cv2.rectangle(frame, (int(x1)-5, int(y1)-5), 
                                (int(x2)+5, int(y2)+5), (0, 0, 255), 3)
                    
                    if robot_id not in robots_with_errors:
                        print_robot_error(robot_id, current_area, max_duration, corner_errors)
                
                # Original center-based tracking (keep for backwards compatibility)
                tracking_id = f"{robot_id}_{current_area}"
                is_stationary = True
                
                if robot_id in robot_positions:
                    last_pos = robot_positions[robot_id]
                    distance = calculate_distance(center_point, last_pos)
                    
                    if distance > MOVEMENT_TOLERANCE:
                        is_stationary = False
                        if robot_id in robots_with_errors and not corner_errors:
                            print_robot_moving_again(robot_id, current_area)
                        if tracking_id in stop_detection:
                            stop_detection[tracking_id] = current_time
                
                robot_positions[robot_id] = center_point
                
                if tracking_id not in stop_detection:
                    stop_detection[tracking_id] = current_time
                elif is_stationary and not corner_errors and current_area != "Area_0":  # Skip error detection for Area_0
                    duration = current_time - stop_detection[tracking_id]
                    
                    status_color = (0, 255, 255)
                    if duration > sec_stop:
                        status_color = (0, 0, 255)
                        cv2.putText(frame, "ERROR!", 
                                   (center_point[0]-30, center_point[1]-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(x1)-5, int(y1)-5), 
                                    (int(x2)+5, int(y2)+5), (0, 0, 255), 3)
                        if robot_id not in robots_with_errors:
                            print_robot_error(robot_id, current_area, duration)
                    
                    status_text = f"{duration:.1f}s"
                    if duration > sec_stop:
                        status_text += " ERROR!"
                    
                    cv2.putText(frame, status_text, 
                               (center_point[0], center_point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                elif current_area == "Area_0":
                    # For robots in Area_0, show duration in green but no error
                    duration = current_time - stop_detection[tracking_id]
                    status_text = f"{duration:.1f}s"
                    cv2.putText(frame, status_text, 
                               (center_point[0], center_point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update tracking
    tracking_ids = list(stop_detection.keys())
    for tracking_id in tracking_ids:
        robot_id = tracking_id.split('_')[0]
        if robot_id not in current_robots:
            del stop_detection[tracking_id]
            if robot_id in robot_positions:
                del robot_positions[robot_id]
            if robot_id in robot_point_positions:
                del robot_point_positions[robot_id]
            if robot_id in error_points:
                del error_points[robot_id]
            if robot_id in robots_with_errors:
                robots_with_errors.remove(robot_id)
                if robot_id in error_durations:
                    del error_durations[robot_id]
            # Clean up Area_0 tracking
            if robot_id in robots_at_area_0:
                del robots_at_area_0[robot_id]
            if robot_id in area_0_message_displayed:
                del area_0_message_displayed[robot_id]
            if robot_id in area_0_notifications_sent:
                del area_0_notifications_sent[robot_id]

    # Draw areas and information
    cv2.polylines(frame, [Area_1], isClosed=True, color=(34, 242, 235), thickness=2)
    cv2.polylines(frame, [Area_0], isClosed=True, color=(255, 0, 0), thickness=2)
    
    cv2.putText(frame, "Area_1", (310, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 242, 235), 2)
    cv2.putText(frame, "Home Base", (1600, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Database status
    db_status = "DB Connected (No Station)" if db_pool and db_pool.pool else "DB Disconnected"
    db_color = (0, 255, 0) if db_pool and db_pool.pool else (0, 0, 255)
    cv2.putText(frame, db_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, db_color, 2)
    
    # WhatsApp status
    wa_status = "WhatsApp Enabled" if whatsapp_config['enabled'] else "WhatsApp Disabled"
    wa_color = (0, 255, 0) if whatsapp_config['enabled'] else (0, 0, 255)
    cv2.putText(frame, wa_status, (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, wa_color, 2)
    
    # Status information
    tz = pytz.timezone('Asia/Jakarta')
    now = datetime.now(tz)
    cv2.putText(frame, now.strftime("%d-%m-%Y %H:%M:%S"), (50, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Robot terdeteksi: {len(current_robots)}", (50, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Configuration info
    cv2.putText(frame, f"Toleransi gerakan: {MOVEMENT_TOLERANCE}px", (50, 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Batas waktu berhenti: {sec_stop}s", (50, 170), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display count of robots at home base
    if robots_at_area_0:
        cv2.putText(frame, f"Robot di Home Base: {len(robots_at_area_0)}", (50, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        area_0_list = ", ".join([f"{r}" for r in robots_at_area_0])
        cv2.putText(frame, f"Di Home Base: {area_0_list}", (50, 290), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if robots_with_errors:
        cv2.putText(frame, f"Robot ERROR: {len(robots_with_errors)}", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        error_list = ", ".join([f"{r}:{error_durations.get(r, 0):.1f}s" for r in robots_with_errors])
        cv2.putText(frame, f"Robot ERROR: {error_list}", (50, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame


def record_rtsp_stream(url):
    """Fungsi untuk membuka stream RTSP dan menangkap frame"""
    logger.info(f"Mencoba membuka stream: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error("Tidak dapat membuka stream")
        return None
    logger.info("Stream berhasil dibuka")
    return cap


def create_tables_if_not_exist():
    """Membuat tabel jika belum ada di database"""
    global db_pool
    
    if not db_pool:
        logger.error("Cannot create tables - database pool not initialized")
        return False
    
    conn = db_pool.get_connection()
    if not conn:
        logger.error("Failed to get connection for creating tables")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create robot_movements table if not exists - without station field
        query = """
        CREATE TABLE IF NOT EXISTS robot_movements (
            id INT AUTO_INCREMENT PRIMARY KEY,
            location VARCHAR(255) NOT NULL,
            status VARCHAR(10) NOT NULL DEFAULT 'IN',
            timestamp DATETIME NOT NULL,
            INDEX(timestamp)
        )
        """
        
        cursor.execute(query)
        conn.commit()
        logger.info("Database tables checked/created successfully")
        cursor.close()
        return True
    
    except mysql.connector.Error as err:
        logger.error(f"Error creating tables: {err}")
        return False
    
    finally:
        conn.close()


def test_whatsapp_connection(area_name="Area_0"):
    """Test the WhatsApp API connection and send a test message"""
    if not whatsapp_config['enabled']:
        logger.info("WhatsApp notifications are disabled, skipping connection test")
        return False
    
    try:
        test_message = (
            f"🔄 *AGV INFORMATION* 🔄\n\n"
            # f"The AGV Robot Detection System has been started and will send notifications "
            # f"when robots are detected as stopped for more than {sec_stop} seconds.\n\n"
            f"*Time:* {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
            f"*Area:* {area_name}\n"
            # f"*Status:* Ready to monitor\n\n"
            # f"This is an automated message."
        )
        
        successful_sends = 0
        
        # Send to each receiver in the list
        for receiver in whatsapp_config['receivers']:
            try:
                payload = json.dumps({
                    "device": whatsapp_config['device'],
                    "receiver": receiver,
                    "type": "chat",
                    "message": test_message,
                    "simulate_typing": 1
                })
                
                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {whatsapp_config["token"]}'
                }
                
                response = requests.request(
                    "POST", 
                    whatsapp_config['api_url'], 
                    headers=headers, 
                    data=payload
                )
                
                if response.status_code == 200:
                    logger.info(f"WhatsApp connection test successful to {receiver}")
                    successful_sends += 1
                else:
                    logger.error(f"WhatsApp connection test failed to {receiver}: {response.text}")
                
            except Exception as e:
                logger.error(f"Error testing WhatsApp connection to {receiver}: {str(e)}")
        
        return successful_sends > 0
        
    except Exception as e:
        logger.error(f"Error testing WhatsApp connection: {str(e)}")
        return False


def main():
    logger.info("Starting Robot Detection System")
    
    # Initialize database connection pool
    global db_pool, db_config
    db_pool = MySQLDatabasePool(db_config)
    
    retry_count = 0
    cap = None
    
    while cap is None and retry_count < 3:
        cap = record_rtsp_stream(url)  # Attempt to connect to RTSP stream
        if cap is None:
            retry_count += 1
            logger.warning(f"Attempt {retry_count} failed. Retrying in 5 seconds...")
            time.sleep(5)
    
    if cap is None:
        logger.error("Failed to open stream after several attempts. Program will stop.")
        return

    # Create threads for receiving and displaying frames
    p1 = threading.Thread(target=Receive)  # Thread for receiving frames from the RTSP stream
    p2 = threading.Thread(target=Display)  # Thread for displaying frames and processing robot detection
    
    p1.start()  # Start the frame receiving thread
    p2.start()  # Start the frame displaying and processing thread
    
    # Frame processing loop (optional additional work here if needed)
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # You can add custom controls here like adjusting time thresholds or toggling features
            # This loop is for handling additional functionalities (e.g., key presses)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Quit program
                logger.info("Program stopped by user")
                break
            elif key == ord('d'):  # Decrease movement tolerance
                global MOVEMENT_TOLERANCE
                MOVEMENT_TOLERANCE = max(5, MOVEMENT_TOLERANCE - 5)
                logger.info(f"Movement tolerance decreased to {MOVEMENT_TOLERANCE} pixels")
            elif key == ord('u'):  # Increase movement tolerance
                MOVEMENT_TOLERANCE = min(100, MOVEMENT_TOLERANCE + 5)
                logger.info(f"Movement tolerance increased to {MOVEMENT_TOLERANCE} pixels")
            elif key == ord('t'):  # Decrease stop time tolerance
                global sec_stop
                sec_stop = max(1, sec_stop - 1)
                logger.info(f"Stop time tolerance decreased to {sec_stop} seconds")
            elif key == ord('y'):  # Increase stop time tolerance
                sec_stop = min(20, sec_stop + 1)
                logger.info(f"Stop time tolerance increased to {sec_stop} seconds")
            elif key == ord('r'):  # Force reconnect to the database
                logger.info("Attempting to reconnect to the database...")
                if db_pool:
                    db_pool.close()
                db_pool = MySQLDatabasePool(db_config)
                create_tables_if_not_exist()
            elif key == ord('w'):  # Toggle WhatsApp notifications
                whatsapp_config['enabled'] = not whatsapp_config['enabled']
                status = "enabled" if whatsapp_config['enabled'] else "disabled"
                logger.info(f"WhatsApp notifications {status}")
                if whatsapp_config['enabled']:
                    test_whatsapp_connection()
            elif key == ord('s'):  # Save a screenshot of the current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_name = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_name, p1.frame)
                logger.info(f"Screenshot saved as {screenshot_name}")
            elif key == ord('h'):  # Test Home Base arrival notification
                test_robot_id = "Test_Robot"
                logger.info(f"Testing Home Base notification for {test_robot_id}")
                robots_at_area_0[test_robot_id] = datetime.now()
                area_0_message_displayed[test_robot_id] = False
                if send_area_0_notification(test_robot_id):
                    area_0_notifications_sent[test_robot_id] = True
                    logger.info("Home Base test notification sent successfully")
                else:
                    logger.warning("Failed to send Home Base test notification")

    except KeyboardInterrupt:
        logger.info("Program dihentikan oleh pengguna dengan keyboard interrupt")
    except Exception as e:
        logger.error(f"Error tidak terduga: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if cap is not None:
            cap.release()
        if db_pool is not None:
            db_pool.close()
            logger.info("Database connection pool ditutup")
        cv2.destroyAllWindows()
        logger.info("Program selesai")


if __name__ == "__main__":
    main()
