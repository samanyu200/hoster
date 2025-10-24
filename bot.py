
import discord
from discord.ext import commands
from discord import ui, app_commands
import os
import random
import string
import json
import subprocess
from dotenv import load_dotenv
import asyncio
import datetime
from datetime import timedelta
import docker
import time
import logging
import traceback
import aiohttp
import socket
import re
import psutil
import platform
import shutil
from typing import Optional, Literal
import sqlite3
import pickle
import base64
import threading
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import docker
import paramiko
import os
from dotenv import load_dotenv
import stripe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('highcore_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HighCoreBot')

# Load environment variables
load_dotenv()

# Bot configuration
TOKEN = os.getenv('DISCORD_TOKEN')
ADMIN_IDS = {int(id_) for id_ in os.getenv('ADMIN_IDS', '1210291131301101618').split(',') if id_.strip()}
ADMIN_ROLE_ID = int(os.getenv('ADMIN_ROLE_ID', '1376177459870961694'))
WATERMARK = "HighCore VPS Service"
WELCOME_MESSAGE = "Welcome To HighCore! Get Started With Us!"
MAX_VPS_PER_USER = int(os.getenv('MAX_VPS_PER_USER', '3'))
DEFAULT_OS_IMAGE = os.getenv('DEFAULT_OS_IMAGE', 'ubuntu:22.04')
DOCKER_NETWORK = os.getenv('DOCKER_NETWORK', 'bridge')
MAX_CONTAINERS = int(os.getenv('MAX_CONTAINERS', '100'))
DB_FILE = 'highcore.db'
BACKUP_FILE = 'highcore_backup.pkl'
STRIPE_API_KEY = os.getenv('STRIPE_API_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
PRICE_PER_CREDIT = float(os.getenv('PRICE_PER_CREDIT', '0.01'))
SUCCESS_URL = os.getenv('SUCCESS_URL', 'https://your-site/success')
CANCEL_URL = os.getenv('CANCEL_URL', 'https://your-site/cancel')

stripe.api_key = STRIPE_API_KEY

# Known miner process names/patterns
MINER_PATTERNS = [
    'xmrig', 'ethminer', 'cgminer', 'sgminer', 'bfgminer',
    'minerd', 'cpuminer', 'cryptonight', 'stratum', 'pool'
]

# Dockerfile template for custom images
DOCKERFILE_TEMPLATE = """
FROM {base_image}

# Prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install systemd, sudo, SSH, Docker and other essential packages
RUN apt-get update && \\
    apt-get install -y systemd systemd-sysv dbus sudo \\
                       curl gnupg2 apt-transport-https ca-certificates \\
                       software-properties-common \\
                       docker.io openssh-server tmate && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Root password
RUN echo "root:{root_password}" | chpasswd

# Create user and set password
RUN useradd -m -s /bin/bash {username} && \\
    echo "{username}:{user_password}" | chpasswd && \\
    usermod -aG sudo {username}

# Enable SSH login
RUN mkdir /var/run/sshd && \\
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \\
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Enable services on boot
RUN systemctl enable ssh && \\
    systemctl enable docker

# HighCore customization
RUN echo '{welcome_message}' > /etc/motd && \\
    echo 'echo "{welcome_message}"' >> /home/{username}/.bashrc && \\
    echo '{watermark}' > /etc/machine-info && \\
    echo 'highcore-{vps_id}' > /etc/hostname

# Configure neofetch to show container specs
RUN echo '#!/bin/bash\n\
echo "Memory: {memory} GB"\n\
echo "CPU: {cpu} cores"\n\
echo "Disk: {disk} GB"\n\
echo "OS: {os_image}"\n\
echo "Hostname: highcore-{vps_id}"\n\
echo "{watermark}"' > /usr/bin/neofetch && \\
    chmod +x /usr/bin/neofetch

# Install additional useful packages
RUN apt-get update && \\
    apt-get install -y htop nano vim wget git tmux net-tools dnsutils iputils-ping && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/*

# Fix systemd inside container
STOPSIGNAL SIGRTMIN+3

# Boot into systemd (like a VM)
CMD ["/sbin/init"]
"""

class Database:
    """Handles all data persistence using SQLite3"""
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._initialize_settings()

    def _create_tables(self):
        """Create necessary tables"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vps_instances (
                token TEXT PRIMARY KEY,
                vps_id TEXT UNIQUE,
                container_id TEXT,
                memory INTEGER,
                cpu INTEGER,
                disk INTEGER,
                username TEXT,
                password TEXT,
                root_password TEXT,
                created_by TEXT,
                created_at TEXT,
                tmate_session TEXT,
                watermark TEXT,
                os_image TEXT,
                restart_count INTEGER DEFAULT 0,
                last_restart TEXT,
                status TEXT DEFAULT 'running',
                use_custom_image BOOLEAN DEFAULT 1,
                expiration_date TEXT,
                renewal_cost INTEGER
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                key TEXT PRIMARY KEY,
                value INTEGER DEFAULT 0
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS banned_users (
                user_id TEXT PRIMARY KEY
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_users (
                user_id TEXT PRIMARY KEY
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_credits (
                user_id TEXT PRIMARY KEY,
                credits INTEGER DEFAULT 0
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vps_plans (
                name TEXT PRIMARY KEY,
                memory INTEGER,
                cpu INTEGER,
                disk INTEGER,
                credits INTEGER
            )
        ''')
        
        self.conn.commit()

    def _initialize_settings(self):
        """Initialize default settings"""
        defaults = {
            'max_containers': str(MAX_CONTAINERS),
            'max_vps_per_user': str(MAX_VPS_PER_USER)
        }
        for key, value in defaults.items():
            self.cursor.execute('INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)', (key, value))
        
        # Load admin users from database
        self.cursor.execute('SELECT user_id FROM admin_users')
        for row in self.cursor.fetchall():
            ADMIN_IDS.add(int(row[0]))
        
        # Initialize default plans if not exist
        default_plans = {
            "starter": {"memory": 1, "cpu": 1, "disk": 10, "credits": 100},
            "basic": {"memory": 2, "cpu": 2, "disk": 20, "credits": 200},
            "standard": {"memory": 4, "cpu": 4, "disk": 50, "credits": 400},
            "premium": {"memory": 8, "cpu": 8, "disk": 100, "credits": 800},
        }
        for name, plan in default_plans.items():
            self.cursor.execute('''
                INSERT OR IGNORE INTO vps_plans (name, memory, cpu, disk, credits)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, plan['memory'], plan['cpu'], plan['disk'], plan['credits']))
        
        self.conn.commit()

    def get_setting(self, key, default=None):
        self.cursor.execute('SELECT value FROM system_settings WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return int(result[0]) if result else default

    def set_setting(self, key, value):
        self.cursor.execute('INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)', (key, str(value)))
        self.conn.commit()

    def get_stat(self, key, default=0):
        self.cursor.execute('SELECT value FROM usage_stats WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return result[0] if result else default

    def increment_stat(self, key, amount=1):
        current = self.get_stat(key)
        self.cursor.execute('INSERT OR REPLACE INTO usage_stats (key, value) VALUES (?, ?)', (key, current + amount))
        self.conn.commit()

    def get_vps_by_id(self, vps_id):
        self.cursor.execute('SELECT * FROM vps_instances WHERE vps_id = ?', (vps_id,))
        row = self.cursor.fetchone()
        if not row:
            return None, None
        columns = [desc[0] for desc in self.cursor.description]
        vps = dict(zip(columns, row))
        return vps['token'], vps

    def get_vps_by_token(self, token):
        self.cursor.execute('SELECT * FROM vps_instances WHERE token = ?', (token,))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def get_user_vps_count(self, user_id):
        self.cursor.execute('SELECT COUNT(*) FROM vps_instances WHERE created_by = ?', (str(user_id),))
        return self.cursor.fetchone()[0]

    def get_user_vps(self, user_id):
        self.cursor.execute('SELECT * FROM vps_instances WHERE created_by = ?', (str(user_id),))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def get_all_vps(self):
        self.cursor.execute('SELECT * FROM vps_instances')
        columns = [desc[0] for desc in self.cursor.description]
        return {row[0]: dict(zip(columns, row)) for row in self.cursor.fetchall()}

    def add_vps(self, vps_data):
        columns = ', '.join(vps_data.keys())
        placeholders = ', '.join('?' for _ in vps_data)
        self.cursor.execute(f'INSERT INTO vps_instances ({columns}) VALUES ({placeholders})', tuple(vps_data.values()))
        self.conn.commit()
        self.increment_stat('total_vps_created')

    def remove_vps(self, token):
        self.cursor.execute('DELETE FROM vps_instances WHERE token = ?', (token,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def update_vps(self, token, updates):
        set_clause = ', '.join(f'{k} = ?' for k in updates)
        values = list(updates.values()) + [token]
        self.cursor.execute(f'UPDATE vps_instances SET {set_clause} WHERE token = ?', values)
        self.conn.commit()
        return self.cursor.rowcount > 0

    def is_user_banned(self, user_id):
        self.cursor.execute('SELECT 1 FROM banned_users WHERE user_id = ?', (str(user_id),))
        return self.cursor.fetchone() is not None

    def ban_user(self, user_id):
        self.cursor.execute('INSERT OR IGNORE INTO banned_users (user_id) VALUES (?)', (str(user_id),))
        self.conn.commit()

    def unban_user(self, user_id):
        self.cursor.execute('DELETE FROM banned_users WHERE user_id = ?', (str(user_id),))
        self.conn.commit()

    def get_banned_users(self):
        self.cursor.execute('SELECT user_id FROM banned_users')
        return [row[0] for row in self.cursor.fetchall()]

    def add_admin(self, user_id):
        self.cursor.execute('INSERT OR IGNORE INTO admin_users (user_id) VALUES (?)', (str(user_id),))
        self.conn.commit()
        ADMIN_IDS.add(int(user_id))

    def remove_admin(self, user_id):
        self.cursor.execute('DELETE FROM admin_users WHERE user_id = ?', (str(user_id),))
        self.conn.commit()
        if int(user_id) in ADMIN_IDS:
            ADMIN_IDS.remove(int(user_id))

    def get_admins(self):
        self.cursor.execute('SELECT user_id FROM admin_users')
        return [row[0] for row in self.cursor.fetchall()]

    def get_credits(self, user_id):
        self.cursor.execute('SELECT credits FROM user_credits WHERE user_id = ?', (str(user_id),))
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def add_credits(self, user_id, amount):
        current = self.get_credits(user_id)
        self.cursor.execute('INSERT OR REPLACE INTO user_credits (user_id, credits) VALUES (?, ?)', (str(user_id), current + amount))
        self.conn.commit()

    def subtract_credits(self, user_id, amount):
        current = self.get_credits(user_id)
        if current < amount:
            return False
        self.cursor.execute('UPDATE user_credits SET credits = ? WHERE user_id = ?', (current - amount, str(user_id)))
        self.conn.commit()
        return True

    def add_plan(self, name, memory, cpu, disk, credits):
        self.cursor.execute('''
            INSERT OR REPLACE INTO vps_plans (name, memory, cpu, disk, credits)
            VALUES (?, ?, ?, ?, ?)
        ''', (name.lower(), memory, cpu, disk, credits))
        self.conn.commit()

    def remove_plan(self, name):
        self.cursor.execute('DELETE FROM vps_plans WHERE name = ?', (name.lower(),))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def get_plans(self):
        self.cursor.execute('SELECT * FROM vps_plans')
        columns = [desc[0] for desc in self.cursor.description]
        return {row[0]: dict(zip(columns, row)) for row in self.cursor.fetchall()}

    def get_plan(self, name):
        self.cursor.execute('SELECT * FROM vps_plans WHERE name = ?', (name.lower(),))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def backup_data(self):
        """Backup all data to a file"""
        data = {
            'vps_instances': self.get_all_vps(),
            'usage_stats': {},
            'system_settings': {},
            'banned_users': self.get_banned_users(),
            'admin_users': self.get_admins(),
            'user_credits': {},
            'vps_plans': self.get_plans()
        }
        
        # Get usage stats
        self.cursor.execute('SELECT * FROM usage_stats')
        for row in self.cursor.fetchall():
            data['usage_stats'][row[0]] = row[1]
            
        # Get system settings
        self.cursor.execute('SELECT * FROM system_settings')
        for row in self.cursor.fetchall():
            data['system_settings'][row[0]] = row[1]
            
        # Get user credits
        self.cursor.execute('SELECT * FROM user_credits')
        for row in self.cursor.fetchall():
            data['user_credits'][row[0]] = row[1]
            
        with open(BACKUP_FILE, 'wb') as f:
            pickle.dump(data, f)
            
        return True

    def restore_data(self):
        """Restore data from backup file"""
        if not os.path.exists(BACKUP_FILE):
            return False
            
        try:
            with open(BACKUP_FILE, 'rb') as f:
                data = pickle.load(f)
                
            # Clear all tables
            self.cursor.execute('DELETE FROM vps_instances')
            self.cursor.execute('DELETE FROM usage_stats')
            self.cursor.execute('DELETE FROM system_settings')
            self.cursor.execute('DELETE FROM banned_users')
            self.cursor.execute('DELETE FROM admin_users')
            self.cursor.execute('DELETE FROM user_credits')
            self.cursor.execute('DELETE FROM vps_plans')
            
            # Restore VPS instances
            for token, vps in data['vps_instances'].items():
                columns = ', '.join(vps.keys())
                placeholders = ', '.join('?' for _ in vps)
                self.cursor.execute(f'INSERT INTO vps_instances ({columns}) VALUES ({placeholders})', tuple(vps.values()))
            
            # Restore usage stats
            for key, value in data['usage_stats'].items():
                self.cursor.execute('INSERT INTO usage_stats (key, value) VALUES (?, ?)', (key, value))
                
            # Restore system settings
            for key, value in data['system_settings'].items():
                self.cursor.execute('INSERT INTO system_settings (key, value) VALUES (?, ?)', (key, value))
                
            # Restore banned users
            for user_id in data['banned_users']:
                self.cursor.execute('INSERT INTO banned_users (user_id) VALUES (?)', (user_id,))
                
            # Restore admin users
            for user_id in data['admin_users']:
                self.cursor.execute('INSERT INTO admin_users (user_id) VALUES (?)', (user_id,))
                ADMIN_IDS.add(int(user_id))
                
            # Restore user credits
            for user_id, credits in data['user_credits'].items():
                self.cursor.execute('INSERT INTO user_credits (user_id, credits) VALUES (?, ?)', (user_id, credits))
                
            # Restore VPS plans
            for name, plan in data['vps_plans'].items():
                self.cursor.execute('''
                    INSERT INTO vps_plans (name, memory, cpu, disk, credits)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, plan['memory'], plan['cpu'], plan['disk'], plan['credits']))
                
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error restoring data: {e}")
            return False

    def close(self):
        self.conn.close()

# Initialize bot with command prefix '!'
class HighCoreBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = Database(DB_FILE)
        self.session = None
        self.docker_client = None
        self.system_stats = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_io': (0, 0),
            'last_updated': 0
        }
        self.my_persistent_views = {}

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
            self.loop.create_task(self.update_system_stats())
            self.loop.create_task(self.anti_miner_monitor())
            self.loop.create_task(self.check_expirations())
            # Reconnect to existing containers
            await self.reconnect_containers()
            # Restore persistent views
            await self.restore_persistent_views()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
        # Start Flask for webhook
        self.flask_thread = threading.Thread(target=self.run_flask)
        self.flask_thread.start()

    def run_flask(self):
        app = Flask(__name__)

        @app.route('/webhook', methods=['POST'])
        def webhook():
            payload = request.get_data()
            sig_header = request.headers.get('Stripe-Signature')
            try:
                event = stripe.Webhook.construct_event(
                    payload, sig_header, STRIPE_WEBHOOK_SECRET
                )
            except ValueError as e:
                return 'Invalid payload', 400
            except stripe.error.SignatureVerificationError as e:
                return 'Signature failed', 400

            if event['type'] == 'checkout.session.completed':
                session = event['data']['object']
                user_id = session.metadata.get('user_id')
                credits = int(session.metadata.get('credits'))
                if user_id and credits:
                    self.db.add_credits(user_id, credits)
                    logger.info(f"Added {credits} credits to user {user_id}")

            return '', 200

        app.run(host='0.0.0.0', port=8080)

    async def check_expirations(self):
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                current_time = datetime.datetime.now().isoformat()
                for token, vps in list(self.db.get_all_vps().items()):
                    expiration = vps.get('expiration_date')
                    if expiration and expiration < current_time and vps['status'] != 'expired':
                        try:
                            container = self.docker_client.containers.get(vps['container_id'])
                            container.stop()
                        except Exception as e:
                            logger.error(f"Error stopping expired VPS {vps['vps_id']}: {e}")
                        self.db.update_vps(token, {'status': 'expired'})
                        logger.info(f"Expired VPS {vps['vps_id']}")
            except Exception as e:
                logger.error(f"Error in check_expirations: {e}")
            await asyncio.sleep(3600)  # Check every hour

    async def reconnect_containers(self):
        """Reconnect to existing containers on startup"""
        if not self.docker_client:
            return
            
        for token, vps in list(self.db.get_all_vps().items()):
            if vps['status'] == 'running':
                try:
                    container = self.docker_client.containers.get(vps['container_id'])
                    if container.status != 'running':
                        container.start()
                    logger.info(f"Reconnected and started container for VPS {vps['vps_id']}")
                except docker.errors.NotFound:
                    logger.warning(f"Container {vps['container_id']} not found, removing from data")
                    self.db.remove_vps(token)
                except Exception as e:
                    logger.error(f"Error reconnecting container {vps['vps_id']}: {e}")

    async def restore_persistent_views(self):
        """Restore persistent views after restart"""
        # This would be implemented to restore any persistent UI components
        pass

    async def anti_miner_monitor(self):
        """Periodically check for mining activities"""
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                for token, vps in self.db.get_all_vps().items():
                    if vps['status'] != 'running':
                        continue
                    try:
                        container = self.docker_client.containers.get(vps['container_id'])
                        if container.status != 'running':
                            continue
                        
                        # Check processes
                        exec_result = container.exec_run("ps aux")
                        output = exec_result.output.decode().lower()
                        
                        for pattern in MINER_PATTERNS:
                            if pattern in output:
                                logger.warning(f"Mining detected in VPS {vps['vps_id']}, suspending...")
                                container.stop()
                                self.db.update_vps(token, {'status': 'suspended'})
                                # Notify owner
                                try:
                                    owner = await self.fetch_user(int(vps['created_by']))
                                    await owner.send(f"⚠️ Your VPS {vps['vps_id']} has been suspended due to detected mining activity. Contact admin to unsuspend.")
                                except:
                                    pass
                                break
                    except Exception as e:
                        logger.error(f"Error checking VPS {vps['vps_id']} for mining: {e}")
            except Exception as e:
                logger.error(f"Error in anti_miner_monitor: {e}")
            await asyncio.sleep(300)  # Check every 5 minutes

    async def update_system_stats(self):
        """Update system statistics periodically"""
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                mem = psutil.virtual_memory()
                
                # Disk usage
                disk = psutil.disk_usage('/')
                
                # Network IO
                net_io = psutil.net_io_counters()
                
                self.system_stats = {
                    'cpu_usage': cpu_percent,
                    'memory_usage': mem.percent,
                    'memory_used': mem.used / (1024 ** 3),  # GB
                    'memory_total': mem.total / (1024 ** 3),  # GB
                    'disk_usage': disk.percent,
                    'disk_used': disk.used / (1024 ** 3),  # GB
                    'disk_total': disk.total / (1024 ** 3),  # GB
                    'network_sent': net_io.bytes_sent / (1024 ** 2),  # MB
                    'network_recv': net_io.bytes_recv / (1024 ** 2),  # MB
                    'last_updated': time.time()
                }
            except Exception as e:
                logger.error(f"Error updating system stats: {e}")
            await asyncio.sleep(30)

    async def close(self):
        await super().close()
        if self.session:
            await self.session.close()
        if self.docker_client:
            self.docker_client.close()
        self.db.close()

def generate_token():
    """Generate a random token for VPS access"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=24))

def generate_vps_id():
    """Generate a unique VPS ID"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def generate_ssh_password():
    """Generate a random SSH password"""
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choices(chars, k=16))

def has_admin_role(ctx):
    """Check if user has admin role or is in ADMIN_IDS"""
    if isinstance(ctx, discord.Interaction):
        user_id = ctx.user.id
        roles = ctx.user.roles
    else:
        user_id = ctx.author.id
        roles = ctx.author.roles
    
    if user_id in ADMIN_IDS:
        return True
    
    return any(role.id == ADMIN_ROLE_ID for role in roles)

async def capture_ssh_session_line(process):
    """Capture the SSH session line from tmate output"""
    try:
        while True:
            output = await process.stdout.readline()
            if not output:
                break
            output = output.decode('utf-8').strip()
            if "ssh session:" in output:
                return output.split("ssh session:")[1].strip()
        return None
    except Exception as e:
        logger.error(f"Error capturing SSH session: {e}")
        return None

async def run_docker_command(container_id, command, timeout=120):
    """Run a Docker command asynchronously with timeout"""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id, *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            if process.returncode != 0:
                raise Exception(f"Command failed: {stderr.decode()}")
            return True, stdout.decode()
        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"Command timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"Error running Docker command: {e}")
        return False, str(e)

async def kill_apt_processes(container_id):
    """Kill any running apt processes"""
    try:
        success, _ = await run_docker_command(container_id, ["bash", "-c", "killall apt apt-get dpkg || true"])
        await asyncio.sleep(2)
        success, _ = await run_docker_command(container_id, ["bash", "-c", "rm -f /var/lib/apt/lists/lock /var/cache/apt/archives/lock /var/lib/dpkg/lock*"])
        await asyncio.sleep(2)
        return success
    except Exception as e:
        logger.error(f"Error killing apt processes: {e}")
        return False

async def wait_for_apt_lock(container_id, status_msg):
    """Wait for apt lock to be released"""
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            await kill_apt_processes(container_id)
            
            process = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id, "bash", "-c", "lsof /var/lib/dpkg/lock-frontend",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return True
                
            if isinstance(status_msg, discord.Interaction):
                await status_msg.followup.send(f"🔄 Waiting for package manager to be ready... (Attempt {attempt + 1}/{max_attempts})", ephemeral=True)
            else:
                await status_msg.edit(content=f"🔄 Waiting for package manager to be ready... (Attempt {attempt + 1}/{max_attempts})")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error checking apt lock: {e}")
            await asyncio.sleep(5)
    
    return False

async def build_custom_image(vps_id, username, root_password, user_password, base_image=DEFAULT_OS_IMAGE):
    """Build a custom Docker image using our template"""
    try:
        # Create a temporary directory for the Dockerfile
        temp_dir = f"temp_dockerfiles/{vps_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate Dockerfile content
        dockerfile_content = DOCKERFILE_TEMPLATE.format(
            base_image=base_image,
            root_password=root_password,
            username=username,
            user_password=user_password,
            welcome_message=WELCOME_MESSAGE,
            watermark=WATERMARK,
            vps_id=vps_id,
            memory=vps_data['memory'],
            cpu=vps_data['cpu'],
            disk=vps_data['disk'],
            os_image=vps_data['os_image']
        )
        
        # Write Dockerfile
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Build the image
        image_tag = f"highcore/{vps_id.lower()}:latest"
        build_process = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", image_tag, temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await build_process.communicate()
        
        if build_process.returncode != 0:
            raise Exception(f"Failed to build image: {stderr.decode()}")
        
        return image_tag
    except Exception as e:
        logger.error(f"Error building custom image: {e}")
        raise
    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")

async def setup_container(container_id, status_msg, memory, username, vps_id=None, use_custom_image=False):
    """Enhanced container setup with HighCore customization"""
    try:
        # Ensure container is running
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("🔍 Checking container status...", ephemeral=True)
        else:
            await status_msg.edit(content="🔍 Checking container status...")
            
        container = bot.docker_client.containers.get(container_id)
        if container.status != "running":
            if isinstance(status_msg, discord.Interaction):
                await status_msg.followup.send("🚀 Starting container...", ephemeral=True)
            else:
                await status_msg.edit(content="🚀 Starting container...")
            container.start()
            await asyncio.sleep(5)

        # Generate SSH password
        ssh_password = generate_ssh_password()
        
        # Install tmate and other required packages
        if not use_custom_image:
            if isinstance(status_msg, discord.Interaction):
                await status_msg.followup.send("📦 Installing required packages...", ephemeral=True)
            else:
                await status_msg.edit(content="📦 Installing required packages...")
                
            # Update package list
            success, output = await run_docker_command(container_id, ["apt-get", "update"])
            if not success:
                raise Exception(f"Failed to update package list: {output}")

            # Install packages
            packages = [
                "tmate", "screen", "wget", "curl", "htop", "nano", "vim", 
                "openssh-server", "sudo", "ufw", "git", "docker.io", "systemd", "systemd-sysv"
            ]
            success, output = await run_docker_command(container_id, ["apt-get", "install", "-y"] + packages)
            if not success:
                raise Exception(f"Failed to install packages: {output}")

        # Setup SSH
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("🔐 Configuring SSH access...", ephemeral=True)
        else:
            await status_msg.edit(content="🔐 Configuring SSH access...")
            
        # Create user and set password (if not using custom image)
        if not use_custom_image:
            user_setup_commands = [
                f"useradd -m -s /bin/bash {username}",
                f"echo '{username}:{ssh_password}' | chpasswd",
                f"usermod -aG sudo {username}",
                "sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config",
                "sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config",
                "service ssh restart"
            ]
            
            for cmd in user_setup_commands:
                success, output = await run_docker_command(container_id, ["bash", "-c", cmd])
                if not success:
                    raise Exception(f"Failed to setup user: {output}")

        # Set HighCore customization
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("🎨 Setting up HighCore customization...", ephemeral=True)
        else:
            await status_msg.edit(content="🎨 Setting up HighCore customization...")
            
        # Create welcome message file
        welcome_cmd = f"echo '{WELCOME_MESSAGE}' > /etc/motd && echo 'echo \"{WELCOME_MESSAGE}\"' >> /home/{username}/.bashrc"
        success, output = await run_docker_command(container_id, ["bash", "-c", welcome_cmd])
        if not success:
            logger.warning(f"Could not set welcome message: {output}")

        # Set hostname and watermark
        if not vps_id:
            vps_id = generate_vps_id()
        hostname_cmd = f"echo 'highcore-{vps_id}' > /etc/hostname && hostname highcore-{vps_id}"
        success, output = await run_docker_command(container_id, ["bash", "-c", hostname_cmd])
        if not success:
            raise Exception(f"Failed to set hostname: {output}")

        # Set memory limit in cgroup
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("⚙️ Setting resource limits...", ephemeral=True)
        else:
            await status_msg.edit(content="⚙️ Setting resource limits...")
            
        memory_bytes = memory * 1024 * 1024 * 1024
        success, output = await run_docker_command(container_id, ["bash", "-c", f"echo {memory_bytes} > /sys/fs/cgroup/memory/memory.limit_in_bytes"])
        if not success:
            logger.warning(f"Could not set memory limit in cgroup: {output}")

        success, output = await run_docker_command(container_id, ["bash", "-c", f"echo {memory_bytes} > /sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"])
        if not success:
            logger.warning(f"Could not set swap memory limit in cgroup: {output}")

        # Set watermark in machine info
        success, output = await run_docker_command(container_id, ["bash", "-c", f"echo '{WATERMARK}' > /etc/machine-info"])
        if not success:
            logger.warning(f"Could not set machine info: {output}")

        # Basic security setup
        security_commands = [
            "ufw allow ssh",
            "ufw --force enable",
            "apt-get -y autoremove",
            "apt-get clean",
            f"chown -R {username}:{username} /home/{username}",
            f"chmod 700 /home/{username}"
        ]
        
        for cmd in security_commands:
            success, output = await run_docker_command(container_id, ["bash", "-c", cmd])
            if not success:
                logger.warning(f"Security setup command failed: {cmd} - {output}")

        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send("✅ HighCore VPS setup completed successfully!", ephemeral=True)
        else:
            await status_msg.edit(content="✅ HighCore VPS setup completed successfully!")
            
        return True, ssh_password, vps_id
    except Exception as e:
        error_msg = f"Setup failed: {str(e)}"
        logger.error(error_msg)
        if isinstance(status_msg, discord.Interaction):
            await status_msg.followup.send(f"❌ {error_msg}", ephemeral=True)
        else:
            await status_msg.edit(content=f"❌ {error_msg}")
        return False, None, None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = HighCoreBot(command_prefix='!', intents=intents, help_command=None)

@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    
    # Auto-start VPS containers based on status
    if bot.docker_client:
        for token, vps in bot.db.get_all_vps().items():
            if vps['status'] == 'running':
                try:
                    container = bot.docker_client.containers.get(vps["container_id"])
                    if container.status != "running":
                        container.start()
                        logger.info(f"Started container for VPS {vps['vps_id']}")
                except docker.errors.NotFound:
                    logger.warning(f"Container {vps['container_id']} not found")
                except Exception as e:
                    logger.error(f"Error starting container: {e}")
    
    try:
        await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="HighCore VPS"))
        synced_commands = await bot.tree.sync()
        logger.info(f"Synced {len(synced_commands)} slash commands")
    except Exception as e:
        logger.error(f"Error syncing slash commands: {e}")

@bot.hybrid_command(name='help', description='Show all available commands')
async def show_commands(ctx):
    """Show all available commands"""
    try:
        embed = discord.Embed(title="🤖 HighCore VPS Bot Commands", color=discord.Color.blue())
        
        # User commands
        embed.add_field(name="User Commands", value="""
`/create_vps` - Create a new VPS (Admin only)
`/connect_vps <token>` - Connect to your VPS
`/list` - List all your VPS instances
`/help` - Show this help message
`/manage_vps <vps_id>` - Manage your VPS
`/transfer_vps <vps_id> <user>` - Transfer VPS ownership
`/vps_stats <vps_id>` - Show VPS resource usage
`/change_ssh_password <vps_id>` - Change SSH password
`/vps_shell <vps_id>` - Get shell access to your VPS
`/vps_console <vps_id>` - Get direct console access to your VPS
`/vps_usage` - Show your VPS usage statistics
!plans - Show all VPS plans
!buy <plan_name> - Buy a VPS plan
!bal - Check your credit balance
""", inline=False)
        
        # Admin commands
        if has_admin_role(ctx):
            embed.add_field(name="Admin Commards", value="""
`/vps_list` - List all VPS instances
`/delete_vps <vps_id>` - Delete a VPS
`/admin_stats` - Show system statistics
`/cleanup_vps` - Cleanup inactive VPS instances
`/add_admin <user>` - Add a new admin
`/remove_admin <user>` - Remove an admin (Owner only)
`/list_admins` - List all admin users
`/system_info` - Show detailed system information
`/container_limit <max>` - Set maximum container limit
`/global_stats` - Show global usage statistics
`/migrate_vps <vps_id>` - Migrate VPS to another host
`/emergency_stop <vps_id>` - Force stop a problematic VPS
`/emergency_remove <vps_id>` - Force remove a problematic VPS
`/suspend_vps <vps_id>` - Suspend a VPS
`/unsuspend_vps <vps_id>` - Unsuspend a VPS
`/edit_vps <vps_id> <memory> <cpu> <disk>` - Edit VPS specifications
`/ban_user <user>` - Ban a user from creating VPS
`/unban_user <user>` - Unban a user
`/list_banned` - List banned users
`/backup_data` - Backup all data
`/restore_data` - Restore from backup
`/reinstall_bot` - Reinstall the bot (Owner only)
!givecredit <amount> <@user> - Give credits to a user
!add_plan <name> <memory> <cpu> <disk> <credits> - Add or update a VPS plan
!remove_plan <name> - Remove a VPS plan
""", inline=False)
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in show_commands: {e}")
        await ctx.send("❌ An error occurred while processing your request.")

@bot.hybrid_command(name='add_admin', description='Add a new admin (Admin only)')
@app_commands.describe(
    user="User to make admin"
)
async def add_admin(ctx, user: discord.User):
    """Add a new admin user"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return
    
    bot.db.add_admin(user.id)
    await ctx.send(f"✅ {user.mention} has been added as an admin!", ephemeral=True)

@bot.hybrid_command(name='remove_admin', description='Remove an admin (Owner only)')
@app_commands.describe(
    user="User to remove from admin"
)
async def remove_admin(ctx, user: discord.User):
    """Remove an admin user (Owner only)"""
    if ctx.author.id != 1210291131301101618:  # Only the owner can remove admins
        await ctx.send("❌ Only the owner can remove admins!", ephemeral=True)
        return
    
    bot.db.remove_admin(user.id)
    await ctx.send(f"✅ {user.mention} has been removed from admins!", ephemeral=True)

@bot.hybrid_command(name='list_admins', description='List all admin users')
async def list_admins(ctx):
    """List all admin users"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return
    
    embed = discord.Embed(title="Admin Users", color=discord.Color.blue())
    
    # List user IDs in ADMIN_IDS
    admin_list = []
    for admin_id in ADMIN_IDS:
        try:
            user = await bot.fetch_user(admin_id)
            admin_list.append(f"{user.name} ({user.id})")
        except:
            admin_list.append(f"Unknown User ({admin_id})")
    
    # List users with admin role
    if ctx.guild:
        admin_role = ctx.guild.get_role(ADMIN_ROLE_ID)
        if admin_role:
            role_admins = [f"{member.name} ({member.id})" for member in admin_role.members]
            admin_list.extend(role_admins)
    
    if not admin_list:
        embed.description = "No admins found"
    else:
        embed.description = "\n".join(sorted(set(admin_list)))  # Remove duplicates
    
    await ctx.send(embed=embed, ephemeral=True)

@bot.hybrid_command(name='create_vps', description='Create a new VPS (Admin only)')
@app_commands.describe(
    memory="Memory in GB",
    cpu="CPU cores",
    disk="Disk space in GB",
    owner="User who will own the VPS",
    os_image="OS image to use",
    use_custom_image="Use custom HighCore image (recommended)"
)
async def create_vps_command(ctx, memory: int, cpu: int, disk: int, owner: discord.Member, 
                           os_image: str = DEFAULT_OS_IMAGE, use_custom_image: bool = True):
    """Create a new VPS with specified parameters (Admin only)"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return

    if bot.db.is_user_banned(owner.id):
        await ctx.send("❌ This user is banned from creating VPS!", ephemeral=True)
        return

    if not ctx.guild:
        await ctx.send("❌ This command can only be used in a server!", ephemeral=True)
        return

    if not bot.docker_client:
        await ctx.send("❌ Docker is not available. Please contact the administrator.", ephemeral=True)
        return

    try:
        # Validate inputs
        if memory < 1 or memory > 512:
            await ctx.send("❌ Memory must be between 1GB and 512GB", ephemeral=True)
            return
        if cpu < 1 or cpu > 32:
            await ctx.send("❌ CPU cores must be between 1 and 32", ephemeral=True)
            return
        if disk < 10 or disk > 1000:
            await ctx.send("❌ Disk space must be between 10GB and 1000GB", ephemeral=True)
            return

        # Check if we've reached container limit
        containers = bot.docker_client.containers.list(all=True)
        if len(containers) >= bot.db.get_setting('max_containers', MAX_CONTAINERS):
            await ctx.send(f"❌ Maximum container limit reached ({bot.db.get_setting('max_containers')}). Please delete some VPS instances first.", ephemeral=True)
            return

        # Check if user already has maximum VPS instances
        if bot.db.get_user_vps_count(owner.id) >= bot.db.get_setting('max_vps_per_user', MAX_VPS_PER_USER):
            await ctx.send(f"❌ {owner.mention} already has the maximum number of VPS instances ({bot.db.get_setting('max_vps_per_user')})", ephemeral=True)
            return

        status_msg = await ctx.send("🚀 Creating HighCore VPS instance... This may take a few minutes.")

        memory_bytes = memory * 1024 * 1024 * 1024
        vps_id = generate_vps_id()
        username = owner.name.lower().replace(" ", "_")[:20]
        root_password = generate_ssh_password()
        user_password = generate_ssh_password()
        token = generate_token()

        if use_custom_image:
            await status_msg.edit(content="🔨 Building custom Docker image...")
            try:
                image_tag = await build_custom_image(vps_id, username, root_password, user_password, os_image)
            except Exception as e:
                await status_msg.edit(content=f"❌ Failed to build Docker image: {str(e)}")
                return

            await status_msg.edit(content="⚙️ Initializing container...")
            try:
                container = bot.docker_client.containers.run(
                    image_tag,
                    detach=True,
                    privileged=True,
                    hostname=f"highcore-{vps_id}",
                    mem_limit=memory_bytes,
                    cpu_period=100000,
                    cpu_quota=int(cpu * 100000),
                    cap_add=["ALL"],
                    network=DOCKER_NETWORK,
                    volumes={
                        f'highcore-{vps_id}': {'bind': '/data', 'mode': 'rw'}
                    },
                    restart_policy={"Name": "always"}
                )
            except Exception as e:
                await status_msg.edit(content=f"❌ Failed to start container: {str(e)}")
                return
        else:
            await status_msg.edit(content="⚙️ Initializing container...")
            try:
                container = bot.docker_client.containers.run(
                    os_image,
                    detach=True,
                    privileged=True,
                    hostname=f"highcore-{vps_id}",
                    mem_limit=memory_bytes,
                    cpu_period=100000,
                    cpu_quota=int(cpu * 100000),
                    cap_add=["ALL"],
                    command="tail -f /dev/null",
                    tty=True,
                    network=DOCKER_NETWORK,
                    volumes={
                        f'highcore-{vps_id}': {'bind': '/data', 'mode': 'rw'}
                    },
                    restart_policy={"Name": "always"}
                )
            except docker.errors.ImageNotFound:
                await status_msg.edit(content=f"❌ OS image {os_image} not found. Using default {DEFAULT_OS_IMAGE}")
                container = bot.docker_client.containers.run(
                    DEFAULT_OS_IMAGE,
                    detach=True,
                    privileged=True,
                    hostname=f"highcore-{vps_id}",
                    mem_limit=memory_bytes,
                    cpu_period=100000,
                    cpu_quota=int(cpu * 100000),
                    cap_add=["ALL"],
                    command="tail -f /dev/null",
                    tty=True,
                    network=DOCKER_NETWORK,
                    volumes={
                        f'highcore-{vps_id}': {'bind': '/data', 'mode': 'rw'}
                    },
                    restart_policy={"Name": "always"}
                )
                os_image = DEFAULT_OS_IMAGE

        await status_msg.edit(content="🔧 Container created. Setting up HighCore environment...")
        await asyncio.sleep(5)

        setup_success, ssh_password, _ = await setup_container(
            container.id, 
            status_msg, 
            memory, 
            username, 
            vps_id,
            use_custom_image=use_custom_image
        )
        if not setup_success:
            raise Exception("Failed to setup container")

        await status_msg.edit(content="🔐 Starting SSH session...")

        exec_cmd = await asyncio.create_subprocess_exec(
            "docker", "exec", container.id, "tmate", "-F",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        ssh_session_line = await capture_ssh_session_line(exec_cmd)
        if not ssh_session_line:
            raise Exception("Failed to get tmate session")
        
        vps_data = {
            "token": token,
            "vps_id": vps_id,
            "container_id": container.id,
            "memory": memory,
            "cpu": cpu,
            "disk": disk,
            "username": username,
            "password": ssh_password,
            "root_password": root_password if use_custom_image else None,
            "created_by": str(owner.id),
            "created_at": str(datetime.datetime.now()),
            "tmate_session": ssh_session_line,
            "watermark": WATERMARK,
            "os_image": os_image,
            "restart_count": 0,
            "last_restart": None,
            "status": "running",
            "use_custom_image": use_custom_image,
            "expiration_date": None,  # Permanent for admin-created
            "renewal_cost": None
        }
        
        bot.db.add_vps(vps_data)
        
        try:
            embed = discord.Embed(title="🎉 HighCore VPS Creation Successful", color=discord.Color.green())
            embed.add_field(name="🆔 VPS ID", value=vps_id, inline=True)
            embed.add_field(name="💾 Memory", value=f"{memory}GB", inline=True)
            embed.add_field(name="⚡ CPU", value=f"{cpu} cores", inline=True)
            embed.add_field(name="💿 Disk", value=f"{disk}GB", inline=True)
            embed.add_field(name="👤 Username", value=username, inline=True)
            embed.add_field(name="🔑 User Password", value=f"||{ssh_password}||", inline=False)
            if use_custom_image:
                embed.add_field(name="🔑 Root Password", value=f"||{root_password}||", inline=False)
            embed.add_field(name="🔒 Tmate Session", value=f"```{ssh_session_line}```", inline=False)
            embed.add_field(name="🔌 Direct SSH", value=f"```ssh {username}@<server-ip>```", inline=False)
            embed.add_field(name="ℹ️ Note", value="This is a HighCore VPS instance. You can install and configure additional packages as needed.", inline=False)
            
            await owner.send(embed=embed)
            await status_msg.edit(content=f"✅ HighCore VPS creation successful! VPS has been created for {owner.mention}. Check your DMs for connection details.")
        except discord.Forbidden:
            await status_msg.edit(content=f"❌ I couldn't send a DM to {owner.mention}. Please ask them to enable DMs from server members.")
            
    except Exception as e:
        error_msg = f"❌ An error occurred while creating the VPS: {str(e)}"
        logger.error(error_msg)
        await ctx.send(error_msg)
        if 'container' in locals():
            try:
                container.stop()
                container.remove()
            except Exception as e:
                logger.error(f"Error cleaning up container: {e}")

@bot.hybrid_command(name='list', description='List all your VPS instances')
async def list_vps(ctx):
    """List all VPS instances owned by the user"""
    try:
        user_vps = bot.db.get_user_vps(ctx.author.id)
        
        if not user_vps:
            await ctx.send("You don't have any VPS instances.", ephemeral=True)
            return

        embed = discord.Embed(title="Your HighCore VPS Instances", color=discord.Color.blue())
        
        for vps in user_vps:
            try:
                # Handle missing container ID gracefully
                container = bot.docker_client.containers.get(vps["container_id"]) if vps["container_id"] else None
                status = vps['status'].capitalize() if vps.get('status') else "Unknown"
            except Exception as e:
                status = "Not Found"
                logger.error(f"Error fetching container {vps['container_id']}: {e}")

            # Adding fields safely to prevent missing keys causing errors
            embed.add_field(
                name=f"VPS {vps['vps_id']}",
                value=f"""
Status: {status}
Memory: {vps.get('memory', 'Unknown')}GB
CPU: {vps.get('cpu', 'Unknown')} cores
Disk Allocated: {vps.get('disk', 'Unknown')}GB
Username: {vps.get('username', 'Unknown')}
OS: {vps.get('os_image', DEFAULT_OS_IMAGE)}
Created: {vps.get('created_at', 'Unknown')}
Restarts: {vps.get('restart_count', 0)}
Expiration: {vps.get('expiration_date', 'Permanent')}
""",
                inline=False
            )
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in list_vps: {e}")
        await ctx.send(f"❌ Error listing VPS instances: {str(e)}")

@bot.hybrid_command(name='vps_list', description='List all VPS instances (Admin only)')
async def admin_list_vps(ctx):
    """List all VPS instances (Admin only)"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return

    try:
        all_vps = bot.db.get_all_vps()
        if not all_vps:
            await ctx.send("No VPS instances found.", ephemeral=True)
            return

        embed = discord.Embed(title="All HighCore VPS Instances", color=discord.Color.blue())
        valid_vps_count = 0
        
        for token, vps in all_vps.items():
            try:
                # Fetch username of the owner with error handling
                user = await bot.fetch_user(int(vps.get("created_by", "0")))
                username = user.name if user else "Unknown User"
            except Exception as e:
                username = "Unknown User"
                logger.error(f"Error fetching user {vps.get('created_by')}: {e}")

            try:
                # Handle missing container ID gracefully
                container = bot.docker_client.containers.get(vps.get("container_id", "")) if vps.get("container_id") else None
                container_status = container.status if container else "Not Found"
            except Exception as e:
                container_status = "Not Found"
                logger.error(f"Error fetching container {vps.get('container_id')}: {e}")

            # Get status and other info with error fallback
            status = vps.get('status', "Unknown").capitalize()

            vps_info = f"""
Owner: {username}
Status: {status} (Container: {container_status})
Memory: {vps.get('memory', 'Unknown')}GB
CPU: {vps.get('cpu', 'Unknown')} cores
Disk: {vps.get('disk', 'Unknown')}GB
Username: {vps.get('username', 'Unknown')}
OS: {vps.get('os_image', DEFAULT_OS_IMAGE)}
Created: {vps.get('created_at', 'Unknown')}
Restarts: {vps.get('restart_count', 0)}
Expiration: {vps.get('expiration_date', 'Permanent')}
"""

            embed.add_field(
                name=f"VPS {vps.get('vps_id', 'Unknown')}",
                value=vps_info,
                inline=False
            )
            valid_vps_count += 1

        if valid_vps_count == 0:
            await ctx.send("No valid VPS instances found.", ephemeral=True)
            return

        embed.set_footer(text=f"Total VPS instances: {valid_vps_count}")
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in admin_list_vps: {e}")
        await ctx.send(f"❌ Error listing VPS instances: {str(e)}")

@bot.hybrid_command(name='delete_vps', description='Delete a VPS instance (Admin only)')
@app_commands.describe(
    vps_id="ID of the VPS to delete"
)
async def delete_vps(ctx, vps_id: str):
    """Delete a VPS instance (Admin only)"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return

    try:
        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps:
            await ctx.send("❌ VPS not found!", ephemeral=True)
            return
        
        try:
            container = bot.docker_client.containers.get(vps["container_id"])
            container.stop()
            container.remove()
            logger.info(f"Deleted container {vps['container_id']} for VPS {vps_id}")
        except Exception as e:
            logger.error(f"Error removing container: {e}")
        
        bot.db.remove_vps(token)
        
        await ctx.send(f"✅ HighCore VPS {vps_id} has been deleted successfully!")
    except Exception as e:
        logger.error(f"Error in delete_vps: {e}")
        await ctx.send(f"❌ Error deleting VPS: {str(e)}")

@bot.hybrid_command(name='connect_vps', description='Connect to a VPS using the provided token')
@app_commands.describe(
    token="Access token for the VPS"
)
async def connect_vps(ctx, token: str):
    """Connect to a VPS using the provided token"""
    vps = bot.db.get_vps_by_token(token)
    if not vps:
        await ctx.send("❌ Invalid token!", ephemeral=True)
        return
        
    if str(ctx.author.id) != vps["created_by"] and not has_admin_role(ctx):
        await ctx.send("❌ You don't have permission to access this VPS!", ephemeral=True)
        return

    try:
        try:
            container = bot.docker_client.containers.get(vps["container_id"])
            if container.status != "running":
                container.start()
                await asyncio.sleep(5)
        except:
            await ctx.send("❌ VPS instance not found or is no longer available.", ephemeral=True)
            return

        exec_cmd = await asyncio.create_subprocess_exec(
            "docker", "exec", vps["container_id"], "tmate", "-F",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        ssh_session_line = await capture_ssh_session_line(exec_cmd)
        if not ssh_session_line:
            raise Exception("Failed to get tmate session")

        bot.db.update_vps(token, {"tmate_session": ssh_session_line})
        
        embed = discord.Embed(title="HighCore VPS Connection Details", color=discord.Color.blue())
        embed.add_field(name="Username", value=vps["username"], inline=True)
        embed.add_field(name="SSH Password", value=f"||{vps.get('password', 'Not set')}||", inline=True)
        embed.add_field(name="Tmate Session", value=f"```{ssh_session_line}```", inline=False)
        embed.add_field(name="Connection Instructions", value="""
1. Copy the Tmate session command
2. Open your terminal
3. Paste and run the command
4. You will be connected to your HighCore VPS

Or use direct SSH:
```ssh {username}@<server-ip>```
""".format(username=vps["username"]), inline=False)
        
        await ctx.author.send(embed=embed)
        await ctx.send("✅ Connection details sent to your DMs! Use the Tmate command to connect to your HighCore VPS.", ephemeral=True)
        
    except discord.Forbidden:
        await ctx.send("❌ I couldn't send you a DM. Please enable DMs from server members.", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in connect_vps: {e}")
        await ctx.send(f"❌ An error occurred while connecting to the VPS: {str(e)}", ephemeral=True)

@bot.hybrid_command(name='vps_stats', description='Show resource usage for a VPS')
@app_commands.describe(
    vps_id="ID of the VPS to check"
)
async def vps_stats(ctx, vps_id: str):
    """Show resource usage for a VPS"""
    try:
        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps or (vps["created_by"] != str(ctx.author.id) and not has_admin_role(ctx)):
            await ctx.send("❌ VPS not found or you don't have access to it!", ephemeral=True)
            return

        try:
            container = bot.docker_client.containers.get(vps["container_id"])
            if container.status != "running":
                await ctx.send("❌ VPS is not running!", ephemeral=True)
                return

            # Get memory stats
            mem_process = await asyncio.create_subprocess_exec(
                "docker", "exec", vps["container_id"], "free", "-m",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            mem_out, mem_err = await mem_process.communicate()
            if mem_process.returncode == 0:
                mem_lines = mem_out.decode().split('\n')
                if len(mem_lines) > 1:
                    mem_info = mem_lines[1].split()
                    mem_total = int(mem_info[1]) / 1024  # Convert to GB
                    mem_used = int(mem_info[2]) / 1024   # Convert to GB
                    mem_usage = (mem_used / mem_total) * 100
                else:
                    mem_total = mem_used = mem_usage = "Unknown"
            else:
                mem_total = mem_used = mem_usage = "Unknown"
                logger.error(f"Error getting memory stats: {mem_err.decode()}")

            # Get CPU stats
            cpu_process = await asyncio.create_subprocess_exec(
                "docker", "exec", vps["container_id"], "top", "-bn1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            cpu_out, cpu_err = await cpu_process.communicate()
            cpu_usage = "Unknown"
            if cpu_process.returncode == 0:
                for line in cpu_out.decode().split('\n'):
                    if line.startswith("%Cpu(s):"):
                        cpu_usage = float(line.split()[1])  # % user CPU
                        break
            else:
                logger.error(f"Error getting CPU stats: {cpu_err.decode()}")

            # Get disk stats
            disk_process = await asyncio.create_subprocess_exec(
                "docker", "exec", vps["container_id"], "df", "-h", "/data",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            disk_out, disk_err = await disk_process.communicate()
            disk_total = disk_used = disk_usage = "Unknown"
            if disk_process.returncode == 0:
                disk_lines = disk_out.decode().split('\n')
                if len(disk_lines) > 1:
                    disk_info = disk_lines[1].split()
                    disk_total = disk_info[1]
                    disk_used = disk_info[2]
                    disk_usage = disk_info[4]
            else:
                logger.error(f"Error getting disk stats: {disk_err.decode()}")

            embed = discord.Embed(title=f"HighCore VPS Stats - {vps_id}", color=discord.Color.blue())
            embed.add_field(name="Memory", value=f"{mem_used:.2f}GB / {mem_total:.2f}GB ({mem_usage:.1f}%)", inline=True)
            embed.add_field(name="CPU Usage", value=f"{cpu_usage}%", inline=True)
            embed.add_field(name="Disk", value=f"{disk_used} / {disk_total} ({disk_usage})", inline=True)
            embed.add_field(name="Status", value=vps['status'].capitalize(), inline=True)
            
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Error getting VPS stats: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in vps_stats: {e}")
        await ctx.send(f"❌ Error: {str(e)}", ephemeral=True)

@bot.hybrid_command(name='change_ssh_password', description='Change SSH password for a VPS')
@app_commands.describe(
    vps_id="ID of the VPS to change password for"
)
async def change_ssh_password(ctx, vps_id: str):
    """Change SSH password for a VPS"""
    try:
        token, vps = bot.db.get_vps_by_id(vps_id)
        if not vps or (vps["created_by"] != str(ctx.author.id) and not has_admin_role(ctx)):
            await ctx.send("❌ VPS not found or you don't have access to it!", ephemeral=True)
            return

        try:
            container = bot.docker_client.containers.get(vps["container_id"])
            if container.status != "running":
                await ctx.send("❌ VPS is not running!", ephemeral=True)
                return

            new_password = generate_ssh_password()
            success, output = await run_docker_command(
                vps["container_id"],
                ["bash", "-c", f"echo '{vps['username']}:{new_password}' | chpasswd"]
            )
            if not success:
                await ctx.send(f"❌ Failed to change password: {output}", ephemeral=True)
                return

            bot.db.update_vps(token, {"password": new_password})

            embed = discord.Embed(title=f"HighCore VPS Password Changed - {vps_id}", color=discord.Color.green())
            embed.add_field(name="Username", value=vps['username'], inline=True)
            embed.add_field(name="New Password", value=f"||{new_password}||", inline=False)
            
            await ctx.author.send(embed=embed)
            await ctx.send("✅ SSH password updated successfully! Check your DMs for the new password.", ephemeral=True)
        except Exception as e:
            await ctx.send(f"❌ Error changing SSH password: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in change_ssh_password: {e}")
        await ctx.send(f"❌ Error: {str(e)}", ephemeral=True)

@bot.hybrid_command(name='admin_stats', description='Show system statistics (Admin only)')
async def admin_stats(ctx):
    """Show system statistics (Admin only)"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return

    try:
        # Get Docker stats
        containers = bot.docker_client.containers.list(all=True) if bot.docker_client else []
        
        # Get system stats
        stats = bot.system_stats
        
        embed = discord.Embed(title="HighCore System Statistics", color=discord.Color.blue())
        embed.add_field(name="VPS Instances", value=f"Total: {len(bot.db.get_all_vps())}\nRunning: {len([c for c in containers if c.status == 'running'])}", inline=True)
        embed.add_field(name="Docker Containers", value=f"Total: {len(containers)}\nRunning: {len([c for c in containers if c.status == 'running'])}", inline=True)
        embed.add_field(name="CPU Usage", value=f"{stats['cpu_usage']}%", inline=True)
        embed.add_field(name="Memory Usage", value=f"{stats['memory_usage']}% ({stats['memory_used']:.2f}GB / {stats['memory_total']:.2f}GB)", inline=True)
        embed.add_field(name="Disk Usage", value=f"{stats['disk_usage']}% ({stats['disk_used']:.2f}GB / {stats['disk_total']:.2f}GB)", inline=True)
        embed.add_field(name="Network", value=f"Sent: {stats['network_sent']:.2f}MB\nRecv: {stats['network_recv']:.2f}MB", inline=True)
        embed.add_field(name="Container Limit", value=f"{len(containers)}/{bot.db.get_setting('max_containers')}", inline=True)
        embed.add_field(name="Last Updated", value=f"<t:{int(stats['last_updated'])}:R>", inline=True)
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in admin_stats: {e}")
        await ctx.send(f"❌ Error getting system stats: {str(e)}", ephemeral=True)

@bot.hybrid_command(name='system_info', description='Show detailed system information (Admin only)')
async def system_info(ctx):
    """Show detailed system information (Admin only)"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return

    try:
        # System information
        uname = platform.uname()
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        
        # CPU information
        cpu_info = f"""
System: {uname.system}
Node Name: {uname.node}
Release: {uname.release}
Version: {uname.version}
Machine: {uname.machine}
Processor: {uname.processor}
Physical cores: {psutil.cpu_count(logical=False)}
Total cores: {psutil.cpu_count(logical=True)}
CPU Usage: {psutil.cpu_percent()}%
"""
        
        # Memory Information
        svmem = psutil.virtual_memory()
        mem_info = f"""
Total: {svmem.total / (1024**3):.2f}GB
Available: {svmem.available / (1024**3):.2f}GB
Used: {svmem.used / (1024**3):.2f}GB
Percentage: {svmem.percent}%
"""
        
        # Disk Information
        partitions = psutil.disk_partitions()
        disk_info = ""
        for partition in partitions:
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                disk_info += f"""
Device: {partition.device}
  Mountpoint: {partition.mountpoint}
  File system type: {partition.fstype}
  Total Size: {partition_usage.total / (1024**3):.2f}GB
  Used: {partition_usage.used / (1024**3):.2f}GB
  Free: {partition_usage.free / (1024**3):.2f}GB
  Percentage: {partition_usage.percent}%
"""
            except PermissionError:
                continue
        
        # Network information
        net_io = psutil.net_io_counters()
        net_info = f"""
Bytes Sent: {net_io.bytes_sent / (1024**2):.2f}MB
Bytes Received: {net_io.bytes_recv / (1024**2):.2f}MB
"""
        
        embed = discord.Embed(title="Detailed System Information", color=discord.Color.blue())
        embed.add_field(name="System", value=f"Boot Time: {boot_time}", inline=False)
        embed.add_field(name="CPU Info", value=f"```{cpu_info}```", inline=False)
        embed.add_field(name="Memory Info", value=f"```{mem_info}```", inline=False)
        embed.add_field(name="Disk Info", value=f"```{disk_info}```", inline=False)
        embed.add_field(name="Network Info", value=f"```{net_info}```", inline=False)
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in system_info: {e}")
        await ctx.send(f"❌ Error getting system info: {str(e)}", ephemeral=True)

@bot.hybrid_command(name='container_limit', description='Set maximum container limit (Owner only)')
@app_commands.describe(
    max_limit="New maximum container limit"
)
async def set_container_limit(ctx, max_limit: int):
    """Set maximum container limit (Owner only)"""
    if ctx.author.id != 1210291131301101618:  # Only the owner can set limit
        await ctx.send("❌ Only the owner can set container limit!", ephemeral=True)
        return
    
    if max_limit < 1 or max_limit > 1000:
        await ctx.send("❌ Container limit must be between 1 and 1000", ephemeral=True)
        return
    
    bot.db.set_setting('max_containers', max_limit)
    await ctx.send(f"✅ Maximum container limit set to {max_limit}", ephemeral=True)

@bot.hybrid_command(name='cleanup_vps', description='Cleanup inactive VPS instances (Admin only)')
async def cleanup_vps(ctx):
    """Cleanup inactive VPS instances (Admin only)"""
    if not has_admin_role(ctx):
        await ctx.send("❌ You must be an admin to use this command!", ephemeral=True)
        return

    try:
        cleanup_count = 0
        
        for token, vps in list(bot.db.get_all_vps().items()):
            try:
                container = bot.docker_client.containers.get(vps['container_id'])
                if container.status != 'running':
                    container.stop()
                    container.remove()
                    bot.db.remove_vps(token)
                    cleanup_count += 1
            except docker.errors.NotFound:
                bot.db.remove_vps(token)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up VPS {vps['vps_id']}: {e}")
                continue
        
        if cleanup_count > 0:
            await ctx.send(f"✅ Cleaned up {cleanup_count} inactive VPS instances!")
        else:
            await ctx.send("ℹ️ No inactive VPS instances found to clean up.")
    except Exception as e:
        logger.error(f"Error in cleanup_vps: {e}")
        await ctx.send(f"❌ Error during cleanup: {str(e)}", ephemeral=True)

@bot.hybrid_command(name='vps_shell', description='Get shell access to your VPS')
@app_commands.describe(
    vps_id="ID of the VPS to access"
)
async def vps_shell(ctx, vps_id: str):
    """Get shell access
