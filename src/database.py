import sqlite3
import json
import uuid
import os
from datetime import datetime

DB_PATH = "nexus_router.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def setup_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # --- Router & Load Balancer Tables ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS router_workers (
            worker_id TEXT PRIMARY KEY,
            worker_name TEXT,
            supported_models TEXT,
            status TEXT,
            last_heartbeat TEXT,
            specs TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS router_tasks (
            task_id TEXT PRIMARY KEY,
            model_name TEXT,
            prompt TEXT,
            status TEXT,
            assigned_worker_id TEXT,
            result_content TEXT,
            created_at TEXT,
            started_at TEXT,
            updated_at TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS router_bottleneck_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            worker_id TEXT,
            prompt TEXT,
            model_name TEXT,
            worker_specs TEXT,
            processing_time_sec REAL,
            created_at TEXT
        )
    ''')
        
    conn.commit()
    conn.close()

class DatabaseManager:
    @staticmethod
    def router_upsert_worker(worker_id: str, worker_name: str, supported_models: str, specs: str, status: str = "ONLINE"):
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT worker_id FROM router_workers WHERE worker_id = ?", (worker_id,))
        if cursor.fetchone():
            cursor.execute('''
                UPDATE router_workers 
                SET worker_name=?, supported_models=?, specs=?, status=?, last_heartbeat=?
                WHERE worker_id=?
            ''', (worker_name, supported_models, specs, status, now_str, worker_id))
        else:
            cursor.execute('''
                INSERT INTO router_workers (worker_id, worker_name, supported_models, status, last_heartbeat, specs)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (worker_id, worker_name, supported_models, status, now_str, specs))
        conn.commit()
        conn.close()

    @staticmethod
    def router_update_worker_heartbeat(worker_id: str):
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE router_workers SET last_heartbeat=?, status='ONLINE' WHERE worker_id=?", (now_str, worker_id))
        conn.commit()
        conn.close()

    @staticmethod
    def router_set_worker_status(worker_id: str, status: str):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE router_workers SET status=? WHERE worker_id=?", (status, worker_id))
        conn.commit()
        conn.close()

    @staticmethod
    def router_get_workers() -> list:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM router_workers")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def router_create_task(model_name: str, prompt: str) -> str:
        task_id = str(uuid.uuid4())
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO router_tasks (task_id, model_name, prompt, status, created_at, updated_at)
            VALUES (?, ?, ?, 'PENDING', ?, ?)
        ''', (task_id, model_name, prompt, now_str, now_str))
        conn.commit()
        conn.close()
        return task_id

    @staticmethod
    def router_get_pending_tasks() -> list:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM router_tasks WHERE status = 'PENDING' ORDER BY created_at ASC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def router_get_processing_tasks() -> list:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM router_tasks WHERE status = 'PROCESSING'")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def router_assign_task(task_id: str, worker_id: str):
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE router_tasks 
            SET status='PROCESSING', assigned_worker_id=?, started_at=?, updated_at=?
            WHERE task_id=?
        ''', (worker_id, now_str, now_str, task_id))
        conn.commit()
        conn.close()

    @staticmethod
    def router_complete_task(task_id: str, result_content: str):
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE router_tasks 
            SET status='COMPLETED', result_content=?, updated_at=?
            WHERE task_id=?
        ''', (result_content, now_str, task_id))
        conn.commit()
        conn.close()

    @staticmethod
    def router_fail_task(task_id: str, new_status: str = "FAILED"):
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE router_tasks 
            SET status=?, updated_at=?
            WHERE task_id=?
        ''', (new_status, now_str, task_id))
        conn.commit()
        conn.close()

    @staticmethod
    def router_requeue_task(task_id: str):
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE router_tasks 
            SET status='PENDING', assigned_worker_id=NULL, started_at=NULL, updated_at=?
            WHERE task_id=?
        ''', (now_str, task_id))
        conn.commit()
        conn.close()

    @staticmethod
    def router_log_bottleneck(task_id: str, worker_id: str, prompt: str, model_name: str, worker_specs: str, processing_time_sec: float):
        now_str = datetime.now().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO router_bottleneck_logs (task_id, worker_id, prompt, model_name, worker_specs, processing_time_sec, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (task_id, worker_id, prompt, model_name, worker_specs, processing_time_sec, now_str))
        conn.commit()
        conn.close()
