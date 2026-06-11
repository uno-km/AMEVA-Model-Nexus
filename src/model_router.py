import asyncio
import uuid
import time
import json
import sqlite3
from datetime import datetime
from database import DatabaseManager, setup_db, get_connection

class WorkerNode:
    """
    Represent a worker that can process LLM requests.
    In a real system, this would make HTTP/Docker calls.
    Here we simulate it using asyncio.sleep.
    """
    def __init__(self, worker_name: str, supported_models: list[str], specs: dict):
        self.worker_id = str(uuid.uuid4())
        self.worker_name = worker_name
        self.supported_models = supported_models
        self.specs = specs
        self.specs_str = json.dumps(specs, ensure_ascii=False)
        self.running_task_id = None
        self.is_crashed = False  # Simulation flag
        self.delay_multiplier = 1.0  # Simulation flag for bottleneck

    async def register(self):
        DatabaseManager.router_upsert_worker(
            worker_id=self.worker_id,
            worker_name=self.worker_name,
            supported_models=",".join(self.supported_models),
            specs=self.specs_str,
            status="ONLINE"
        )

    async def heartbeat_loop(self):
        """Periodically update heartbeat in DB"""
        while True:
            if not self.is_crashed:
                DatabaseManager.router_update_worker_heartbeat(self.worker_id)
            await asyncio.sleep(5)

    async def process_task(self, task_id: str, prompt: str):
        """Simulate processing a task"""
        self.running_task_id = task_id
        DatabaseManager.router_set_worker_status(self.worker_id, "BUSY")
        
        # Simulate processing time based on prompt length
        process_time = (len(prompt) * 0.1) * self.delay_multiplier
        
        # Simulate work
        elapsed = 0
        while elapsed < process_time:
            if self.is_crashed:
                # If crashed during processing, we just stop reporting
                return
            await asyncio.sleep(1)
            elapsed += 1

        # Finished successfully
        result = f"[{self.worker_name}] Answer for: '{prompt}'"
        DatabaseManager.router_complete_task(task_id, result)
        DatabaseManager.router_set_worker_status(self.worker_id, "ONLINE")
        self.running_task_id = None

class ModelRouter:
    """
    Central router that assigns pending tasks to online workers
    and monitors for bottlenecks and crashes.
    """
    def __init__(self, watchdog_interval=5.0, offline_timeout=15.0, task_timeout=30.0):
        self.watchdog_interval = watchdog_interval
        self.offline_timeout = offline_timeout
        self.task_timeout = task_timeout  # Using 30s instead of 5m for simulation

    async def dispatcher_loop(self):
        """Continuously assigns PENDING tasks to ONLINE workers"""
        print("[Router] Dispatcher Loop Started")
        while True:
            pending_tasks = DatabaseManager.router_get_pending_tasks()
            if pending_tasks:
                workers = DatabaseManager.router_get_workers()
                online_workers = [w for w in workers if w['status'] == 'ONLINE']
                
                for task in pending_tasks:
                    task_id = task['task_id']
                    req_model = task['model_name']
                    
                    # Find a suitable worker
                    suitable_worker = next((w for w in online_workers if req_model in w['supported_models']), None)
                    
                    if suitable_worker:
                        worker_id = suitable_worker['worker_id']
                        print(f"[Router] Assigning Task {task_id[:8]} to Worker {suitable_worker['worker_name']}")
                        
                        # Assign in DB
                        DatabaseManager.router_assign_task(task_id, worker_id)
                        DatabaseManager.router_set_worker_status(worker_id, "BUSY")
                        
                        # In a real system, we would trigger an API call to the worker here.
                        # Since we are simulating, we will let the global simulation loop trigger it,
                        # or we just assume the worker pulls it. For this simulation, we emit an event.
                        # We will use a simple global dictionary to find the worker instance and trigger it.
                        worker_instance = GLOBAL_WORKERS.get(worker_id)
                        if worker_instance:
                            asyncio.create_task(worker_instance.process_task(task_id, task['prompt']))
                        
                        # Remove from online pool to prevent double assignment
                        online_workers.remove(suitable_worker)

            await asyncio.sleep(2)

    async def watchdog_loop(self):
        """Monitors for crashed workers and bottlenecked tasks"""
        print("[Watchdog] Watchdog Loop Started")
        while True:
            now = datetime.now()
            
            # 1. Check for Crashed Workers (Offline Timeout)
            workers = DatabaseManager.router_get_workers()
            for w in workers:
                # Parse last_heartbeat
                if w['last_heartbeat']:
                    last_hb = datetime.fromisoformat(w['last_heartbeat'])
                    diff_sec = (now - last_hb).total_seconds()
                    if diff_sec > self.offline_timeout and w['status'] != 'OFFLINE':
                        print(f"[Watchdog] [ALERT] Worker {w['worker_name']} timed out (no heartbeat for {diff_sec:.1f}s). Marking OFFLINE.")
                        DatabaseManager.router_set_worker_status(w['worker_id'], 'OFFLINE')

            # 2. Check for Bottlenecked / Crashed Tasks
            processing_tasks = DatabaseManager.router_get_processing_tasks()
            for task in processing_tasks:
                started_at = datetime.fromisoformat(task['started_at'])
                worker_id = task['assigned_worker_id']
                diff_sec = (now - started_at).total_seconds()
                
                # Check if the assigned worker is now OFFLINE
                assigned_worker = next((w for w in workers if w['worker_id'] == worker_id), None)
                
                if assigned_worker and assigned_worker['status'] == 'OFFLINE':
                    print(f"[Watchdog] [RE-QUEUE] Task {task['task_id'][:8]} was assigned to crashed worker {assigned_worker['worker_name']}. Re-queueing.")
                    DatabaseManager.router_requeue_task(task['task_id'])
                    continue

                # Check for Bottleneck (Timeout)
                if diff_sec > self.task_timeout:
                    print(f"[Watchdog] [BOTTLENECK] Task {task['task_id'][:8]} taking too long ({diff_sec:.1f}s). Cancelling & Logging.")
                    
                    worker_specs = assigned_worker['specs'] if assigned_worker else "{}"
                    
                    # 1. Log detailed info
                    DatabaseManager.router_log_bottleneck(
                        task_id=task['task_id'],
                        worker_id=worker_id,
                        prompt=task['prompt'],
                        model_name=task['model_name'],
                        worker_specs=worker_specs,
                        processing_time_sec=diff_sec
                    )
                    
                    # 2. Mark as TIMEOUT (or we could re-queue, but let's cancel as requested)
                    DatabaseManager.router_fail_task(task['task_id'], new_status="TIMEOUT")
                    
                    # Free up the worker just in case it's still alive but stuck
                    if assigned_worker and assigned_worker['status'] == 'BUSY':
                        DatabaseManager.router_set_worker_status(worker_id, "ONLINE")

            await asyncio.sleep(self.watchdog_interval)

# For simulation, we keep a global ref to worker objects
GLOBAL_WORKERS = {}

async def simulation_scenario():
    # Setup DB
    setup_db()
    print("Database Initialized.")
    
    # Clean up any previous state
    conn = get_connection() # workaround to run raw sql
    cursor = conn.cursor()
    cursor.execute("DELETE FROM router_tasks")
    cursor.execute("DELETE FROM router_workers")
    cursor.execute("DELETE FROM router_bottleneck_logs")
    conn.commit()
    conn.close()

    router = ModelRouter(watchdog_interval=2.0, offline_timeout=10.0, task_timeout=15.0)
    
    # 1. Start Router Loops
    asyncio.create_task(router.dispatcher_loop())
    asyncio.create_task(router.watchdog_loop())

    # 2. Create Workers
    w1 = WorkerNode("PC_Docker_8B", ["llama3-8b"], {"cpu": "i7", "vram": "12GB", "type": "desktop"})
    w2 = WorkerNode("Galaxy_S24", ["gemma-2b"], {"cpu": "Snapdragon 8 Gen 3", "ram": "12GB", "type": "mobile"})
    w3 = WorkerNode("Slow_Old_Laptop", ["llama3-8b"], {"cpu": "i3", "vram": "4GB", "type": "laptop"})

    for w in [w1, w2, w3]:
        await w.register()
        GLOBAL_WORKERS[w.worker_id] = w
        asyncio.create_task(w.heartbeat_loop())
    
    print("\n--- [Scenario 1] Normal processing ---")
    DatabaseManager.router_create_task("gemma-2b", "What is Python?")
    DatabaseManager.router_create_task("llama3-8b", "Explain async.")
    
    await asyncio.sleep(5)
    
    print("\n--- [Scenario 2] Worker Crash & Re-queue ---")
    print("PC_Docker_8B crashes suddenly...")
    w1.is_crashed = True # Stop heartbeat and processing
    
    # Submit new task that needs 8B. It might get assigned to w1 if it hasn't timed out yet, 
    # or to w3. Let's submit two to guarantee one gets stuck on w1 if it was free.
    t3 = DatabaseManager.router_create_task("llama3-8b", "How to fix a bug?")
    
    await asyncio.sleep(15) # Wait for watchdog to notice crash and re-queue
    
    print("\n--- [Scenario 3] Bottleneck (Timeout) & Detailed Logging ---")
    print("Slow_Old_Laptop starts a massive task but is too slow...")
    w3.delay_multiplier = 100.0 # Make it super slow
    t4 = DatabaseManager.router_create_task("llama3-8b", "Write a 50 page essay about the universe.")
    
    await asyncio.sleep(20) # Wait for watchdog to timeout the task
    
    print("\n--- Final Database Checks ---")
    pending = DatabaseManager.router_get_pending_tasks()
    print(f"Pending tasks remaining: {len(pending)}")
    
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM router_tasks WHERE task_id=?", (t4,))
    t4_rec = cursor.fetchone()
    print(f"Task 4 Status: {t4_rec['status']}")
    
    cursor.execute("SELECT * FROM router_bottleneck_logs")
    logs = cursor.fetchall()
    print(f"Bottleneck logs count: {len(logs)}")
    if logs:
        print("Latest Bottleneck Log:")
        print(dict(logs[-1]))
    conn.close()

    print("Simulation Complete. Exiting.")

if __name__ == "__main__":
    asyncio.run(simulation_scenario())
