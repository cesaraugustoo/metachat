import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

class DesignDatabase:
    def __init__(self, db_path: str = "designs.db"):
        """Initialize the design database with the specified path."""
        self.db_path = db_path
        self._create_tables_if_not_exist()
        
    def _create_tables_if_not_exist(self):
        """Create the database tables if they don't already exist."""
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create designs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS designs (
                id TEXT PRIMARY KEY,
                design_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parameters TEXT NOT NULL,  -- JSON string of parameters
                gds_file_path TEXT,
                success INTEGER NOT NULL,
                description TEXT,
                user_id TEXT
            )
            ''')
            
            # Create files table for associated files (plots, etc.)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS design_files (
                id TEXT PRIMARY KEY,
                design_id TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (design_id) REFERENCES designs (id)
            )
            ''')
            
            conn.commit()
    
    def save_design(self, 
                   design_type: str, 
                   parameters: Dict[str, Any], 
                   gds_file_path: Optional[str] = None,
                   success: bool = True,
                   description: str = "",
                   user_id: Optional[str] = None,
                   associated_files: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Save a design to the database.
        
        Args:
            design_type: Type of design (e.g., "metalens", "deflector")
            parameters: Dictionary of design parameters
            gds_file_path: Path to the GDS file (if generated)
            success: Whether the design completed successfully
            description: Optional description of the design
            user_id: Optional ID of the user who created the design
            associated_files: List of dictionaries with file info: 
                             {"file_type": "...", "file_path": "...", "description": "..."}
                             
        Returns:
            The ID of the saved design
        """
        design_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert the design
            cursor.execute(
                "INSERT INTO designs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    design_id,
                    design_type,
                    timestamp,
                    json.dumps(parameters),
                    gds_file_path,
                    1 if success else 0,
                    description,
                    user_id
                )
            )
            
            # Insert associated files if any
            if associated_files:
                for file_info in associated_files:
                    file_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO design_files VALUES (?, ?, ?, ?, ?)",
                        (
                            file_id,
                            design_id,
                            file_info.get("file_type", "unknown"),
                            file_info.get("file_path", ""),
                            file_info.get("description", "")
                        )
                    )
                    
            conn.commit()
        return design_id

    def get_design(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a design from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM designs WHERE id = ?", (design_id,))
            row = cursor.fetchone()
            if row:
                design = dict(row)
                design["parameters"] = json.loads(design["parameters"])
                
                # Get associated files
                cursor.execute("SELECT * FROM design_files WHERE design_id = ?", (design_id,))
                design["associated_files"] = [dict(f) for f in cursor.fetchall()]
                return design
        return None
